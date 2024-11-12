from dialoguekit.core.annotated_utterance import AnnotatedUtterance
from dialoguekit.core.dialogue_act import DialogueAct
from dialoguekit.core.utterance import Utterance
from dialoguekit.participant.agent import Agent
from dialoguekit.participant.participant import DialogueParticipant

from flask import jsonify
from rapidfuzz import fuzz
from nltk import ngrams

import mysql.connector
from mysql.connector import Error

from random import choice
import re

from collections import Counter

import joblib


class MusicAgent(Agent):
    def __init__(self, id: str):
        """Initialize the MusicBot agent with natural language model and DB connection."""
        super().__init__(id)
        
        # Load the trained natural language model for intent classification
        self.natural_model = joblib.load('app/model/intent_classifier_model.pkl')
        self.vectorizer = joblib.load('app/model/vectorizer.pkl')
        
        # Database connection configuration
        db_config = {
            'user': 'root',      # Use your actual DB user
            'password': 'musicpwd',   # Use your actual DB password
            'host': 'localhost',      # Host configuration
            'database': 'MusicDB',    # Database name
        }

        # Establishing the connection
        self.db_conn = self.connect_to_db(db_config)
        self.tracks_data = self.fetch_all_tracks()
        self.artists_data = self.extract_unique_artists()
        self.top_songs = None 

    def connect_to_db(self, db_config):
        """Connect to MySQL database."""
        try:
            conn = mysql.connector.connect(**db_config)
            if conn.is_connected():
                print('Connected to MySQL database')
            return conn
        except Error as err:
            print(f"Database connection error: {err}")
            return None
        
    ######################
    ### INITIALIZATION ###
    ######################
        
    def fetch_all_tracks(self):
        """Fetch all tracks and their attributes once upon initialization."""
        try:
            cursor = self.db_conn.cursor(dictionary=True)
            cursor.execute("SELECT id, track_name, artist_name, genre, popularity FROM Tracks")
            tracks = cursor.fetchall()
        except mysql.connector.Error as e:
            print(f"Database error during initialization: {e}")
            tracks = []
        finally:
            cursor.close()
        return tracks
    
    def extract_unique_artists(self):
        """Extract unique artists from tracks data and handle multiple artists per track."""
        artists = set()  # Using a set to avoid duplicates
        for track in self.tracks_data:
            artist_names = track["artist_name"].split(';')  # Split multiple artists
            for artist in artist_names:
                artists.add(artist.strip().lower())  # Add artist, stripping whitespace and lowercasing
        return list(artists)  # Convert set back to a list for further use
    
    #####################
    ### MODEL METHODS ###
    #####################

    def preprocess_input(self, text):
        """Preprocess user input (e.g., lowercasing, removing punctuation)."""
        return text.lower()

    def interpret_input(self, user_input):
        """Use the SVM model to interpret the intent from user input."""
        processed_input = self.preprocess_input(user_input)
        vectorized_input = self.vectorizer.transform([processed_input])  # Vectorize the input
        
        # Make the prediction
        prediction = self.natural_model.predict(vectorized_input)
        return prediction
    
    #########################
    ### CHATBOT RESPONSES ###
    #########################
    
    def welcome(self) -> None:
        """Sends the agent's welcome message."""
        utterance = AnnotatedUtterance(
            "Hello, I'm MusicBot. What can I help you with?",
            participant=DialogueParticipant.AGENT,
        )
        self._dialogue_connector.register_agent_utterance(utterance)

    def goodbye(self) -> None:
        """Sends the agent's goodbye message."""
        utterance = AnnotatedUtterance(
            "It was nice talking to you. Bye",
            dialogue_acts=[DialogueAct(intent=self.stop_intent)],
            participant=DialogueParticipant.AGENT,
        )
        self._dialogue_connector.register_agent_utterance(utterance)

    def receive_utterance(self, utterance: Utterance) -> None:
        """Handle incoming user input based on predicted intent."""
        user_input = utterance.text
        intent = self.interpret_input(user_input)

        # Dispatch to the appropriate method based on intent
        if self.top_songs is None:
            if intent == "add":
                response = self.find_possible_songs(user_input)
                utterance = AnnotatedUtterance(
                    response,
                    participant=DialogueParticipant.AGENT,
                )
            elif intent == "remove":
                response = self.remove_songs(user_input)
                utterance = AnnotatedUtterance(
                    response,
                    participant=DialogueParticipant.AGENT,
                )
            elif intent == "view":
                response = self.view_playlist()
                utterance = AnnotatedUtterance(
                    response,
                    participant=DialogueParticipant.AGENT,
                )
            elif intent == "clear":
                response = self.clear_playlist()
                utterance = AnnotatedUtterance(
                    response,
                    participant=DialogueParticipant.AGENT,
                )
            elif intent == "recommend":
                response = self.recommend_songs()
                utterance = AnnotatedUtterance(
                    response,
                    participant=DialogueParticipant.AGENT,
                )
            elif intent == "functionality":
                response = self.functionality()
                utterance = AnnotatedUtterance(
                    response,
                    participant=DialogueParticipant.AGENT,
                )
            elif intent == "greeting":
                utterance = self.welcome()
            elif intent == "exit":
                utterance = self.goodbye()
            else:
                response = "I'm sorry, I didn't understand you. Please try again."
                utterance = AnnotatedUtterance(
                    response,
                    participant=DialogueParticipant.AGENT,
                )
        else:
            if user_input == "exit":
                self.top_songs = None  # Reset top_songs after exiting
                response = "Not adding song to playlist. Exiting."
                utterance = AnnotatedUtterance(
                    response,
                    participant=DialogueParticipant.AGENT,
                )
            else:
                try:
                    user_input = int(user_input)
                    response = self.select_song(user_input)
                    utterance = AnnotatedUtterance(
                        response,
                        participant=DialogueParticipant.AGENT,
                    )
                except ValueError:
                    response = f"Please choose a number between 1 and {len(self.top_songs)}, or enter 'exit' to exit."
                    utterance = AnnotatedUtterance(
                        response,
                        participant=DialogueParticipant.AGENT,
                    )
        self._dialogue_connector.register_agent_utterance(utterance)

    def fetch_track_from_db(self, song_name): # why u here
        """Fetch track details from the database based on the song name."""
        try:
            cursor = self.db_conn.cursor(dictionary=True)
            query = "SELECT * FROM Tracks WHERE track_name = %s"
            cursor.execute(query, (song_name,))
            result = cursor.fetchone()
            cursor.close()
            return result
        except Error as e:
            print(f"Error fetching track from database: {e}")
            return None
        
    def find_possible_songs(self, user_input):
        """Add song by extracting possible artists and finding top track matches."""

        # Step 1: Generate n-grams from the input sentence to identify possible artists
        split_sentence = user_input.lower().split()
        possible_artists = []
        
        # Generate n-grams (1 to 3 words) from the input sentence for artist name matching
        for n in range(1, 4):
            ngram_list = [' '.join(ngram) for ngram in ngrams(split_sentence, n)]
            possible_artists.extend(ngram_list)

        # Filter possible artists to those that match known artists in artists_data
        matching_artists = [
            artist for artist in possible_artists if any(
                fuzz.token_sort_ratio(artist, known_artist) > 70 for known_artist in self.artists_data
            )
        ]
        
        # Step 2: Calculate fuzz scores for each track based on track name and matching artists
        scored_tracks = []
        for track in self.tracks_data:
            track_score = fuzz.token_sort_ratio(track["track_name"].lower(), user_input.lower())
            
            # Check if any of the track's artists match known artists
            track_artists = [artist.strip().lower() for artist in track["artist_name"].split(';')]
            artist_score_boost = any(
                artist in matching_artists for artist in track_artists
            )
            
            # Apply artist boost and popularity adjustment
            adjusted_score = track_score
            if artist_score_boost:
                adjusted_score += 30  # Increase score if artist matches
            adjusted_score += track["popularity"] / 10  # Adjust for popularity, scaled down

            scored_tracks.append((track, adjusted_score))
        
        # Step 3: Sort by adjusted score and take the top 10 tracks
        top_tracks = sorted(scored_tracks, key=lambda x: x[1], reverse=True)[:10]
        self.top_songs = [track['id'] for track, _ in top_tracks]  # Store the top 10 song IDs
        
        # Format the response to include the top recommended tracks
        response = "Based on your input, here are the top matching tracks:\n"
        response += " -- ".join(
            f"{i+1}. {track['track_name']} by {track['artist_name']} (Score: {score:.1f}, Popularity: {track['popularity']})"
            for i, (track, score) in enumerate(top_tracks)
        )
        response += ". To add a song to your playlist, please enter the number corresponding to the song you want to add. If you want to exit, enter 'exit'."
        
        return response
    
    def recommend_songs(self):
        """Recommend songs based on the genres of tracks in the playlist."""

        try:
            cursor = self.db_conn.cursor(dictionary=True)

            # Step 1: Fetch all track IDs in the playlist and join with Tracks table to get genres
            query = """
                SELECT Tracks.genre 
                FROM Playlist 
                JOIN Tracks ON Playlist.trackID = Tracks.id
            """
            cursor.execute(query)
            playlist_genres_raw = [row['genre'] for row in cursor.fetchall()]

            # Split genres by semicolon and flatten into a list of individual genres
            playlist_genres = []
            for genre_string in playlist_genres_raw:
                playlist_genres.extend([genre.strip() for genre in genre_string.split(';')])

            # Check if the playlist is empty
            if not playlist_genres:
                return "Your playlist is empty. Add some songs to get recommendations!"

            # Step 2: Find the two most common genres
            genre_counts = Counter(playlist_genres)
            top_genres = [genre for genre, _ in genre_counts.most_common(3)]

            print(top_genres)

            # Step 3: Fetch recommendations for each genre, excluding songs already in the playlist
            recommendations = []

            for genre in top_genres:
                query = """
                    SELECT id, track_name, artist_name, popularity 
                    FROM Tracks 
                    WHERE genre LIKE %s AND id NOT IN (
                        SELECT trackID FROM Playlist
                    )
                    ORDER BY popularity DESC 
                    LIMIT 5
                """
                cursor.execute(query, (f"%{genre}%",))
                results = cursor.fetchall()
                recommendations.extend(results)

            # Limit to 15 recommendations if more than 3 genres exist
            recommendations = recommendations[:15]

            self.top_songs = [rec['id'] for rec in recommendations]

            # Format recommendations as strings for response
            recommendation_list = " -- ".join(
                f"{i+1}. {rec['track_name']} by {rec['artist_name']} (Popularity: {rec['popularity']})" for i, rec in enumerate(recommendations)
            )
            response = f"Your playlist is based on the following genres: {', '.join(top_genres)}. Based on your playlist, here are some song recommendations: {recommendation_list}"
            response += ". To add a song to your playlist, please enter the number corresponding to the song you want to add. If you want to exit, enter 'exit'."

        except mysql.connector.Error as e:
            response = "There was an error fetching recommendations. Please try again."
            print(f"Database error: {e}")
        finally:
            cursor.close()

        return response
    
    def select_song(self, choice):
        """Select a song from the top_songs list by index and add it to the playlist."""
        if self.top_songs and 1 <= int(choice) <= len(self.top_songs):
            # Get the selected song ID based on user input (1-based index)
            selected_song_id = self.top_songs[choice - 1]

            # Add the selected song to the playlist in the database
            try:
                cursor = self.db_conn.cursor()
                cursor.execute("INSERT INTO Playlist (trackID) VALUES (%s)", (selected_song_id,))
                self.db_conn.commit()
                response = "Song successfully added to your playlist."
                self.top_songs = None  # Reset top_songs after adding a song
            except mysql.connector.Error as e:
                response = "There was an error adding the song to your playlist. Please try again."
                print(f"Database error: {e}")
            finally:
                cursor.close()
        else:
            response = f"Please choose a number between 1 and {len(self.top_songs)}."
            
        return response

    def remove_songs(self, user_input):
        """Remove the top-matching song from the playlist based on the user's input."""

        # Step 1: Generate n-grams from the input sentence to identify possible artists
        split_sentence = user_input.lower().split()
        possible_artists = []
        
        # Generate n-grams (1 to 3 words) from the input sentence for artist name matching
        for n in range(1, 4):
            ngram_list = [' '.join(ngram) for ngram in ngrams(split_sentence, n)]
            possible_artists.extend(ngram_list)

        try:
            # Step 2: Fetch all tracks in the playlist with their details
            cursor = self.db_conn.cursor(dictionary=True)
            query = """
                SELECT Tracks.id, Tracks.track_name, Tracks.artist_name 
                FROM Playlist 
                JOIN Tracks ON Playlist.trackID = Tracks.id
            """
            cursor.execute(query)
            playlist_tracks = cursor.fetchall()
            
            # Check if the playlist is empty
            if not playlist_tracks:
                return "Your playlist is empty. There are no songs to remove."

            # Step 3: Calculate fuzz scores for each track in the playlist
            scored_tracks = []
            for track in playlist_tracks:
                track_score = fuzz.token_sort_ratio(user_input.lower(), track["track_name"].lower())
                
                # Check if any of the track's artists match known artists
                track_artists = [artist.strip().lower() for artist in track["artist_name"].split(';')]
                artist_score_boost = any(
                    fuzz.token_sort_ratio(artist, possible_artist) > 70 for artist in track_artists for possible_artist in possible_artists
                )
                
                # Apply artist boost
                adjusted_score = track_score
                if artist_score_boost:
                    adjusted_score += 30  # Increase score if artist matches

                scored_tracks.append((track, adjusted_score))
            
            # Step 4: Sort by adjusted score and select the top track
            best_match = max(scored_tracks, key=lambda x: x[1], default=None)

            if best_match and best_match[1] > 60:  # Threshold for match confidence
                track_to_remove = best_match[0]
                # Remove the selected track from the playlist
                cursor.execute("DELETE FROM Playlist WHERE trackID = %s", (track_to_remove['id'],))
                self.db_conn.commit()
                response = f"Removed {track_to_remove['track_name']} by {track_to_remove['artist_name']} from your playlist."
            else:
                response = "No matching song found in your playlist. Please try again with a different title or artist."

        except mysql.connector.Error as e:
            response = "There was an error removing the song from your playlist. Please try again."
            print(f"Database error: {e}")
        finally:
            cursor.close()

        return response
    
    def clear_playlist(self):
        """Clear the playlist."""
        try:
            cursor = self.db_conn.cursor()
            query = "DELETE FROM Playlist"
            cursor.execute(query)
            self.db_conn.commit()
            response = "Playlist cleared. All songs removed"
        except Error as e:
            response = "There was an error clearing the playlist. Please try again."
            print(f"Database error: {e}")
        finally:
            cursor.close()
        return response
    
    def update_frontend_playlist(self, playlist):
        """Update the front-end dynamically by sending the playlist data."""
        try:
            # Assuming `self.db_conn` is being shared with Flask
            cursor = self.db_conn.cursor(dictionary=True)
            
            # Prepare data for the front end
            playlist_data = [
                {
                    "track_name": song["track_name"],
                    "artist_name": song["artist_name"],
                    "album_name": song["album_name"],
                    "genre": song["genre"],
                    "popularity": song["popularity"],
                }
                for song in playlist
            ]

            # Send data via a Flask API (ensure the endpoint exists)
            return jsonify(playlist_data)

        except Exception as e:
            print(f"Error updating front end: {e}")

    def view_playlist(self):
        """View the playlist."""
        songs = []
        try:
            cursor = self.db_conn.cursor(dictionary=True)
            query = "SELECT id, track_name, artist_name, album_name, genre, popularity FROM Tracks WHERE id IN (SELECT trackID FROM Playlist)"
            cursor.execute(query)
            songs = cursor.fetchall()
        except Error as e:
            print(f"Error viewing playlist: {e}")
        if not songs:
            response = "Your playlist is empty."
            return response
        playlist_info = "Your playlist contains the following songs: "
        for song in songs:
            playlist_info += f"{song['track_name']} by {song['artist_name']} from the album {song['album_name']}. Genre: {song['genre']}. Popularity: {song['popularity']}"
            if song != songs[-1]:
                playlist_info += " -- "
        response = playlist_info

        # Send playlist data to the front end
        self.update_frontend_playlist(songs)
        return response
    
    def functionality(self):
        """What can you do?"""
        response = "I'm a virtual assistant that helps you find music. I can add music to your playlist, search for music, and even tell you about the artists and albums you like."
        return response