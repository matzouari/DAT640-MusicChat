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

        self.genre_model = joblib.load('app/model/genre_intent_classifier_model.pkl')
        self.genre_vectorizer = joblib.load('app/model/genre_vectorizer.pkl')

        self.question_classifier = joblib.load("app/model/question_classifier_model.pkl")
        
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
        self.albums_data = self.fetch_all_albums()
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
    
    def fetch_all_albums(self):
        """Retrieve all albums with their associated artists from the database."""
        try:
            cursor = self.db_conn.cursor(dictionary=True)
            cursor.execute("SELECT DISTINCT album_name, artist_name FROM Tracks")
            albums = [(row['album_name'], row['artist_name']) for row in cursor.fetchall()]
            cursor.close()
            return albums
        except Exception as e:
            print(f"Database error while fetching albums and artists: {e}")
            return []
    
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
            elif intent == "create":
                response = self.create_playlist(user_input)
                utterance = AnnotatedUtterance(
                    response,
                    participant=DialogueParticipant.AGENT,
                )
            elif intent == "question":
                response = self.answer_question(user_input)
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
                response = self.select_song(user_input)
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
                adjusted_score += 30 # Increase score if artist matches
            adjusted_score += track["popularity"] / 10  # Adjust for popularity, scaled down

            scored_tracks.append((track, adjusted_score))
        
        # Step 3: Sort by adjusted score and take the top 10 tracks
        top_tracks = sorted(scored_tracks, key=lambda x: x[1], reverse=True)[:10]
        self.top_songs = [track['id'] for track, _ in top_tracks]  # Store the top 10 song IDs
        
        # Format the response to include the top recommended tracks
        response = "Based on your input, here are the top matching tracks: "
        response += " -- ".join(
            f"{i+1}. {track['track_name']} by {track['artist_name']}. Popularity: {track['popularity']}"
            for i, (track, score) in enumerate(top_tracks)
        )
        response += ". To add a song to your playlist, please enter the number corresponding to the song you want to add. For multiple songs, enter a comma-separated list of numbers. If you want to exit, enter 'exit'."
        
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
            response += ". To add a song to your playlist, please enter the number corresponding to the song you want to add. For multiple songs, enter a comma-separated list of numbers. If you want to exit, enter 'exit'."

        except mysql.connector.Error as e:
            response = "There was an error fetching recommendations. Please try again."
            print(f"Database error: {e}")
        finally:
            cursor.close()

        return response
    
    def select_song(self, choice):
        """Select one or more songs from the top_songs list based on user input."""
        if not self.top_songs:
            return "No songs available for selection. Please request recommendations or matches first."

        # Step 1: Parse the choice into a list of indices
        try:
            indices = [int(num.strip()) - 1 for num in choice.split(',')]  # Convert 1-based input to 0-based index
        except ValueError:
            return "Invalid input. Please specify song numbers separated by commas (e.g., '1, 3, 7')."

        # Step 2: Validate indices and collect selected songs
        selected_song_ids = []
        for idx in indices:
            if 0 <= idx < len(self.top_songs):
                selected_song_ids.append(self.top_songs[idx])
            else:
                print(f"Skipping invalid song number {idx + 1}.")  # Inform about out-of-range indices

        if not selected_song_ids:
            return "No valid song numbers found. Please select valid song numbers from the list."

        # Step 3: Add selected songs to the playlist and provide feedback
        try:
            cursor = self.db_conn.cursor(dictionary=True)
            added_songs = []
            
            for song_id in selected_song_ids:
                # Assuming Playlist table has a primary key or unique constraint to avoid duplicate entries
                cursor.execute("INSERT IGNORE INTO Playlist (trackID) VALUES (%s)", (song_id,))
                # Fetch song details for feedback
                cursor.execute("SELECT track_name, artist_name FROM Tracks WHERE id = %s", (song_id,))
                song = cursor.fetchone()
                added_songs.append(f"{song['track_name']} by {song['artist_name']}")
                
            self.db_conn.commit()
            cursor.close()
            self.top_songs = None  # Reset top_songs after adding a song
            return f"Successfully added song(s): {', '.join(added_songs)}" if added_songs else "No new songs were added to your playlist."

        except mysql.connector.Error as e:
            print(f"Database error: {e}")
            return "There was an error adding the songs to your playlist. Please try again."
        
    def add_songs_by_genre(self, genre, num_songs=5):
        """Add the most popular songs from a specified genre to the playlist."""
        try:
            cursor = self.db_conn.cursor(dictionary=True)

            # Step 1: Fetch the top songs of the specified genre, ordered by popularity
            query = """
                SELECT id, track_name, artist_name, popularity
                FROM Tracks
                WHERE genre LIKE %s
                ORDER BY popularity DESC
                LIMIT %s
            """
            cursor.execute(query, (f"%{genre}%", num_songs))
            top_songs = cursor.fetchall()

            if not top_songs:
                return f"No songs found for the genre '{genre}'. Please try another genre."

            # Step 2: Add each song to the playlist, ignoring duplicates
            added_songs = []
            for song in top_songs:
                cursor.execute("INSERT IGNORE INTO Playlist (trackID) VALUES (%s)", (song['id'],))
                added_songs.append(f"{song['track_name']} by {song['artist_name']} (Popularity: {song['popularity']})")
            
            self.db_conn.commit()
            cursor.close()

            # Step 3: Provide feedback to the user
            return (
                f"Added the following {len(added_songs)} song(s) from the genre '{genre}' to your playlist:" +
                " -- ".join(added_songs)
            )

        except mysql.connector.Error as e:
            print(f"Database error: {e}")
            return "There was an error adding songs to your playlist. Please try again."


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
    
    def create_playlist(self, user_input):
        """Create a playlist based on genre classification from the SVM model."""
        
        # Step 1: Use the SVM model to classify the genre category directly from the user input
        vectorized_input = self.genre_vectorizer.transform([user_input])  # Vectorize the input
        predicted_genre_category = self.genre_model.predict(vectorized_input)[0]

        # Step 2: Add songs from the identified genre category to the playlist
        response = self.add_songs_by_genre_category(predicted_genre_category, num_songs=10)
        return response

    def add_songs_by_genre_category(self, genre_category, num_songs=10):
        """Adds multiple songs from all genres within a specified genre category, avoiding duplicates."""

        # Define genre mappings for categories to retrieve multiple genres from each category
        genre_mapping = {
            "classical": ["classical", "classical performance", "baroque", "early music", "orchestral performance", 
                        "classical era", "late romantic era", "early romantic era", "german romanticism", 
                        "post-romantic era", "german baroque", "early modern classical", "opera", "choral"],
            "rock": ["rock", "classic rock", "folk rock", "country rock", "album rock", "roots rock", 
                    "art rock", "blues rock", "rock-and-roll"],
            "jazz": ["jazz", "vocal jazz", "swing", "cool jazz", "big band", "brill building pop", 
                    "adult standards"],
            "hip hop": ["hip hop", "rap", "gangster rap", "pop rap", "southern hip hop", "trap"],
            "pop": ["dance pop", "progressive house", "lounge", "easy listening", "mellow gold", 
                    "latin", "funk", "soft rock"],
            "folk": ["folk", "country rock", "tropical"],
            "miscellaneous": ["sleep"]
        }

        genres_to_add = genre_mapping.get(genre_category, [])
        added_songs = []

        try:
            cursor = self.db_conn.cursor(dictionary=True)

            # Step 1: Retrieve IDs of songs already in the playlist
            cursor.execute("SELECT trackID FROM Playlist")
            existing_song_ids = {row["trackID"] for row in cursor.fetchall()}

            # Step 2: Fetch popular songs from each genre in the category, avoiding songs already in the playlist
            for genre in genres_to_add:
                if existing_song_ids:
                    format_strings = ','.join(['%s'] * len(existing_song_ids))
                    query = f"""
                        SELECT id, track_name, artist_name, popularity
                        FROM Tracks
                        WHERE genre LIKE %s AND id NOT IN ({format_strings})
                        ORDER BY popularity DESC
                        LIMIT %s
                    """
                    params = [f"%{genre}%"] + list(existing_song_ids) + [num_songs // len(genres_to_add) or 1]
                else:
                    # If no songs are in the playlist, no need for NOT IN clause
                    query = """
                        SELECT id, track_name, artist_name, popularity
                        FROM Tracks
                        WHERE genre LIKE %s
                        ORDER BY popularity DESC
                        LIMIT %s
                    """
                    params = [f"%{genre}%", num_songs // len(genres_to_add) or 1]
                
                cursor.execute(query, params)
                top_songs = cursor.fetchall()

                for song in top_songs:
                    # Add new songs to the playlist
                    cursor.execute("INSERT IGNORE INTO Playlist (trackID) VALUES (%s)", (song['id'],))
                    added_songs.append(f"{song['track_name']} by {song['artist_name']}")
                    existing_song_ids.add(song['id'])  # Update the set to include the newly added song

            self.db_conn.commit()
            cursor.close()

            # Provide feedback on the added songs
            if added_songs:
                return f"Added new songs to your playlist from the {genre_category} category:\n" + " -- ".join(added_songs)
            else:
                return f"No new songs found for the {genre_category} category."

        except mysql.connector.Error as e:
            print(f"Database error: {e}")
            return "There was an error adding songs to your playlist. Please try again."

    def answer_question(self, question):
        """Determine the intent of the question and respond accordingly using SVM model classification."""
        
        # Step 1: Classify the question intent
        intent = self.question_classifier.predict([question])[0]
        
        # Step 2: Extract relevant entities based on intent and query the database
        if intent == "count_songs_in_album":
            album_name = self.extract_entity(question, "album")
            return self.count_songs_in_album(album_name)
        
        elif intent == "count_albums_by_artist":
            artist_name = self.extract_entity(question, "artist")
            return self.count_albums_by_artist(artist_name)
        
        else:
            return "I'm sorry, I couldn't understand your question. Could you rephrase it?"

    def extract_entity(self, question, entity_type):
        """Extracts the most relevant entity (album or artist) using n-grams and fuzz score matching, with adjusted scoring for artist matches when looking for albums."""
        
        words = question.lower().split()
        best_match = None
        highest_score = 0

        # Case 1: If looking for an artist, only search self.artists_data
        if entity_type == "artist":
            for size in range(1, len(words) + 1):
                for i in range(len(words) - size + 1):
                    ngram = " ".join(words[i:i + size])

                    for artist in self.artists_data:
                        artist_score = fuzz.ratio(ngram, artist.lower())
                        weighted_score = artist_score * (1 + 0.1 * size)  # Boost for longer n-grams

                        if weighted_score > highest_score:
                            highest_score = weighted_score
                            best_match = artist

            return best_match if highest_score > 60 else None

        # Case 2: If looking for an album, search both albums and artists with adjusted scoring for album-artist matches
        elif entity_type == "album":
            # Step 1: Attempt to find an artist in the question
            artist_match = None
            highest_artist_score = 0
            for size in range(1, len(words) + 1):
                for i in range(len(words) - size + 1):
                    ngram = " ".join(words[i:i + size])

                    for artist in self.artists_data:
                        artist_score = fuzz.ratio(ngram, artist.lower())
                        if artist_score > highest_artist_score:
                            highest_artist_score = artist_score
                            artist_match = artist

            # Step 2: Find the best matching album, adjusting score if the artist matches
            for size in range(1, len(words) + 1):
                for i in range(len(words) - size + 1):
                    ngram = " ".join(words[i:i + size])

                    for album, album_artist in self.albums_data:
                        album_score = fuzz.ratio(ngram, album.lower())
                        
                        # Adjust album score if there's an artist match
                        if artist_match and fuzz.ratio(artist_match.lower(), album_artist.lower()) > 60:
                            album_score *= 1.2  # Boost score by 20% for artist match

                        # Further boost for longer n-grams
                        weighted_score = album_score * (1 + 0.1 * size)

                        if weighted_score > highest_score:
                            highest_score = weighted_score
                            best_match = album

            return best_match if highest_score > 60 else None

        else:
            return None

    def count_songs_in_album(self, album_name):
        """Return the count of songs in the specified album, along with their names."""
        try:
            cursor = self.db_conn.cursor()
            # Get both the count and the list of song names in the album
            query = "SELECT track_name FROM Tracks WHERE album_name = %s"
            cursor.execute(query, (album_name,))
            songs = cursor.fetchall()
            cursor.close()

            if songs:
                song_names = [song[0] for song in songs]
                song_count = len(song_names)
                return f"The album '{album_name}' contains {song_count} song(s)."
            else:
                return f"I couldn't find any songs in the album '{album_name}'."
        except Exception as e:
            print(f"Database error: {e}")
            return "There was an error retrieving the album information. Please try again."

    def count_albums_by_artist(self, artist_name):
        """Return the count of albums by the specified artist, along with their names."""
        try:
            cursor = self.db_conn.cursor()
            # Get both the count and the list of album names by the artist
            query = "SELECT DISTINCT album_name FROM Tracks WHERE artist_name LIKE %s"
            cursor.execute(query, (f"%{artist_name}%",))
            albums = cursor.fetchall()
            cursor.close()

            if albums:
                album_names = [album[0] for album in albums]
                album_count = len(album_names)
                return f"The artist '{artist_name}' is featured on {album_count} album(s)."
            else:
                return f"I couldn't find any albums by the artist '{artist_name}'."
        except Exception as e:
            print(f"Database error: {e}")
            return "There was an error retrieving the artist information. Please try again."
    
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
            playlist_info += f"{song['track_name']} by {song['artist_name']} from the album {song['album_name']}. Genre: {song['genre']}."
            if song != songs[-1]:
                playlist_info += " -- "
        response = playlist_info
        return response
    
    def functionality(self):
        """What can you do?"""
        response = "I'm a virtual assistant that helps you find music. I can add music to your playlist, search for music, and even tell you about the artists and albums you like."
        return response
    

"""
R4 : 10 points
R5 : 8 points
- Model = 6 points OK
- Position-based prompts = 3 points TODO
- Query-based prompts = 3 points TODO 2/3 OK
R6 : 8 points
- Recommendation = 2 points OK
- Way to select = 3 points OK (5 TODO if we add natural language processing)
- Entire playlist = 3 points OK
"""