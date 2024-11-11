from dialoguekit.core.annotated_utterance import AnnotatedUtterance
from dialoguekit.core.dialogue_act import DialogueAct
from dialoguekit.core.utterance import Utterance
from dialoguekit.participant.agent import Agent
from dialoguekit.participant.participant import DialogueParticipant

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
        if intent == "add":
            response = self.add_songs(user_input)
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
        
    def extract_song_name(self, sentence):
        """Extract the most likely song name from the sentence, using consecutive word matches."""
        # Fetch all song titles from the database
        all_songs = self.fetch_all_tracks_from_db()
        
        # Preprocess input by removing the first word (e.g., "add", "play", etc.) and lowercasing
        split_sentence = sentence.split(' ')
        new_sentence = ' '.join(split_sentence[1:]).lower()

        # Generate 2-3 word n-grams from the user's input for consecutive matching
        ngram_matches = []
        for n in range(1, 4):  # Generate bigrams and trigrams
            ngram_matches.extend([' '.join(gram) for gram in ngrams(new_sentence.split(), n)])
        
        # Initialize variables for tracking the best match
        best_song_match = None
        highest_song_score = 0
        best_artist_match = None
        highest_artist_score = 0

        # Compare each n-gram to song titles in the database
        for song in all_songs:
            song_title = song["track_name"].lower()
            title_word_count = len(song_title.split())

            artist_name = song["artist_name"].lower()
            for ngram in ngram_matches:
                # Score each n-gram against the song title
                score = fuzz.token_sort_ratio(ngram, song_title)

                # Apply weighting based on word count in the title
                if title_word_count == 1:
                    weighted_score = score * 0.8  # Penalize single-word titles
                elif 2 <= title_word_count <= 3:
                    weighted_score = score * 1.2  # Boost for 2-3 word titles
                else:
                    weighted_score = score / (1 + (title_word_count - 3) * 0.1)  # Penalize longer titles

                # Update best match if this weighted score is the highest
                if weighted_score > highest_song_score:
                    highest_song_score = weighted_score
                    best_song_match = song_title

                # Score each n-gram against the artist name
                artist_score = fuzz.token_sort_ratio(ngram, artist_name)

                # Update best artist match if this score is the highest
                if artist_score > highest_artist_score:
                    highest_artist_score = artist_score
                    best_artist_match = artist_name
    

        # Return the best match if it meets a reasonable threshold
        if highest_song_score > 60:
            if highest_artist_score > 80:
                return best_song_match, best_artist_match
            return best_song_match, None
        else:
            return None, None  # No sufficiently close match found
    
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
                    SELECT track_name, artist_name, popularity 
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

            # Format recommendations as strings for response
            recommendation_list = ", ".join(
                f"{rec['track_name']} by {rec['artist_name']} (Popularity: {rec['popularity']})" for rec in recommendations
            )
            response = f"Your playlist is based on the following genres: {', '.join(top_genres)}. Based on your playlist, here are some song recommendations:\n{recommendation_list}"

        except mysql.connector.Error as e:
            response = "There was an error fetching recommendations. Please try again."
            print(f"Database error: {e}")
        finally:
            cursor.close()

        return response
    
    def functionality(self):
        """What can you do?"""
        response = "I'm a virtual assistant that helps you find music. I can add music to your playlist, search for music, and even tell you about the artists and albums you like."
        return response
        
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

    def add_songs(self, user_input):
        # Split the input to see if the artist is mentioned
        song_name, specified_artist = self.extract_song_name(user_input)
        tracks = self.fetch_tracks_from_db_by_track_name(song_name)

        if tracks:
            # If multiple tracks are found and no artist was specified in the input
            if len(tracks) > 1 and not specified_artist:
                # Collect the unique artist names for this track
                possible_tracks = [
                    f"{track['track_name']} by {track['artist_name']}"
                    for track in sorted(tracks, key=lambda x: x['popularity'], reverse=True)
                ]

                # Construct a message listing the artists
                track_list = ", ".join(possible_tracks)
                response = f"There are multiple tracks containing '{song_name}'. Possible tracks: {track_list}. Please specify the artist."
            else:
                # If artist is provided or only one track matches, filter or use the correct track
                if specified_artist:
                    # Filter by specified artist; check if the specified artist matches any in the semicolon-separated list
                    matching_tracks = []
                    for track in tracks:
                        track_artists = [artist.strip().lower() for artist in track['artist_name'].split(';')]
                        if specified_artist in track_artists:
                            matching_tracks.append(track)

                    if matching_tracks:  # Proceed if there's a matching track
                        track_info = matching_tracks[0]  # First match (or modify to let the user choose)
                        if self.add_track_to_playlist(track_info["id"]):
                            illusionOfFreeChoice = choice([1,2,3])
                            if illusionOfFreeChoice == 1:
                                response = f"Adding {track_info['track_name']} by {track_info['artist_name']} to the playlist. You can ask me for more information about the song, just say 'tell me about song {track_info['track_name']}' or 'what is song {track_info['track_name']} by artist {track_info['artist_name']}'."
                            elif illusionOfFreeChoice == 2:
                                response = f"Adding {track_info['track_name']} by {track_info['artist_name']} to the playlist. You can ask me for the album this song is in, just say 'in which album is song {track_info['track_name']} by {track_info['artist_name']}'."
                            else:
                                response = f"Adding {track_info['track_name']} by {track_info['artist_name']} to the playlist. You can ask me for more information about the artist, just say 'show me all songs by {track_info['artist_name']}'."
                        else:
                            response = "Track already exists in playlist",
                    else:
                        response = f"No track found named '{song_name}' with artist '{specified_artist}'."
                else:
                    # If no artist was specified and there's only one match
                    track_info = tracks[0]
                    if self.add_track_to_playlist(track_info["id"]):
                        illusionOfFreeChoice = choice([1,2,3])
                        if illusionOfFreeChoice == 1:
                            response = f"Adding {track_info['track_name']} by {track_info['artist_name']} to the playlist. You can ask me for more information about the song, just say 'tell me about song {track_info['track_name']}' or 'what is song {track_info['track_name']} by artist {track_info['artist_name']}'."
                        elif illusionOfFreeChoice == 2:
                            response = f"Adding {track_info['track_name']} by {track_info['artist_name']} to the playlist. You can ask me for the album this song is in, just say 'in which album is song {track_info['track_name']} by {track_info['artist_name']}'."
                        else:
                            response = f"Adding {track_info['track_name']} by {track_info['artist_name']} to the playlist. You can ask me for more information about the artist, just say 'show me all songs by {track_info['artist_name']}'."
                    else:
                        response = "Track already exists in playlist"
        else:
            response = "Track not found in the database."
        return response

    def remove_songs(self, user_input):
        """Remove songs from the playlist."""
        song_name, specified_artist = self.extract_song_name(user_input)

        if self.remove_track_from_playlist_by_name(song_name, specified_artist):
            response = f"Removed {song_name} from the playlist."
        else:
            response = f"Song {song_name} not found in playlist"
        return response
    
    def clear_playlist(self):
        """Clear the playlist."""
        self.clear_playlist_from_db()
        response = "Playlist cleared. All songs removed"
        return response
    
    def view_playlist(self):
        """View the playlist."""
        songs = self.view_playlist_from_db()
        if not songs:
            response = "Your playlist is empty."
            return response
        playlist_info = "Your playlist contains the following songs: "
        for song in songs:
            playlist_info += f"{song['track_name']} by {song['artist_name']} from the album {song['album_name']}. Genre: {song['genre']}. Popularity: {song['popularity']}"
            if song != songs[-1]:
                playlist_info += ", "
        response = playlist_info
        return response
    
    def number_of_songs_in_album(self, user_input):
        """Get the number of songs in an album."""
        album_name = user_input.split("album")[1].strip()
        song_count = self.count_songs_in_album(album_name)
        response = f"The album '{album_name}' contains {song_count} song(s)."
        return response
    
    def number_of_songs_by_artist(self, user_input):
        """Get the number of songs in an album."""
        artist_name = user_input.split("by")[1].strip()
        song_count = self.count_songs_by_artist(artist_name)
        response = f"The artist '{artist_name}' has written {song_count} song(s)."
        return response
    
    def who_wrote_song(self, user_input):
        """Get the artist of a song."""
        track_name = user_input.split("song")[1].strip()
        artist = self.get_artist_of_song(track_name)
        if artist:
            response = f"The song '{track_name}' was written by {artist}."
        else:
            response = f"I couldn't find the artist for the song '{track_name}'."
        return response
    
    def in_which_album(self, user_input):
        """In which album is song X."""
        if "song" in user_input:
            if "by" in user_input:
                parts = user_input.split("by")
                track_name = parts[0].split("song")[1].strip()
                specified_artist = parts[1].strip().lower()
            else:
                track_name = user_input.split("song")[1].strip()
                specified_artist = None  # No artist specified initially
        else:
            response = "I'm sorry, I didn't understand you. Please try again."
            return response

        tracks = self.fetch_tracks_from_db_by_track_name(track_name)

        if tracks:
            # If multiple tracks are found and no artist was specified in the input
            if len(tracks) > 1 and not specified_artist:
                # Collect the unique artist names for this track
                possible_tracks = [
                    f"{track['track_name']} by {track['artist_name']}"
                    for track in sorted(tracks, key=lambda x: x['popularity'], reverse=True)
                ]

                # Construct a message listing the artists
                track_list = ", ".join(possible_tracks)
                response = f"There are multiple tracks containing '{track_name}'. Possible tracks: {track_list}. Please specify the artist."
            else:
                # If artist is provided or only one track matches, filter or use the correct track
                if specified_artist:
                    # Filter by specified artist; check if the specified artist matches any in the semicolon-separated list
                    matching_tracks = []
                    for track in tracks:
                        track_artists = [artist.strip().lower() for artist in track['artist_name'].split(';')]
                        if specified_artist in track_artists:
                            matching_tracks.append(track)

                    if matching_tracks:  # Proceed if there's a matching track
                        track_info = matching_tracks[0]  # First match (or modify to let the user choose)
                        response = f"The song '{track_info['track_name']}' was written by {track_info['artist_name']} and is from the album '{track_info['album_name']}'."
                    else:
                        response = f"No track found named '{track_name}' with artist '{specified_artist}'."
                else:
                    # If no artist was specified and there's only one match
                    track_info = tracks[0]
                    response = f"The song '{track_info['track_name']}' was written by {track_info['artist_name']} and is from the album '{track_info['album_name']}'."
        else:
            response = "Track not found in the database."
        return response
    
    def tell_me_about_song(self, user_input):
        """Tell me about a song."""
        if "song" in user_input:
            if "by" in user_input:
                parts = user_input.split("by")
                track_name = parts[0].split("song")[1].strip()
                specified_artist = parts[1].strip().lower()
            else:
                track_name = user_input.split("song")[1].strip()
                specified_artist = None  # No artist specified initially
        else:
            response = "I'm sorry, I didn't understand you. Please try again."
            return response

        tracks = self.fetch_tracks_from_db_by_track_name(track_name)

        if tracks:
            # If multiple tracks are found and no artist was specified in the input
            if len(tracks) > 1 and not specified_artist:
                # Collect the unique artist names for this track
                possible_tracks = [
                    f"{track['track_name']} by {track['artist_name']}"
                    for track in sorted(tracks, key=lambda x: x['popularity'], reverse=True)
                ]

                # Construct a message listing the artists
                track_list = ", ".join(possible_tracks)
                response = f"There are multiple tracks containing '{track_name}'. Possible tracks: {track_list}. Please specify the artist."
            else:
                # If artist is provided or only one track matches, filter or use the correct track
                if specified_artist:
                    # Filter by specified artist; check if the specified artist matches any in the semicolon-separated list
                    matching_tracks = []
                    for track in tracks:
                        track_artists = [artist.strip().lower() for artist in track['artist_name'].split(';')]
                        if specified_artist in track_artists:
                            matching_tracks.append(track)

                    if matching_tracks:  # Proceed if there's a matching track
                        track_info = matching_tracks[0]  # First match (or modify to let the user choose)
                        response = f"The song '{track_info['track_name']}' was written by {track_info['artist_name']} and is from the album '{track_info['album_name']}'."
                    else:
                        response = f"No track found named '{track_name}' with artist '{specified_artist}'."
                else:
                    # If no artist was specified and there's only one match
                    track_info = tracks[0]
                    response = f"The song '{track_info['track_name']}' was written by {track_info['artist_name']} and is from the album '{track_info['album_name']}'."
        else:
            response = "Track not found in the database."
        return response
    
    def show_songs_from_album(self, user_input):
        """Show all songs from an album."""
        album_name = user_input.split("album")[1].strip()
        songs = self.get_songs_from_album(album_name)
        
        if songs:
            song_list = ', '.join([f"{song['track_name']} by {song['artist_name']}" for song in songs])
            response = f"Here are the songs from the album '{album_name}': {song_list}."
        else:
            response = f"I couldn't find any songs from the album '{album_name}'."
        return response
    
    def show_songs_by_artist(self, user_input):
        """Show all songs by an artist."""
        artist_name = user_input.split("by")[1].strip()
        songs = self.get_songs_by_artist(artist_name)
        
        if songs:
            song_list = ', '.join([f"{song['track_name']} by {song['artist_name']}" for song in songs])
            response = f"Here are the songs by the artist '{artist_name}': {song_list}."
        else:
            response = f"I couldn't find any songs by the artist '{artist_name}'."
        return response


    ########################
    ### DATABASE PROMPTS ###
    ########################

    ### PLAYLIST PROMPTS ###

    def add_track_to_playlist(self, track_id):
        """Add a track to the playlist."""
        try:
            cursor = self.db_conn.cursor()
            query = "INSERT INTO Playlist (trackID) VALUES (%s)"
            cursor.execute(query, (track_id,))
            print(track_id)
            self.db_conn.commit()
            return True
        except Error as e:
            print(f"Error adding track to playlist: {e}")
            return False

    def remove_track_from_playlist(self, track_id):
        """Remove a track from the playlist."""
        try:
            cursor = self.db_conn.cursor()
            query = "DELETE FROM Playlist WHERE trackID = %s"
            cursor.execute(query, (track_id,))
            self.db_conn.commit()
            return True
        except Error as e:
            print(f"Error removing track from playlist: {e}")
            return False
        
    def remove_track_from_playlist_by_name(self, track_name, artist_name = None):
        """Remove a track from the playlist by name."""
        try:
            cursor = self.db_conn.cursor()
            if artist_name is not None:
                query = "SELECT t.id FROM Tracks t JOIN Playlist p ON t.id = p.trackID WHERE t.track_name = %s AND t.artist_name = %s"
                cursor.execute(query, (track_name, artist_name))
            else:
                query = "SELECT t.id FROM Tracks t JOIN Playlist p ON t.id = p.trackID WHERE t.track_name = %s"
                cursor.execute(query, (track_name,))
            result = cursor.fetchone()
            if result:
                track_id = result[0]
                query = "DELETE FROM Playlist WHERE trackID = %s"
                cursor.execute(query, (track_id,))
                self.db_conn.commit()
                return True
            else:
                return False
        except Error as e:
            print(f"Error removing track from playlist: {e}")
            return False
        
    def clear_playlist_from_db(self):
        """Clear the playlist."""
        try:
            cursor = self.db_conn.cursor()
            query = "DELETE FROM Playlist"
            cursor.execute(query)
            self.db_conn.commit()
            return True
        except Error as e:
            print(f"Error clearing playlist: {e}")
            return False
    
    def view_playlist_from_db(self):
        """View the playlist."""
        try:
            cursor = self.db_conn.cursor(dictionary=True)
            query = "SELECT id, track_name, artist_name, album_name, genre, popularity FROM Tracks WHERE id IN (SELECT trackID FROM Playlist)"
            cursor.execute(query)
            result = cursor.fetchall()
            return result if result else []
        except Error as e:
            print(f"Error viewing playlist: {e}")
            return []

    ### DATABASE PROMPTS ###

    def fetch_all_tracks_from_db(self):
        """Fetch all tracks from the database."""
        try:
            cursor = self.db_conn.cursor(dictionary=True)
            query = "SELECT id, track_name, artist_name, album_name FROM Tracks"
            cursor.execute(query)
            result = cursor.fetchall()
            return result if result else []
        except Error as e:
            print(f"Error fetching all tracks: {e}")
            return []

    def fetch_tracks_from_db_by_track_name(self, track_name):
        """Fetch up to 10 tracks from the database ordered by popularity, 
        accounting for possible mispunctuation in the track name."""
        try:
            cursor = self.db_conn.cursor(dictionary=True)
            
            # Create a search pattern by removing common punctuation and matching variants
            sanitized_track_name = re.sub(r"[^\w\s]", "", track_name)
            search_pattern = f"%{sanitized_track_name}%"
            
            query = """
            SELECT id, track_name, artist_name, album_name, popularity
            FROM Tracks 
            WHERE REPLACE(track_name, "'", '') LIKE %s
            ORDER BY popularity DESC
            LIMIT 10
            """
            
            cursor.execute(query, (search_pattern,))
            results = cursor.fetchall()
            return results
        except Error as e:
            print(f"Error fetching tracks: {e}")
            return []

    def count_songs_in_album(self, album_name):
        """Count how many songs are in the specified album."""
        try:
            cursor = self.db_conn.cursor()
            query = "SELECT COUNT(*) FROM Tracks WHERE album_name = %s"
            cursor.execute(query, (album_name,))
            result = cursor.fetchone()
            return result[0] if result else 0
        except Error as e:
            print(f"Error counting songs: {e}")
            return 0
        
    def count_songs_by_artist(self, artist_name):
        """Count how many songs are in the specified album."""
        try:
            cursor = self.db_conn.cursor()
            query = "SELECT COUNT(*) FROM Tracks WHERE artist_name LIKE %s"
            cursor.execute(query, (artist_name,))
            result = cursor.fetchone()
            return result[0] if result else 0
        except Error as e:
            print(f"Error counting songs: {e}")
            return 0
    
    def get_artist_of_song(self, track_name):
        """Find the artist of the specified song, ignoring featured artists."""
        try:
            cursor = self.db_conn.cursor()
            # Modify query to use LIKE and allow matching without the featured artist part
            query = "SELECT artist_name FROM Tracks WHERE track_name LIKE %s"
            # Add a wildcard at the end of the track name to ignore any "(feat. ...)"
            search_pattern = track_name.strip() + '%'
            cursor.execute(query, (search_pattern,))
            result = cursor.fetchone()
            return result[0] if result else None
        except Error as e:
            print(f"Error fetching artist: {e}")
            return None

    def get_songs_from_album(self, album_name):
        """Fetch all songs from the specified album."""
        try:
            cursor = self.db_conn.cursor(dictionary=True)
            query = "SELECT track_name, artist_name FROM Tracks WHERE album_name = %s"
            cursor.execute(query, (album_name,))
            result = cursor.fetchall()
            return result if result else []
        except Error as e:
            print(f"Error fetching songs: {e}")
            return []
        
    def get_songs_by_artist(self, artist_name):
        """Fetch all songs by the specified artist."""
        try:
            cursor = self.db_conn.cursor(dictionary=True)
            query = "SELECT track_name, artist_name FROM Tracks WHERE artist_name LIKE %s"
            cursor.execute(query, (artist_name,))
            result = cursor.fetchall()
            return result if result else []
        except Error as e:
            print(f"Error fetching songs: {e}")
            return []
        
## TODO ## 
# Change the playlist to be fully handled by the database
# Frontend show the playlist

