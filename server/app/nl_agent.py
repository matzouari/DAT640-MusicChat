from dialoguekit.core.annotated_utterance import AnnotatedUtterance
from dialoguekit.core.dialogue_act import DialogueAct
from dialoguekit.core.utterance import Utterance
from dialoguekit.participant.agent import Agent
from dialoguekit.participant.participant import DialogueParticipant

from fuzzywuzzy import fuzz

import mysql.connector
from mysql.connector import Error

from random import choice
import re

import joblib
from nltk import ngrams

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

    def interpret_input(self, user_input):
        """Use the SVM model to interpret the intent from user input."""
        processed_input = self.preprocess_input(user_input)
        vectorized_input = self.vectorizer.transform([processed_input])  # Vectorize the input
        
        # Make the prediction
        prediction = self.natural_model.predict(vectorized_input)
        return prediction

    def preprocess_input(self, text):
        """Preprocess user input (e.g., lowercasing, removing punctuation)."""
        return text.lower()

    def receive_utterance(self, utterance: Utterance) -> None:
        """Handle incoming user input based on predicted intent."""
        user_input = utterance.text
        intent = self.interpret_input(user_input)

        # Dispatch to the appropriate method based on intent
        if user_input == "EXIT":
            self.goodbye()
        if intent == "add":
            response = self.add_songs(user_input)
        elif intent == "remove":
            response = self.remove_song(user_input)
        elif intent == "view":
            response = self.view_playlist()
        elif intent == "clear":
            response = self.clear_playlist()
        else:
            response = AnnotatedUtterance(
                "I'm sorry, I didn't understand you. Please try again.",
                participant=DialogueParticipant.AGENT,
            )
        self._dialogue_connector.register_agent_utterance(response)

    def fetch_track_from_db(self, song_name):
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
        """Extract the most likely song name from the sentence."""
        # Step 1: Fetch all song titles from the database
        all_songs = self.fetch_tracks_from_db()  # Assuming this returns a list of track names
        
        # Step 2: Calculate similarity scores for each song title
        best_match = None
        highest_score = 0
        for song in all_songs:
            # Calculate the similarity score between input sentence and each song title
            score = fuzz.ratio(sentence.lower(), song.lower())
            if score > highest_score:
                highest_score = score
                best_match = song
        
        # Step 3: Return the best match if it meets a reasonable threshold
        # A threshold (e.g., 60) ensures it only returns if a match is reasonably close
        if highest_score > 60:  
            return best_match
        else:
            return None  # No sufficiently close match found

    def is_song_in_database(self, song_name):
        """Helper function to check if a song title exists in the database."""
        try:
            cursor = self.db_conn.cursor()
            query = "SELECT COUNT(*) FROM Tracks WHERE LOWER(track_name) = %s"
            cursor.execute(query, (song_name.lower(),))
            result = cursor.fetchone()
            cursor.close()
            return result[0] > 0
        except Error as e:
            print(f"Database error: {e}")
            return False

    #########################
    ### CHATBOT RESPONSES ###
    #########################

    def add_songs(self, user_input): # Use Naturally for song name for testing.
        # Split the input to see if the artist is mentioned
        song_name = self.extract_song_name(user_input)
        tracks = self.fetch_tracks_from_db(song_name)
        specified_artist = None

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
                response = AnnotatedUtterance(
                    f"There are multiple tracks containing '{song_name}'. Possible tracks: {track_list}. Please specify the artist.",
                    participant=DialogueParticipant.AGENT,
                )
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
                                response = AnnotatedUtterance(
                                    f"Adding {track_info['track_name']} by {track_info['artist_name']} to the playlist. You can ask me for more information about the song, just say 'tell me about song {track_info['track_name']}' or 'what is song {track_info['track_name']} by artist {track_info['artist_name']}'.",
                                    participant=DialogueParticipant.AGENT,
                                )
                            elif illusionOfFreeChoice == 2:
                                response = AnnotatedUtterance(
                                    f"Adding {track_info['track_name']} by {track_info['artist_name']} to the playlist. You can ask me for the album this song is in, just say 'in which album is song {track_info['track_name']} by {track_info['artist_name']}'.",
                                    participant=DialogueParticipant.AGENT,
                                )
                            else:
                                response = AnnotatedUtterance(
                                    f"Adding {track_info['track_name']} by {track_info['artist_name']} to the playlist. You can ask me for more information about the artist, just say 'show me all songs by {track_info['artist_name']}'.",
                                    participant=DialogueParticipant.AGENT,
                                )
                        else:
                            response = AnnotatedUtterance(
                                "Track already exists in playlist",
                                participant=DialogueParticipant.AGENT,
                            )
                    else:
                        response = AnnotatedUtterance(
                            f"No track found named '{song_name}' with artist '{specified_artist}'.",
                            participant=DialogueParticipant.AGENT,
                        )
                else:
                    # If no artist was specified and there's only one match
                    track_info = tracks[0]
                    if self.add_track_to_playlist(track_info["id"]):
                        illusionOfFreeChoice = choice([1,2,3])
                        if illusionOfFreeChoice == 1:
                            response = AnnotatedUtterance(
                                f"Adding {track_info['track_name']} by {track_info['artist_name']} to the playlist. You can ask me for more information about the song, just say 'tell me about song {track_info['track_name']}' or 'what is song {track_info['track_name']} by artist {track_info['artist_name']}'.",
                                participant=DialogueParticipant.AGENT,
                            )
                        elif illusionOfFreeChoice == 2:
                            response = AnnotatedUtterance(
                                f"Adding {track_info['track_name']} by {track_info['artist_name']} to the playlist. You can ask me for the album this song is in, just say 'in which album is song {track_info['track_name']} by {track_info['artist_name']}'.",
                                participant=DialogueParticipant.AGENT,
                            )
                        else:
                            response = AnnotatedUtterance(
                                f"Adding {track_info['track_name']} by {track_info['artist_name']} to the playlist. You can ask me for more information about the artist, just say 'show me all songs by {track_info['artist_name']}'.",
                                participant=DialogueParticipant.AGENT,
                            )
                    else:
                        response = AnnotatedUtterance(
                            "Track already exists in playlist",
                            participant=DialogueParticipant.AGENT,
                        )
        else:
            response = AnnotatedUtterance(
                "Track not found in the database.",
                participant=DialogueParticipant.AGENT,
            )
        return response

    def remove_songs(self, user_input):
        """Remove songs from the playlist."""
        song_name = self.extract_song_name(user_input)
        specified_artist = None

        if self.remove_track_from_playlist_by_name(song_name, specified_artist):
            response = AnnotatedUtterance(
                f"Removed {song_name} from the playlist.",
                participant=DialogueParticipant.AGENT,
            )
        else:
            response = AnnotatedUtterance(
                f"Song {song_name} not found in playlist",
                participant=DialogueParticipant.AGENT,
            )
        return response
    
    def clear_playlist(self):
        """Clear the playlist."""
        self.clear_playlist_from_db()
        response = AnnotatedUtterance(
            "Playlist cleared. All songs removed",
            participant=DialogueParticipant.AGENT,
        )
        return response
    
    def view_playlist(self):
        """View the playlist."""
        songs = self.view_playlist_from_db()
        if not songs:
            response = AnnotatedUtterance(
                "Your playlist is empty.",
                participant=DialogueParticipant.AGENT,
            )
            return response
        playlist_info = "Your playlist contains the following songs: "
        for song in songs:
            playlist_info += f"{song['track_name']} by {song['artist_name']} from the album {song['album_name']}"
            if song != songs[-1]:
                playlist_info += ", "
        response = AnnotatedUtterance(
            playlist_info,
            participant=DialogueParticipant.AGENT,
        )
        return response
    
    def number_of_songs_in_album(self, user_input):
        """Get the number of songs in an album."""
        album_name = user_input.split("album")[1].strip()
        song_count = self.count_songs_in_album(album_name)
        response = AnnotatedUtterance(
            f"The album '{album_name}' contains {song_count} song(s).",
            participant=DialogueParticipant.AGENT,
        )
        return response
    
    def number_of_songs_by_artist(self, user_input):
        """Get the number of songs in an album."""
        artist_name = user_input.split("by")[1].strip()
        song_count = self.count_songs_by_artist(artist_name)
        response = AnnotatedUtterance(
            f"The artist '{artist_name}' has written {song_count} song(s).",
            participant=DialogueParticipant.AGENT,
        )
        return response
    
    def who_wrote_song(self, user_input):
        """Get the artist of a song."""
        track_name = user_input.split("song")[1].strip()
        artist = self.get_artist_of_song(track_name)
        if artist:
            response = AnnotatedUtterance(
                f"The song '{track_name}' was written by {artist}.",
                participant=DialogueParticipant.AGENT,
            )
        else:
            response = AnnotatedUtterance(
                f"I couldn't find the artist for the song '{track_name}'.",
                participant=DialogueParticipant.AGENT,
            )
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
            response = AnnotatedUtterance(
                "I'm sorry, I didn't understand you. Please try again.",
                participant=DialogueParticipant.AGENT,
            )
            return response

        tracks = self.fetch_tracks_from_db(track_name)

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
                response = AnnotatedUtterance(
                    f"There are multiple tracks containing '{track_name}'. Possible tracks: {track_list}. Please specify the artist.",
                    participant=DialogueParticipant.AGENT,
                )
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
                        response = AnnotatedUtterance(
                            f"The song '{track_info['track_name']}' was written by {track_info['artist_name']} and is from the album '{track_info['album_name']}'.",
                            participant=DialogueParticipant.AGENT,
                        )
                    else:
                        response = AnnotatedUtterance(
                            f"No track found named '{track_name}' with artist '{specified_artist}'.",
                            participant=DialogueParticipant.AGENT,
                        )
                else:
                    # If no artist was specified and there's only one match
                    track_info = tracks[0]
                    response = AnnotatedUtterance(
                        f"The song '{track_info['track_name']}' was written by {track_info['artist_name']} and is from the album '{track_info['album_name']}'.",
                        participant=DialogueParticipant.AGENT,
                    )
        else:
            response = AnnotatedUtterance(
                "Track not found in the database.",
                participant=DialogueParticipant.AGENT,
            )
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
            response = AnnotatedUtterance(
                "I'm sorry, I didn't understand you. Please try again.",
                participant=DialogueParticipant.AGENT,
            )
            return response

        tracks = self.fetch_tracks_from_db(track_name)

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
                response = AnnotatedUtterance(
                    f"There are multiple tracks containing '{track_name}'. Possible tracks: {track_list}. Please specify the artist.",
                    participant=DialogueParticipant.AGENT,
                )
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
                        response = AnnotatedUtterance(
                            f"The song '{track_info['track_name']}' was written by {track_info['artist_name']} and is from the album '{track_info['album_name']}'.",
                            participant=DialogueParticipant.AGENT,
                        )
                    else:
                        response = AnnotatedUtterance(
                            f"No track found named '{track_name}' with artist '{specified_artist}'.",
                            participant=DialogueParticipant.AGENT,
                        )
                else:
                    # If no artist was specified and there's only one match
                    track_info = tracks[0]
                    response = AnnotatedUtterance(
                        f"The song '{track_info['track_name']}' was written by {track_info['artist_name']} and is from the album '{track_info['album_name']}'.",
                        participant=DialogueParticipant.AGENT,
                    )
        else:
            response = AnnotatedUtterance(
                "Track not found in the database.",
                participant=DialogueParticipant.AGENT,
            )
        return response
    
    def show_songs_from_album(self, user_input):
        """Show all songs from an album."""
        album_name = user_input.split("album")[1].strip()
        songs = self.get_songs_from_album(album_name)
        
        if songs:
            song_list = ', '.join([f"{song['track_name']} by {song['artist_name']}" for song in songs])
            response = AnnotatedUtterance(
                f"Here are the songs from the album '{album_name}': {song_list}.",
                participant=DialogueParticipant.AGENT,
            )
        else:
            response = AnnotatedUtterance(
                f"I couldn't find any songs from the album '{album_name}'.",
                participant=DialogueParticipant.AGENT,
            )
        return response
    
    def show_songs_by_artist(self, user_input):
        """Show all songs by an artist."""
        artist_name = user_input.split("by")[1].strip()
        songs = self.get_songs_by_artist(artist_name)
        
        if songs:
            song_list = ', '.join([f"{song['track_name']} by {song['artist_name']}" for song in songs])
            response = AnnotatedUtterance(
                f"Here are the songs by the artist '{artist_name}': {song_list}.",
                participant=DialogueParticipant.AGENT,
            )
        else:
            response = AnnotatedUtterance(
                f"I couldn't find any songs by the artist '{artist_name}'.",
                participant=DialogueParticipant.AGENT,
            )
        return response
    
    def how_do_you_work(self):
        """How do you work?"""
        response = AnnotatedUtterance(
            "I'm a virtual assistant that helps you find music. I can add music to your playlist, search for music, and even tell you about the artists and albums you like.",
            participant=DialogueParticipant.AGENT,
        )
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
            query = "SELECT id, track_name, artist_name, album_name FROM Tracks WHERE id IN (SELECT trackID FROM Playlist)"
            cursor.execute(query)
            result = cursor.fetchall()
            return result if result else []
        except Error as e:
            print(f"Error viewing playlist: {e}")
            return []

    ### DATABASE PROMPTS ###

    def fetch_tracks_from_db(self, track_name):
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

