from dialoguekit.core.annotated_utterance import AnnotatedUtterance
from dialoguekit.core.dialogue_act import DialogueAct
from dialoguekit.core.utterance import Utterance
from dialoguekit.participant.agent import Agent
from dialoguekit.participant.participant import DialogueParticipant

import mysql.connector
from mysql.connector import Error
from playlist import Playlist, Track

class MusicAgent(Agent):
    def __init__(self, id: str):
        """Initialize MusicBot agent."""
        super().__init__(id)
        self.pl = Playlist(name="My playlist", tracks=[])

        # Database connection configuration
        db_config = {
            'user': 'root',  # or your DB user
            'password': 'musicpwd',
            'host': 'localhost',
            'database': 'MusicDB',
        }
        # Establishing the connection
        def connect_to_db():
            try:
                conn = mysql.connector.connect(**db_config)
                if conn.is_connected():
                    print('Connected to MySQL database')
                return conn
            except mysql.connector.Error as err:
                print(f"Error: {err}")
                return None
        self.db_conn = connect_to_db()

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
        """Gets called each time there is a new user utterance.

        If the received message is "EXIT" it will close the conversation.

        Args:
            utterance: User utterance.
        """
        # Convert the input text to lowercase
        user_input = utterance.text.lower()

        if user_input == "EXIT":
            self.goodbye()
            return
        elif "add" in user_input:
            response = self.add_songs(user_input)
        elif "remove" in user_input or "delete" in user_input:
            response = self.remove_songs(user_input)
        elif "clear" in user_input:
            response = self.clear_playlist()
        elif "view" in user_input or "see" in user_input:
            response = self.view_playlist()
        elif "how many songs" in user_input and "album" in user_input: # How many songs are in album X
            response = self.number_of_songs_in_album(user_input)
        elif "who wrote" in user_input or "which artist" in user_input: # Who wrote song X, Which artist wrote song X
            response = self.who_wrote_song(user_input)
        elif "show me all songs" in user_input and "album" in user_input: # Show me all songs from album X
            response = self.show_songs_from_album(user_input)
        else: 
            response = AnnotatedUtterance(
                "I'm sorry, I didn't understand you. Please try again.",
                participant=DialogueParticipant.AGENT,
            )
        self._dialogue_connector.register_agent_utterance(response)

    #########################
    ### CHATBOT RESPONSES ###
    #########################

    def add_songs(self, user_input):
        # Split the input to see if the artist is mentioned
        if "by" in user_input:
            parts = user_input.split("by")
            track_name = parts[0][4:].strip()  # The part after "add"
            specified_artist = parts[1].strip().lower()
        else:
            track_name = user_input[4:].strip()
            specified_artist = None  # No artist specified initially

        tracks = self.fetch_tracks_from_db(track_name)

        if tracks:
            # If multiple tracks are found and no artist was specified in the input
            if len(tracks) > 1 and not specified_artist:
                # Collect the unique artist names for this track
                possible_artists = {track['artist_name'] for track in tracks}

                # Construct a message listing the artists
                artist_list = ", ".join(possible_artists)
                response = AnnotatedUtterance(
                    f"There are multiple tracks named '{track_name}'. Possible artists: {artist_list}. Please specify the artist.",
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
                        track = Track(
                            name=track_info['track_name'],
                            artist=track_info['artist_name'],
                            album=track_info['album_name']
                        )
                        if self.pl.add_track(track):
                            response = AnnotatedUtterance(
                                f"Adding {track.name} by {track.artist} to the playlist.",
                                participant=DialogueParticipant.AGENT,
                            )
                        else:
                            response = AnnotatedUtterance(
                                "Track already exists in playlist",
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
                    track = Track(
                        name=track_info['track_name'],
                        artist=track_info['artist_name'],
                        album=track_info['album_name']
                    )
                    if self.pl.add_track(track):
                        response = AnnotatedUtterance(
                            f"Adding {track.name} by {track.artist} to the playlist.",
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
        if self.pl.remove_track(Track(user_input[7:])):
            response = AnnotatedUtterance(
                "Removing song from playlist",
                participant=DialogueParticipant.AGENT,
            )
        else:
            response = AnnotatedUtterance(
                "Song not found in playlist",
                participant=DialogueParticipant.AGENT,
            )
        return response
    
    def clear_playlist(self):
        """Clear the playlist."""
        self.pl.clear_playlist()
        response = AnnotatedUtterance(
            "Playlist cleared. All songs removed",
            participant=DialogueParticipant.AGENT,
        )
        return response
    
    def view_playlist(self):
        """View the playlist."""
        response = AnnotatedUtterance(
            str(self.pl),
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
    
    
    ########################
    ### DATABASE PROMPTS ###
    ########################

    def fetch_tracks_from_db(self, track_name):
        """Fetch tracks from the database by track name, handling multiple results."""
        try:
            cursor = self.db_conn.cursor(dictionary=True)
            query = """
            SELECT id, track_name, artist_name, album_name 
            FROM Tracks 
            WHERE track_name LIKE %s
            """
            search_pattern = track_name.strip() + '%'
            cursor.execute(query, (search_pattern,))
            result = cursor.fetchall()
            return result
        except Error as e:
            print(f"Error fetching tracks: {e}")
            return None

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
