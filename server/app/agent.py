from dialoguekit.core.annotated_utterance import AnnotatedUtterance
from dialoguekit.core.dialogue_act import DialogueAct
from dialoguekit.core.utterance import Utterance
from dialoguekit.participant.agent import Agent
from dialoguekit.participant.participant import DialogueParticipant

import mysql.connector
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

    def fetch_track_from_db(self, track_name):
        """Fetch track from database."""
        try:
            cursor = self.db_conn.cursor(dictionary=True)
            query = "SELECT id, track_name, artist_name, album_name FROM Tracks WHERE track_name = %s"
            cursor.execute(query, (track_name,))
            result = cursor.fetchone()
            return result
        except Error as e:
            print(f"Error fetching track: {e}")
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
        """Gets called each time there is a new user utterance.

        If the received message is "EXIT" it will close the conversation.

        Args:
            utterance: User utterance.
        """
        if utterance.text == "EXIT":
            self.goodbye()
            return
        elif "add" in utterance.text:
            track_name = utterance.text[4:]
            track_info = self.fetch_track_from_db(track_name)

            if track_info:
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
        elif "remove" in utterance.text or "delete" in utterance.text:
            if self.pl.remove_track(Track(utterance.text[7:])):
                response = AnnotatedUtterance(
                    "Removing song from playlist",
                    participant=DialogueParticipant.AGENT,
                )
            else:
                response = AnnotatedUtterance(
                    "Song not found in playlist",
                    participant=DialogueParticipant.AGENT,
                )
        elif "view" in utterance.text or "see" in utterance.text:
            response = AnnotatedUtterance(
                str(self.pl),
                participant=DialogueParticipant.AGENT,
            )
        elif "clear" in utterance.text:
            self.pl.clear_playlist()
            response = AnnotatedUtterance(
                "Playlist cleared. All songs removed",
                participant=DialogueParticipant.AGENT,
            )
        elif "hi" in utterance.text or "hello" in utterance.text:
            response = AnnotatedUtterance(
                "Hello, I'm MusicBot. What can I help you with?",
                participant=DialogueParticipant.AGENT,
            )
        else: 
            response = AnnotatedUtterance(
                "I'm sorry, I didn't understand you. Please try again.",
                participant=DialogueParticipant.AGENT,
            )
        self._dialogue_connector.register_agent_utterance(response)