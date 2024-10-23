from dialoguekit.core.annotated_utterance import AnnotatedUtterance
from dialoguekit.core.dialogue_act import DialogueAct
from dialoguekit.core.utterance import Utterance
from dialoguekit.participant.agent import Agent
from dialoguekit.participant.participant import DialogueParticipant

from playlist import *

class MusicAgent(Agent):
    def __init__(self, id: str):
        """Parrot agent.

        This agent parrots back what the user utters.
        To end the conversation the user has to say `EXIT`.

        Args:
            id: Agent id.
        """
        super().__init__(id)
        self.pl = Playlist(name="My playlist", tracks=[])
        # self.playlist_list = []

    def welcome(self) -> None:
        """Sends the agent's welcome message."""
        utterance = AnnotatedUtterance(
            "Hello, I'm MusicBot. What can I help u with?",
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
        elif "create" in utterance.text: # If we want to create multiple playlists
            # self.playlist_list.append(Playlist(name="Playlist " + len(self.playlist_list + 1), tracks=[]))
            response = AnnotatedUtterance(
                "Playlist already exists",
                participant=DialogueParticipant.AGENT,
            ) 
        elif "add" in utterance.text:
            if self.pl.add_track(Track(utterance.text[3:])):
                response = AnnotatedUtterance(
                    "Adding song to playlist",
                    participant=DialogueParticipant.AGENT,
                )
            else:
                response = AnnotatedUtterance(
                    "Song already exists in playlist",
                    participant=DialogueParticipant.AGENT,
                )
        elif "remove" in utterance.text or "delete" in utterance.text:
            if self.pl.remove_track(Track(utterance.text)):
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
        else: 
            response = AnnotatedUtterance(
                "I'm sorry, I didn't understand you. Please try again.",
                participant=DialogueParticipant.AGENT,
            )
        self._dialogue_connector.register_agent_utterance(response)