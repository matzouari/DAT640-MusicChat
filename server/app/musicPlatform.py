"""The Platform facilitates displaying of the conversation."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Type, cast

from flask import Flask, Request, request
from flask_socketio import Namespace, SocketIO

from dialoguekit.core.annotated_utterance import AnnotatedUtterance
from dialoguekit.platforms.platform import Platform

if TYPE_CHECKING:
    from dialoguekit.core import Utterance
    from dialoguekit.participant.agent import Agent


logger = logging.getLogger(__name__)


class SocketIORequest(Request):
    """A request that contains a sid attribute."""

    sid: str


@dataclass
class Message:
    text: str
    dialogue_acts: List[Dict[str, Any]] = None

    @classmethod
    def from_utterance(self, utterance: Utterance) -> Message:
        """Converts an utterance to a message.

        Args:
            utterance: An instance of Utterance.

        Returns:
            An instance of Message.
        """
        message = Message(utterance.text)
        if isinstance(utterance, AnnotatedUtterance):
            message.dialogue_acts = [
                {
                    "intent": str(da.intent),
                    "annotations": [
                        {"slot": annotation.slot, "value": annotation.value}
                        for annotation in da.annotations
                    ],
                }
                for da in utterance.dialogue_acts
            ]
        return message


@dataclass
class Response:
    recipient: str
    message: Message


class FlaskSocketPlatform(Platform):
    def __init__(self, agent_class: Type[Agent]) -> None:
        """Represents a platform that uses Flask-SocketIO.

        Args:
            agent_class: The class of the agent.
        """
        super().__init__(agent_class)
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")

    def start(self, host: str = "127.0.0.1", port: str = "5000") -> None:
        """Starts the platform.

        Args:
            host: Hostname.
            port: Port.
        """
        self.socketio.on_namespace(ChatNamespace("/", self))
        self.socketio.run(self.app, host=host, port=port)

    def display_agent_utterance(
        self, user_id: str, utterance: Utterance
    ) -> None:
        """Emits agent utterance to the client.

        Args:
            user_id: User ID.
            utterance: An instance of Utterance.
        """
        message = Message.from_utterance(utterance)
        print(message)
        self.socketio.send(
            asdict(Response(user_id, message)),
            room=user_id,
        )

    def display_user_utterance(
        self, user_id: str, utterance: Utterance
    ) -> None:
        """Overrides the method in Platform to avoid raising an error.

        This method is not used in FlaskSocketPlatform.

        Args:
            user_id: User ID.
            utterance: An instance of Utterance.
        """
        pass


class ChatNamespace(Namespace):
    def __init__(self, namespace: str, platform: FlaskSocketPlatform) -> None:
        """Represents a namespace.

        Args:
            namespace: Namespace.
            platform: An instance of FlaskSocketPlatform.
        """
        super().__init__(namespace)
        self._platform = platform

    def on_connect(self, sid: str) -> None:
        """Connects client to platform."""
        self._platform.connect(sid)  # Use sid directly
        logger.info(f"Client connected; user_id: {sid}")

    def on_disconnect(self, sid: str) -> None:
        """Disconnects client from server."""
        self._platform.disconnect(sid)  # Use sid directly
        logger.info(f"Client disconnected; user_id: {sid}")

    def on_message(self, sid: str, data: dict) -> None:
        """Receives message from client and sends response.

        Args:
            sid: Session ID from client.
            data: Data received from client.
        """
        self._platform.message(sid, data["message"])  # Use sid directly
        logger.info(f"Message received from {sid}: {data}")

    def on_feedback(self, sid: str, data: dict) -> None:
        """Receives feedback from client.

        Args:
            sid: Session ID from client.
            data: Data received from client.
        """
        logger.info(f"Feedback received from {sid}: {data}")
        self._platform.feedback(sid, **data["feedback"])  # Use sid directly