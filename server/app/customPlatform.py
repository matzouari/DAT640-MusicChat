"""The Platform facilitates displaying of the conversation."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Type, cast

from flask import Flask, Request, request
from flask_cors import CORS
from flask_socketio import Namespace, SocketIO

import mysql.connector

from dialoguekit.core import AnnotatedUtterance
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


class musicPlatform(Platform):
    def __init__(self, agent_class: Type[Agent]) -> None:
        """Represents a platform that uses Flask-SocketIO.

        Args:
            agent_class: The class of the agent.
        """
        self.app = Flask(__name__)
        CORS(self.app, origins="*", support_credentials=True)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")

        super().__init__(agent_class)

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
    def __init__(self, namespace: str, platform: musicPlatform) -> None:
        """Represents a namespace.

        Args:
            namespace: Namespace.
            platform: An instance of FlaskSocketPlatform.
        """
        super().__init__(namespace)
        self._platform = platform

    def on_connect(self) -> None:
        """Connects client to platform."""
        req: SocketIORequest = cast(SocketIORequest, request)
        self._platform.connect(req.sid)
        logger.info(f"Client connected; user_id: {req.sid}")

    def on_disconnect(self) -> None:
        """Disconnects client from server."""
        req: SocketIORequest = cast(SocketIORequest, request)
        self._platform.disconnect(req.sid)
        logger.info(f"Client disconnected; user_id: {req.sid}")

    def on_message(self, data: dict) -> None:
        """Receives message from client and sends response.

        Args:
            data: Data received from client.
        """
        req: SocketIORequest = cast(SocketIORequest, request)
        self._platform.message(req.sid, data["message"])
        logger.info(f"Message received: {data}")

    def on_feedback(self, data: dict) -> None:
        """Receives feedback from client.

        Args:
            data: Data received from client.
        """
        req: SocketIORequest = cast(SocketIORequest, request)
        logger.info(f"Feedback received: {data}")
        self._platform.feedback(req.sid, **data["feedback"])