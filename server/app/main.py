from dialoguekit.platforms.flask_socket_platform import FlaskSocketPlatform
from agent import MusicAgent

platform = FlaskSocketPlatform(MusicAgent)
app = platform.app

if __name__ == '__main__':
    platform.start()