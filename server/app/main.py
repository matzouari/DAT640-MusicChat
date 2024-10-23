from customPlatform import musicPlatform
from agent import MusicAgent

platform = musicPlatform(MusicAgent)
app = platform.app

if __name__ == '__main__':
    platform.start()