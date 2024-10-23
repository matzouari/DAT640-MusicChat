from typing import List

class Track():
    def __init__(self, name, artist = None, album = None, genre = None, duration = None):
        self.name = name
        self.artist = artist 
        self.album = album
        self.genre = genre
        self.duration = duration

    def __str__(self):
        return f"Track: {self.name}, Artist: {self.artist}, Album: {self.album}, Genre: {self.genre}, Duration: {self.duration}"

class Playlist():
    def __init__(self, name: str, tracks: List[Track]):
        self.name = name
        self.tracks = [] if tracks is None else tracks

    def __str__(self):
        return f"Playlist: {self.name}. Tracks: {[str(track) for track in self.tracks]}"
    
    def add_track(self, track):
        if track not in self.tracks:
            self.tracks.append(track)
            return True
        else:
            return False
        
    def remove_track(self, track):
        if track in self.tracks:
            self.tracks.remove(track)
            return True
        else:
            return False