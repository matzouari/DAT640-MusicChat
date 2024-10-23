DROP TABLE IF EXISTS TracksPlaylist;
DROP TABLE IF EXISTS Tracks;
DROP TABLE IF EXISTS Playlist;

CREATE TABLE Tracks (
    id VARCHAR(255) PRIMARY KEY,
    track_name VARCHAR(255),
    artist_name VARCHAR(255),
    album_name VARCHAR(255),
    duration_ms INT,
    release_date DATE,
    danceability FLOAT,
    energy FLOAT,
    loudness FLOAT,
    speechiness FLOAT,
    acousticness FLOAT,
    instrumentalness FLOAT,
    liveness FLOAT,
    valence FLOAT,
    tempo FLOAT,
    genre VARCHAR(255),
    popularity INT DEFAULT NULL
);

CREATE TABLE Playlist (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(80) NOT NULL UNIQUE,
    creator VARCHAR(30) NOT NULL
);

CREATE TABLE TracksPlaylist (
    idTrack VARCHAR(255),
    idPlaylist INT,
    PRIMARY KEY (idTrack, idPlaylist),
    FOREIGN KEY (idTrack) REFERENCES Tracks(id),
    FOREIGN KEY (idPlaylist) REFERENCES Playlist(id)
);