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
    trackID VARCHAR(255)
);