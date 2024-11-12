import mysql.connector
from flask import Flask, jsonify
from mysql.connector import connect, Error

from customPlatform import musicPlatform
from nl_agent import MusicAgent

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


# Initialize global database connection
conn = connect_to_db()  # Call the function and store the connection globally

platform = musicPlatform(MusicAgent)
app = platform.app

@app.route('/get_playlist', methods=['GET'])
def get_playlist():
    """API endpoint to fetch playlist for the front end."""
    try:
        if conn and conn.is_connected():
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT track_name, artist_name, album_name, genre, popularity FROM playlist")
            songs = cursor.fetchall()
            cursor.close()
            return jsonify(songs)
        else:
            return jsonify({"error": "Database connection is not available"}), 500
    except Error as err:
        print(f"Error: {err}")
        return jsonify({"error": "Failed to fetch playlist"}), 500


if __name__ == '__main__':
    platform.start()
