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

platform = musicPlatform(MusicAgent)
app = platform.app


if __name__ == '__main__':
    platform.start()
