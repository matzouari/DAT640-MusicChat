from flask import Flask, request, jsonify
from flask_socketio import SocketIO, send
from flask_cors import CORS
from chatbot import get_bot_response

app = Flask(__name__)

# Enable CORS for your Flask app
CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize SocketIO and allow CORS
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route('/api/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    bot_response = get_bot_response(user_message)
    return jsonify({'reply': bot_response})

if __name__ == '__main__':
    socketio.run(app, debug=True)
