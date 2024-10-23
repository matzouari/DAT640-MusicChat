import React, { useEffect, useState } from 'react';
import io from 'socket.io-client';

const socket = io('http://localhost:5000'); // Connect to your Flask backend

function Chat() {
  const [messages, setMessages] = useState([]);

  // Load the chat widget script
  useEffect(() => {
    const script = document.createElement('script');
    script.src = "https://cdn.jsdelivr.net/npm/iaigroup-chatwidget@latest/build/bundle.min.js"; // Replace with the correct URL
    script.async = true;

    // Add the script to the document
    document.body.appendChild(script);

    // Initialize the chat widget after the script loads
    script.onload = () => {
      window.ChatWidget.init({
        onSend: handleNewUserMessage,  // Set the send message handler
        messages: [],  // Initialize messages
      });
    };

    // Cleanup function to remove the script on component unmount
    return () => {
      document.body.removeChild(script);
    };
  }, []);

  // Handle new message from the user
  const handleNewUserMessage = (message) => {
    const newMessage = { sender: 'user', text: message };
    setMessages((prevMessages) => [...prevMessages, newMessage]);

    // Send the message to the server via Socket.IO
    socket.emit('message', message); // Emit message to the server
  };

  // Listen for incoming messages from the server
  useEffect(() => {
    socket.on('message', (msg) => {
      const botReply = { sender: 'bot', text: msg }; // Create a bot reply object
      setMessages((prevMessages) => [...prevMessages, botReply]); // Update messages state
    });

    // Cleanup on component unmount
    return () => {
      socket.off('message'); // Remove the listener
    };
  }, []);

  return (
    <div>
      {/* ChatWidget will be initialized in the script */}
      {/* Optionally, you can also render messages here if needed */}
      <div>
        {messages.map((msg, index) => (
          <div key={index} className={msg.sender}>
            {msg.text}
          </div>
        ))}
      </div>
    </div>
  );
}

export default Chat;
