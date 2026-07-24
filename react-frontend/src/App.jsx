import { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import ChatFeed from './components/ChatFeed';
import ChatInput from './components/ChatInput';
import { chatWithAgent } from './api';
import './App.css';

export default function App() {
  const [sessionId, setSessionId] = useState('');
  const [messages, setMessages] = useState([]);
  const [webSearchEnabled, setWebSearchEnabled] = useState(true);
  const [isTyping, setIsTyping] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Initialize session ID from localStorage or create a new one
    let storedSessionId = localStorage.getItem('chat_session_id');
    if (!storedSessionId) {
      storedSessionId = crypto.randomUUID();
      localStorage.setItem('chat_session_id', storedSessionId);
    }
    setSessionId(storedSessionId);

    // Initialize messages from localStorage if available
    const storedMessages = localStorage.getItem('chat_messages');
    if (storedMessages) {
      try {
        setMessages(JSON.parse(storedMessages));
      } catch (e) {
        console.error('Failed to parse stored messages', e);
      }
    } else {
      // Welcome message
      setMessages(prev => prev.length === 0 ? [{
        id: crypto.randomUUID(),
        sender: 'agent',
        text: 'Hello! How can I help you today?',
        traces: []
      }] : prev);
    }
  }, []);

  // Save messages to localStorage whenever they change
  useEffect(() => {
    if (messages.length > 0) {
      localStorage.setItem('chat_messages', JSON.stringify(messages));
    }
  }, [messages]);

  const handleSendMessage = async (text) => {
    const userMessage = {
      id: crypto.randomUUID(),
      sender: 'user',
      text,
      traces: []
    };
    
    setMessages(prev => [...prev, userMessage]);
    setIsTyping(true);
    setError(null);

    try {
      const result = await chatWithAgent(sessionId, text, webSearchEnabled);
      
      const agentMessage = {
        id: crypto.randomUUID(),
        sender: 'agent',
        text: result.response || "Sorry, I couldn't generate a response.",
        traces: result.trace_events || []
      };
      
      setMessages(prev => [...prev, agentMessage]);
    } catch (err) {
      setError(err.message);
      const errorMessage = {
        id: crypto.randomUUID(),
        sender: 'agent',
        text: `Error: ${err.message}`,
        traces: []
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsTyping(false);
    }
  };

  return (
    <div className="app-container">
      <Sidebar 
        webSearchEnabled={webSearchEnabled} 
        setWebSearchEnabled={setWebSearchEnabled} 
      />
      <main className="main-content">
        <header className="main-header">
          <h1>Agent Workspace</h1>
          <span className="session-badge">Session: {sessionId.substring(0, 8)}...</span>
        </header>
        <div className="chat-container">
          <ChatFeed messages={messages} isTyping={isTyping} />
        </div>
        <div className="input-container">
          {error && <div className="error-banner">{error}</div>}
          <ChatInput onSendMessage={handleSendMessage} disabled={isTyping} />
        </div>
      </main>
    </div>
  );
}
