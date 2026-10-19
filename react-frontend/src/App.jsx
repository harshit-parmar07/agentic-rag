import { useState, useEffect } from 'react';
import { RotateCcw } from 'lucide-react';
import Sidebar from './components/Sidebar';
import ChatFeed from './components/ChatFeed';
import ChatInput from './components/ChatInput';
import { chatWithAgent } from './api';
import './App.css';

const createWelcomeMessage = () => ({
  id: crypto.randomUUID(),
  sender: 'agent',
  text: 'Hello! Upload a PDF to your session knowledge base or ask me anything.',
  traces: [],
});

export default function App() {
  const [sessionId, setSessionId] = useState('');
  const [messages, setMessages] = useState([]);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [webSearchEnabled, setWebSearchEnabled] = useState(true);
  const [isTyping, setIsTyping] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    let storedSessionId = localStorage.getItem('chat_session_id');
    if (!storedSessionId) {
      storedSessionId = crypto.randomUUID();
      localStorage.setItem('chat_session_id', storedSessionId);
    }
    setSessionId(storedSessionId);

    const storedMessages = localStorage.getItem('chat_messages');
    if (storedMessages) {
      try {
        setMessages(JSON.parse(storedMessages));
      } catch (e) {
        console.error('Failed to parse stored messages', e);
      }
    } else {
      setMessages((prev) => (prev.length === 0 ? [createWelcomeMessage()] : prev));
    }

    const storedFiles = localStorage.getItem('chat_uploaded_files');
    if (storedFiles) {
      try {
        setUploadedFiles(JSON.parse(storedFiles));
      } catch (e) {
        console.error('Failed to parse stored uploaded files', e);
      }
    }
  }, []);

  useEffect(() => {
    if (messages.length > 0) {
      localStorage.setItem('chat_messages', JSON.stringify(messages));
    }
  }, [messages]);

  useEffect(() => {
    localStorage.setItem('chat_uploaded_files', JSON.stringify(uploadedFiles));
  }, [uploadedFiles]);

  const handleNewChat = () => {
    const newSessionId = crypto.randomUUID();
    const initialMessages = [createWelcomeMessage()];
    localStorage.setItem('chat_session_id', newSessionId);
    localStorage.setItem('chat_messages', JSON.stringify(initialMessages));
    localStorage.removeItem('chat_uploaded_files');
    setSessionId(newSessionId);
    setMessages(initialMessages);
    setUploadedFiles([]);
    setError(null);
  };

  const handleFileUploaded = (fileInfo) => {
    setUploadedFiles((prev) => {
      const existingIndex = prev.findIndex((item) => item.name === fileInfo.name);
      if (existingIndex !== -1) {
        const updated = [...prev];
        updated[existingIndex] = fileInfo;
        return updated;
      }
      return [...prev, fileInfo];
    });
  };

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
        sessionId={sessionId}
        webSearchEnabled={webSearchEnabled}
        setWebSearchEnabled={setWebSearchEnabled}
        uploadedFiles={uploadedFiles}
        onFileUploaded={handleFileUploaded}
      />
      <main className="main-content">
        <header className="main-header">
          <h1>Agent Workspace</h1>
          <div className="header-actions">
            <span className="session-badge">Session: {sessionId.substring(0, 8)}...</span>
            <button
              type="button"
              className="new-chat-btn"
              onClick={handleNewChat}
              disabled={isTyping}
              title="Start a new isolated session"
            >
              <RotateCcw size={14} />
              <span>New Chat</span>
            </button>
          </div>
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
