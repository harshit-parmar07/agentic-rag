import { useEffect, useRef } from 'react';
import MessageBubble from './MessageBubble';
import { Loader2 } from 'lucide-react';
import './ChatFeed.css';

export default function ChatFeed({ messages, isTyping }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  return (
    <div className="chat-feed">
      {messages.length === 0 ? (
        <div className="empty-state">
          <h3>Welcome to the AI Assistant</h3>
          <p>Ask a question or upload a document to get started.</p>
        </div>
      ) : (
        messages.map((msg) => <MessageBubble key={msg.id} message={msg} />)
      )}
      
      {isTyping && (
        <div className="typing-indicator">
          <Loader2 size={16} className="spinner" />
          <span>Agent is thinking...</span>
        </div>
      )}
      <div ref={bottomRef} />
    </div>
  );
}
