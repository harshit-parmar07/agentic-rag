import TraceViewer from './TraceViewer';
import { Bot, User } from 'lucide-react';
import './MessageBubble.css';

export default function MessageBubble({ message }) {
  const isAgent = message.sender === 'agent';

  return (
    <div className={`message-wrapper ${isAgent ? 'agent' : 'user'}`}>
      <div className="message-avatar">
        {isAgent ? <Bot size={20} /> : <User size={20} />}
      </div>
      <div className="message-content">
        <div className="message-text">
          {message.text}
        </div>
        {isAgent && message.traces && message.traces.length > 0 && (
          <TraceViewer traces={message.traces} />
        )}
      </div>
    </div>
  );
}
