import { useState } from 'react';
import { ChevronDown, ChevronUp, Network, Search, FileSearch, CheckCircle2, Repeat2 } from 'lucide-react';
import './TraceViewer.css';

const getNodeIcon = (nodeName) => {
  switch (nodeName) {
    case 'router': return <Network size={16} className="trace-icon text-blue" />;
    case 'query_rewriter': return <Repeat2 size={16} className="trace-icon text-purple" />;
    case 'rag_lookup': return <FileSearch size={16} className="trace-icon text-green" />;
    case 'web_search': return <Search size={16} className="trace-icon text-orange" />;
    case 'answer': return <CheckCircle2 size={16} className="trace-icon text-primary" />;
    default: return <CheckCircle2 size={16} className="trace-icon text-gray" />;
  }
};

export default function TraceViewer({ traces }) {
  const [expanded, setExpanded] = useState(false);

  if (!traces || traces.length === 0) return null;

  return (
    <div className="trace-viewer">
      <button className="trace-toggle" onClick={() => setExpanded(!expanded)}>
        <Network size={16} />
        <span>Agent Execution Trace ({traces.length} steps)</span>
        {expanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
      </button>

      {expanded && (
        <div className="trace-content">
          {traces.map((trace, index) => (
            <div key={index} className="trace-step">
              <div className="trace-step-header">
                <span className="step-number">{trace.step}</span>
                {getNodeIcon(trace.node_name)}
                <span className="node-name">{trace.node_name}</span>
              </div>
              <div className="trace-desc">
                {trace.description}
              </div>
              {trace.details && Object.keys(trace.details).length > 0 && (
                <div className="trace-details">
                  {Object.entries(trace.details).map(([key, value]) => (
                    <div key={key} className="trace-detail-item">
                      <strong>{key}:</strong> {typeof value === 'object' ? JSON.stringify(value) : value}
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
