import { useState } from 'react';
import { uploadDocument } from '../api';
import { UploadCloud, CheckCircle, AlertCircle, Loader2 } from 'lucide-react';
import './Sidebar.css';

export default function Sidebar({ webSearchEnabled, setWebSearchEnabled }) {
  const [file, setFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState('idle'); // idle, uploading, success, error
  const [statusMessage, setStatusMessage] = useState('');

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploadStatus('uploading');
    setStatusMessage('Uploading and indexing document...');
    try {
      const result = await uploadDocument(file);
      setUploadStatus('success');
      setStatusMessage(result.message || 'Document indexed successfully!');
    } catch (error) {
      setUploadStatus('error');
      setStatusMessage(error.message);
    }
  };

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <h2>Agent Config</h2>
      </div>

      <div className="sidebar-section">
        <h3>Knowledge Base Upload</h3>
        <p className="sidebar-desc">Upload PDF documents to expand the agent's knowledge.</p>
        
        <div className="file-input-wrapper">
          <input 
            type="file" 
            accept="application/pdf" 
            onChange={handleFileChange}
            id="file-upload"
            className="file-input"
          />
          <label htmlFor="file-upload" className="file-label">
            <UploadCloud size={20} />
            {file ? file.name : 'Choose a PDF file'}
          </label>
        </div>

        <button 
          className="upload-btn" 
          onClick={handleUpload} 
          disabled={!file || uploadStatus === 'uploading'}
        >
          {uploadStatus === 'uploading' ? (
            <><Loader2 size={16} className="spinner" /> Indexing...</>
          ) : 'Upload to Vector Store'}
        </button>

        {uploadStatus === 'success' && (
          <div className="status-toast success">
            <CheckCircle size={16} />
            <span>{statusMessage}</span>
          </div>
        )}
        {uploadStatus === 'error' && (
          <div className="status-toast error">
            <AlertCircle size={16} />
            <span>{statusMessage}</span>
          </div>
        )}
      </div>

      <div className="sidebar-section">
        <h3>Agent Settings</h3>
        <div className="toggle-wrapper">
          <label className="toggle-label">
            Enable Web Search
            <span className="toggle-desc">Allow agent to search the web for current events</span>
          </label>
          <label className="switch">
            <input 
              type="checkbox" 
              checked={webSearchEnabled} 
              onChange={(e) => setWebSearchEnabled(e.target.checked)} 
            />
            <span className="slider round"></span>
          </label>
        </div>
      </div>
    </aside>
  );
}
