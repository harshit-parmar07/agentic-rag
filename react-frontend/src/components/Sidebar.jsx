import { useState } from 'react';
import { uploadDocument } from '../api';
import { UploadCloud, CheckCircle, AlertCircle, Loader2, FileText } from 'lucide-react';
import './Sidebar.css';

const MAX_FILE_SIZE_BYTES = 4.5 * 1024 * 1024; // 4.5 MB

export default function Sidebar({
  sessionId,
  webSearchEnabled,
  setWebSearchEnabled,
  uploadedFiles = [],
  onFileUploaded,
}) {
  const [file, setFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState('idle'); // idle, uploading, success, error
  const [statusMessage, setStatusMessage] = useState('');

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const selected = e.target.files[0];
      if (selected.size > MAX_FILE_SIZE_BYTES) {
        setUploadStatus('error');
        setStatusMessage('File exceeds the 4.5 MB maximum limit.');
        setFile(null);
        e.target.value = '';
        return;
      }
      setUploadStatus('idle');
      setStatusMessage('');
      setFile(selected);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploadStatus('uploading');
    setStatusMessage('Uploading and indexing document...');
    try {
      const result = await uploadDocument(file, sessionId);
      setUploadStatus('success');
      setStatusMessage(result.message || 'Document indexed successfully!');
      if (onFileUploaded) {
        onFileUploaded({
          name: result.filename || file.name,
          chunks: result.processed_chunks || 0,
        });
      }
      setFile(null);
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

        {uploadedFiles.length > 0 && (
          <div className="indexed-files-wrapper">
            <h4 className="indexed-files-title">Session Documents ({uploadedFiles.length})</h4>
            <ul className="indexed-files-list">
              {uploadedFiles.map((doc, idx) => (
                <li key={`${doc.name}-${idx}`} className="indexed-file-item">
                  <FileText size={14} className="indexed-file-icon" />
                  <span className="indexed-file-name" title={doc.name}>{doc.name}</span>
                  <span className="indexed-file-badge">{doc.chunks} chunks</span>
                </li>
              ))}
            </ul>
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
