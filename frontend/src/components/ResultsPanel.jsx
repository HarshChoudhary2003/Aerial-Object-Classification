import React, { useRef } from 'react';
import { UploadCloud, Search, AlertCircle } from 'lucide-react';

const ResultsPanel = ({ 
  previewUrl, 
  results, 
  error, 
  confThreshold, 
  onFileSelect, 
  onReset 
}) => {
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onFileSelect(e.dataTransfer.files[0]);
    }
  };

  return (
    <div className="glass-panel results-container fade-in" style={{animationDelay: '0.1s'}}>
      {!previewUrl ? (
        <div 
          className="upload-zone"
          onClick={() => fileInputRef.current?.click()}
          onDragOver={handleDragOver}
          onDrop={handleDrop}
        >
          <UploadCloud className="upload-icon" />
          <div className="upload-text">Drag & drop aerial footage here</div>
          <p style={{color: 'var(--text-secondary)', fontSize: '0.9rem'}}>Supports JPG, PNG (Max 50MB)</p>
          <input 
            type="file" 
            ref={fileInputRef} 
            onChange={(e) => {
              if (e.target.files && e.target.files[0]) {
                onFileSelect(e.target.files[0]);
              }
            }} 
            accept="image/jpeg, image/png" 
            style={{display: 'none'}} 
          />
        </div>
      ) : (
        <>
          <div className="image-preview">
            {results?.detection?.image_base64 ? (
              <img src={`data:image/jpeg;base64,${results.detection.image_base64}`} alt="Detection Results" />
            ) : (
              <img src={previewUrl} alt="Preview" />
            )}
          </div>
          
          <div style={{display:'flex', justifyContent:'space-between', alignItems:'center'}}>
            <button className="btn-primary" style={{width: 'auto', background: 'rgba(255,255,255,0.1)'}} onClick={onReset}>
              <Search size={16} /> New Analysis
            </button>
            {results?.detection?.image_base64 && (
              <a 
                href={`data:image/jpeg;base64,${results.detection.image_base64}`}
                download={`detected_result.jpg`}
                className="btn-primary"
                style={{width: 'auto', textDecoration: 'none'}}
              >
                Download Result
              </a>
            )}
          </div>

          {error && (
            <div style={{padding: '1rem', background: 'rgba(239, 68, 68, 0.2)', borderLeft: '4px solid #ef4444', borderRadius: '8px', display: 'flex', gap: '0.5rem', alignItems: 'center'}}>
              <AlertCircle color="#ef4444" /> {error}
            </div>
          )}

          {results && (
            <div className="metrics-grid">
              {results.classification && (
                <div className={`glass-panel metric-card ${results.classification.label === 'DRONE' ? 'drone' : 'bird'}`}>
                  <div className="metric-label">Classification</div>
                  <div className="metric-value" style={{color: results.classification.label === 'DRONE' ? 'var(--danger)' : 'var(--success)'}}>
                    {results.classification.label === 'DRONE' ? '🚁 DRONE' : '🐦 BIRD'}
                  </div>
                  <div style={{color: 'var(--text-secondary)', fontSize: '0.9rem'}}>
                    Confidence: {(results.classification.confidence * 100).toFixed(1)}%
                  </div>
                </div>
              )}
              
              {results.detection && (
                <div className="glass-panel metric-card">
                  <div className="metric-label">Objects Detected</div>
                  <div className="metric-value">{results.detection.count}</div>
                  <div style={{color: 'var(--text-secondary)', fontSize: '0.9rem'}}>
                    Threshold: {confThreshold}
                  </div>
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default ResultsPanel;
