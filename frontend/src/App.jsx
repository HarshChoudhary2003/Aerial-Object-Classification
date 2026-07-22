import React, { useState, useRef } from 'react';
import axios from 'axios';
import { UploadCloud, Settings, Activity, AlertCircle, Clock, Search } from 'lucide-react';
import './index.css';

const API_URL = "http://localhost:8000";

function App() {
  const [task, setTask] = useState('both');
  const [modelType, setModelType] = useState('resnet50');
  const [confThreshold, setConfThreshold] = useState(0.5);
  
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  
  // Session History State
  const [history, setHistory] = useState([]);

  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      if (file.size > 50 * 1024 * 1024) {
        setError("File too large. Max 50MB.");
        return;
      }
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResults(null);
      setError(null);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResults(null);
      setError(null);
    }
  };

  const analyzeImage = async () => {
    if (!selectedFile) return;
    
    setLoading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("task", task);
    formData.append("cls_model_type", modelType);
    formData.append("conf_threshold", confThreshold);

    try {
      const response = await axios.post(`${API_URL}/api/analyze/image`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      
      const newResults = response.data.results;
      setResults(newResults);
      
      // Add to session history
      const historyItem = {
        id: Date.now(),
        filename: selectedFile.name,
        previewUrl: previewUrl,
        results: newResults,
        task: task,
        modelType: modelType,
        timestamp: new Date().toLocaleTimeString()
      };
      setHistory(prev => [historyItem, ...prev]);
      
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.detail || "An error occurred during analysis.");
    } finally {
      setLoading(false);
    }
  };

  const loadHistoryItem = (item) => {
    // We don't have the original File object easily reconstructible, 
    // but we have the previewUrl and results.
    setPreviewUrl(item.previewUrl);
    setResults(item.results);
    setTask(item.task);
    setModelType(item.modelType);
    setError(null);
  };

  return (
    <div className="app-container">
      <header className="glass-panel glass-header fade-in">
        <h1 className="title-gradient">AERIAL SURVEILLANCE AI</h1>
        <p className="subtitle">Next-Gen Computer Vision for Airspace Safety & Wildlife Monitoring</p>
      </header>

      <main className="main-content">
        {/* Left Sidebar - Configuration */}
        <aside className="glass-panel sidebar fade-in">
          <h3><Settings size={18} style={{display:'inline', verticalAlign:'middle', marginRight: '8px'}} /> Configuration</h3>
          
          <div className="control-group">
            <label>Select Task</label>
            <select value={task} onChange={(e) => setTask(e.target.value)}>
              <option value="classification">Classification Only</option>
              <option value="detection">Detection Only</option>
              <option value="both">Both Tasks</option>
            </select>
          </div>

          <div className="control-group">
            <label>Classification Model</label>
            <select value={modelType} onChange={(e) => setModelType(e.target.value)} disabled={task === 'detection'}>
              <option value="resnet50">ResNet50 Transfer Learning</option>
              <option value="cnn">Custom CNN</option>
            </select>
          </div>

          <div className="control-group">
            <label>Detection Confidence ({confThreshold})</label>
            <input 
              type="range" 
              min="0.1" max="0.9" step="0.05" 
              value={confThreshold} 
              onChange={(e) => setConfThreshold(parseFloat(e.target.value))}
              disabled={task === 'classification'}
            />
          </div>

          <button 
            className="btn-primary" 
            onClick={analyzeImage}
            disabled={!selectedFile || loading}
            style={{marginTop: 'auto'}}
          >
            {loading ? (
              <><div className="loader-spinner" style={{width:'20px',height:'20px',borderWidth:'2px'}}></div> Processing...</>
            ) : (
              <><Activity size={18} /> Analyze Target</>
            )}
          </button>
        </aside>

        {/* Center Main Area - Results */}
        <div className="glass-panel results-container fade-in" style={{animationDelay: '0.1s'}}>
          {!previewUrl ? (
            <div 
              className="upload-zone"
              onClick={() => fileInputRef.current.click()}
              onDragOver={handleDragOver}
              onDrop={handleDrop}
            >
              <UploadCloud className="upload-icon" />
              <div className="upload-text">Drag & drop aerial footage here</div>
              <p style={{color: 'var(--text-secondary)', fontSize: '0.9rem'}}>Supports JPG, PNG (Max 50MB)</p>
              <input 
                type="file" 
                ref={fileInputRef} 
                onChange={handleFileChange} 
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
                <button className="btn-primary" style={{width: 'auto', background: 'rgba(255,255,255,0.1)'}} onClick={() => {
                  setSelectedFile(null);
                  setPreviewUrl(null);
                  setResults(null);
                  setError(null);
                }}>
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
        
        {/* Right Sidebar - Session History */}
        <aside className="glass-panel history-panel fade-in" style={{animationDelay: '0.2s'}}>
          <h3><Clock size={18} style={{display:'inline', verticalAlign:'middle', marginRight: '8px'}} /> Session History</h3>
          
          {history.length === 0 ? (
            <div style={{color: 'var(--text-secondary)', textAlign: 'center', padding: '2rem 0', fontSize: '0.9rem'}}>
              No analyses yet. Upload an image to start.
            </div>
          ) : (
            history.map((item) => (
              <div key={item.id} className="history-item" onClick={() => loadHistoryItem(item)}>
                <img 
                  src={item.results?.detection?.image_base64 ? `data:image/jpeg;base64,${item.results.detection.image_base64}` : item.previewUrl} 
                  alt="History thumbnail" 
                />
                
                <div style={{marginBottom: '0.5rem', fontSize: '0.85rem', color: 'var(--text-secondary)'}}>
                  {item.timestamp} • {item.filename.length > 15 ? item.filename.substring(0,15)+'...' : item.filename}
                </div>
                
                <div>
                  <span className="badge model">{item.modelType === 'resnet50' ? 'ResNet50' : 'CNN'}</span>
                  
                  {item.results?.classification && (
                     <span className={`badge ${item.results.classification.label === 'DRONE' ? 'drone' : 'bird'}`}>
                        {item.results.classification.label}
                     </span>
                  )}
                  
                  {item.results?.detection && (
                    <span className="badge" style={{background: 'rgba(255,255,255,0.1)'}}>
                      Obj: {item.results.detection.count}
                    </span>
                  )}
                </div>
              </div>
            ))
          )}
        </aside>
      </main>
    </div>
  );
}

export default App;
