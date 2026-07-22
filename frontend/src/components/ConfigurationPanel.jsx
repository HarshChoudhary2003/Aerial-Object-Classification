import React from 'react';
import { Settings, Activity } from 'lucide-react';

const ConfigurationPanel = ({ 
  task, setTask, 
  modelType, setModelType, 
  confThreshold, setConfThreshold, 
  onAnalyze, loading, selectedFile 
}) => {
  return (
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
        onClick={onAnalyze}
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
  );
};

export default ConfigurationPanel;
