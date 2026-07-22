import React from 'react';
import { Clock } from 'lucide-react';

const HistoryPanel = ({ history, onLoadHistoryItem }) => {
  return (
    <aside className="glass-panel history-panel fade-in" style={{animationDelay: '0.2s'}}>
      <h3><Clock size={18} style={{display:'inline', verticalAlign:'middle', marginRight: '8px'}} /> Session History</h3>
      
      {history.length === 0 ? (
        <div style={{color: 'var(--text-secondary)', textAlign: 'center', padding: '2rem 0', fontSize: '0.9rem'}}>
          No analyses yet. Upload an image to start.
        </div>
      ) : (
        history.map((item) => (
          <div key={item.id} className="history-item" onClick={() => onLoadHistoryItem(item)}>
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
  );
};

export default HistoryPanel;
