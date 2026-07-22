import React, { useState } from 'react';
import Header from './components/Header';
import ConfigurationPanel from './components/ConfigurationPanel';
import ResultsPanel from './components/ResultsPanel';
import HistoryPanel from './components/HistoryPanel';
import { analyzeImage } from './services/api';
import './styles/index.css';
import './styles/App.css';

function App() {
  const [task, setTask] = useState('both');
  const [modelType, setModelType] = useState('resnet50');
  const [confThreshold, setConfThreshold] = useState(0.5);
  
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  
  const [history, setHistory] = useState([]);

  const handleFileSelect = (file) => {
    if (file.size > 50 * 1024 * 1024) {
      setError("File too large. Max 50MB.");
      return;
    }
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResults(null);
    setError(null);
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;
    
    setLoading(true);
    setError(null);

    try {
      const newResults = await analyzeImage(selectedFile, task, modelType, confThreshold);
      setResults(newResults);
      
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
    setPreviewUrl(item.previewUrl);
    setResults(item.results);
    setTask(item.task);
    setModelType(item.modelType);
    setError(null);
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setResults(null);
    setError(null);
  };

  return (
    <div className="app-container">
      <Header />
      <main className="main-content">
        <ConfigurationPanel 
          task={task} setTask={setTask}
          modelType={modelType} setModelType={setModelType}
          confThreshold={confThreshold} setConfThreshold={setConfThreshold}
          onAnalyze={handleAnalyze}
          loading={loading}
          selectedFile={selectedFile}
        />
        <ResultsPanel 
          previewUrl={previewUrl}
          results={results}
          error={error}
          confThreshold={confThreshold}
          onFileSelect={handleFileSelect}
          onReset={handleReset}
        />
        <HistoryPanel 
          history={history}
          onLoadHistoryItem={loadHistoryItem}
        />
      </main>
    </div>
  );
}

export default App;
