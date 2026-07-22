import axios from 'axios';

const API_URL = "http://localhost:8000";

export const analyzeImage = async (file, task, modelType, confThreshold) => {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("task", task);
    formData.append("cls_model_type", modelType);
    formData.append("conf_threshold", confThreshold);

    const response = await axios.post(`${API_URL}/api/analyze/image`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
    });
    
    return response.data.results;
};
