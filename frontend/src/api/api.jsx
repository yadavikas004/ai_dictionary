import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    "Content-Type": "application/json",
    "Accept": "application/json",
  },
  timeout: 15000,
  withCredentials: true
});

// Add request interceptor
api.interceptors.request.use(
  (config) => {
    // Ensure the URL is properly formatted
    if (!config.url.startsWith('http')) {
      config.url = `${API_URL}${config.url}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Add response interceptor
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.code === 'ERR_NETWORK') {
      throw new Error('Network error. Please check your connection.');
    }
    if (error.response) {
      throw new Error(error.response.data?.detail || 'An error occurred');
    }
    throw error;
  }
);

export const searchWord = async (word) => {
  try {
    const response = await api.get(`/meaning/${encodeURIComponent(word)}`);
    return response.data;
  } catch (error) {
    console.error('Search error:', error);
    throw error;
  }
};

export const getSearchHistory = async () => {
  try {
    const response = await api.get('/search-history');
    return response.data;
  } catch (error) {
    throw error;
  }
};

export const getSearchHistoryWithCounts = async () => {
  try {
    const response = await api.get('/search-history/counts');
    return response.data;
  } catch (error) {
    throw error;
  }
};

export const getAdvancedAnalysis = async (word) => {
  try {
    const response = await api.get(`/advanced/${encodeURIComponent(word)}`);
    return response.data;
  } catch (error) {
    console.error('Advanced analysis error:', error);
    throw error;
  }
};
