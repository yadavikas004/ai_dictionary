import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    "Content-Type": "application/json",
    "Accept": "application/json",
  },
  timeout: 10000,
});

// Add response interceptor for error handling
api.interceptors.response.use(
  response => response,
  error => {
    console.error('API Error:', error);
    if (error.response) {
      // Server responded with error
      throw new Error(error.response.data.detail || 'An error occurred');
    } else if (error.request) {
      // Request made but no response
      throw new Error('No response from server');
    } else {
      // Error setting up request
      throw new Error('Error setting up request');
    }
  }
);

export const searchWord = async (word) => {
  try {
    const response = await api.get(`/meaning/${word}`);
    return response.data;
  } catch (error) {
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
