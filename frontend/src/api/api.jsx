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

export const searchWord = async (word) => {
  try {
    const response = await api.get(`/meaning/${word}`);
    return response.data;
  } catch (error) {
    console.error('API Error:', error);
    if (error.response) {
      // The request was made and the server responded with a status code
      // that falls out of the range of 2xx
      throw new Error(error.response.data.detail || 'Failed to fetch word meaning');
    } else if (error.request) {
      // The request was made but no response was received
      throw new Error('No response from server. Please check your connection.');
    } else {
      // Something happened in setting up the request
      throw new Error('Error setting up the request');
    }
  }
};

export const getSearchHistory = async () => {
  try {
    const response = await api.get('/search-history');
    return response.data;
  } catch (error) {
    console.error('History Error:', error);
    throw new Error('Failed to fetch search history');
  }
};

export const getSearchHistoryWithCounts = async () => {
  try {
    const response = await api.get(`/search-history`);
    return response.data;
  } catch (error) {
    throw error.response?.data || { detail: 'Failed to fetch search history' };
  }
};
