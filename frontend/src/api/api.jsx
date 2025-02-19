import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL;

export const searchWord = async (word) => {
  try {
    const response = await axios.get(`${API_URL}/meaning/${word}`);
    return response.data; // Ensure this returns the expected structure
  } catch (error) {
    throw error.response?.data || { detail: 'Failed to fetch word meaning' };
  }
};

export const getSearchHistory = async (page = 1, limit = 10) => {
  try {
    const response = await axios.get(`${API_URL}/history?page=${page}&limit=${limit}`);
    return response.data;
  } catch (error) {
    throw error.response?.data || { detail: 'Failed to fetch search history' };
  }
};

export const getSearchHistoryWithCounts = async () => {
  try {
    const response = await axios.get(`${API_URL}/search-history`);
    return response.data;
  } catch (error) {
    throw error.response?.data || { detail: 'Failed to fetch search history' };
  }
};
