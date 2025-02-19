import React, { useState } from 'react';
import api from './api/index';
import SearchResults from './components/SearchResults';
import Dictionary from './components/Dictionary';
import SearchHistory from './components/SearchHistory';
import SearchHistoryWithCounts from './components/SearchHistoryWithCounts';
import VoiceSearch from './components/VoiceSearch';
import DownloadHistory from './components/DownloadHistory';
import './App.css';
import { searchWord } from './api/api';

function App() {
  const [searchTerm, setSearchTerm] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [searchHistory, setSearchHistory] = useState([]);
  const [result, setResult] = useState(null);

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!searchTerm.trim()) return;

    setLoading(true);
    setError(null);
    try {
      const data = await searchWord(searchTerm);
      console.log('API Response:', data);
      setResult(data);
      setSearchResults(data);
      updateSearchHistory(searchTerm);
    } catch (err) {
      setError(err.detail || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (e) => {
    setSearchTerm(e.target.value);
    if (error) setError(null);
  };

  const updateSearchHistory = (newWord) => {
    setSearchHistory((prevHistory) => [...prevHistory, newWord]);
  };

  const handleVoiceSearch = async (query) => {
    const response = await fetch(`/api/search?query=${query}`);
    const data = await response.json();
    setResult(data);
    setSearchResults(data);
    updateSearchHistory(query);
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>AI Dictionary</h1>
        <p className="subtitle">Search for word meanings and definitions</p>
      </header>

      <main className="main-content">
        <Dictionary updateSearchHistory={updateSearchHistory} />
        <SearchHistory history={searchHistory} />
        <SearchHistoryWithCounts />

        {/* <div className="search-section">
          <form onSubmit={handleSearch}>
            <input
              type="text"
              value={searchTerm}
              onChange={handleInputChange}
              placeholder="Enter a word to search..."
              className="search-input"
              disabled={loading}
            />
            <button 
              type="submit" 
              className="search-button"
              disabled={loading || !searchTerm.trim()}
            >
              {loading ? 'Searching...' : 'Search'}
            </button>
          </form>
        </div> */}

        <VoiceSearch onSearch={handleVoiceSearch} />
        <DownloadHistory />

        {error && (
          <div className="error-message">
            {error}
          </div>
        )}

        {loading && (
          <div className="loading-message">
            Searching for meanings...
          </div>
        )}

        {result && (
          <div className="results-section">
            <h2>Meanings:</h2>
            <ul className="meanings-list">
              {result.map((meaning, index) => (
                <li key={index} className="meaning-item">
                  {meaning}
                </li>
              ))}
            </ul>
          </div>
        )}

        <SearchResults word={searchTerm} result={result} />
      </main>

      <footer className="app-footer">
        <p>© 2024 AI Dictionary. All rights reserved.</p>
      </footer>
    </div>
  );
}

export default App;
