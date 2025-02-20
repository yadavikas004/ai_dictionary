import React, { useState, useEffect } from 'react';
import { getSearchHistoryWithCounts } from '../api/api';
import './SearchHistory.css';

const SearchHistoryWithCounts = () => {
  const [searchHistory, setSearchHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchSearchHistory = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getSearchHistoryWithCounts();
      setSearchHistory(Array.isArray(data) ? data : []);
    } catch (err) {
      setError(err.message || 'Failed to fetch search frequency');
      console.error('Error fetching frequency:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSearchHistory();
  }, []);

  return (
    <div className="search-history-container">
      <h2>Search Frequency</h2>
      
      <button 
        onClick={fetchSearchHistory} 
        className="refresh-button"
        disabled={loading}
      >
        {loading ? 'Loading...' : '🔄 Refresh'}
      </button>

      {loading && <div className="loading">Loading search frequency...</div>}
      
      {error && (
        <div className="error-message">
          <p>Error: {error}</p>
          <button 
            onClick={fetchSearchHistory}
            className="retry-button"
          >
            Try Again
          </button>
        </div>
      )}

      {!loading && !error && searchHistory.length > 0 && (
        <div className="frequency-list">
          {searchHistory.map((item, index) => (
            <div key={index} className="frequency-item">
              <span className="word">{item.word}</span>
              <span className="frequency-badge">
                {item.count} {item.count === 1 ? 'search' : 'searches'}
              </span>
            </div>
          ))}
        </div>
      )}

      {!loading && !error && searchHistory.length === 0 && (
        <div className="no-data">No search data available</div>
      )}
    </div>
  );
};

export default SearchHistoryWithCounts; 