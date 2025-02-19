import React, { useState, useEffect } from 'react';
import { getSearchHistoryWithCounts } from '../api/api'; // Create this function in api.jsx

const SearchHistoryWithCounts = () => {
  const [searchHistory, setSearchHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchSearchHistory = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getSearchHistoryWithCounts();
      setSearchHistory(data);
    } catch (err) {
      setError(err.detail || 'Failed to fetch search history');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="search-history-container">
      <button onClick={fetchSearchHistory} className="fetch-history-button">
        Fetch Search History
      </button>

      {loading && <p>Loading...</p>}
      {error && <p className="error-message">{error}</p>}

      {searchHistory.length > 0 && (
        <ul className="search-history-list">
          {searchHistory.map((item, index) => (
            <li key={index} className="search-history-item">
              {item.word}: {item.count} times
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default SearchHistoryWithCounts; 