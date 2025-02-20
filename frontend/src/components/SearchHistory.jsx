import React, { useState, useEffect } from 'react';
import { getSearchHistory } from '../api/api';
import './SearchHistory.css';

const SearchHistory = () => {
  const [history, setHistory] = useState([]);
  const [currentPage, setCurrentPage] = useState(1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [itemsPerPage] = useState(10);

  // Fetch search history
  const fetchHistory = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getSearchHistory();
      setHistory(Array.isArray(data) ? data : []);
    } catch (err) {
      setError(err.message || 'Failed to fetch search history');
      console.error('Error fetching history:', err);
    } finally {
      setLoading(false);
    }
  };

  // Fetch history on component mount
  useEffect(() => {
    fetchHistory();
  }, []);

  // Format date function
  const formatDate = (isoString) => {
    if (!isoString) return 'N/A';
    
    try {
      const date = new Date(isoString);
      if (isNaN(date.getTime())) return 'N/A';
      
      return date.toLocaleString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        hour12: true
      });
    } catch (error) {
      console.error('Date formatting error:', error);
      return 'N/A';
    }
  };

  // Calculate pagination
  const indexOfLastItem = currentPage * itemsPerPage;
  const indexOfFirstItem = indexOfLastItem - itemsPerPage;
  const currentItems = history.slice(indexOfFirstItem, indexOfLastItem);
  const totalPages = Math.ceil(history.length / itemsPerPage);

  return (
    <div className="search-history">
      <h2>Search History</h2>
      
      <button 
        onClick={fetchHistory} 
        className="refresh-button"
        disabled={loading}
      >
        {loading ? 'Loading...' : '🔄 Refresh History'}
      </button>

      {loading && (
        <div className="loading">Loading search history...</div>
      )}
      
      {error && (
        <div className="error-message">
          <p>Error: {error}</p>
          <button 
            onClick={fetchHistory}
            className="retry-button"
          >
            Try Again
          </button>
        </div>
      )}
      
      {!loading && !error && history.length > 0 && (
        <>
          <div className="history-list">
            {currentItems.map((item, index) => (
              <div key={`${item.word}-${index}`} className="history-item">
                <span className="word">{item.word}</span>
                <span className="count">
                  {item.count} {item.count === 1 ? 'search' : 'searches'}
                </span>
                <span className="timestamp">
                  Last searched: {formatDate(item.last_searched)}
                </span>
              </div>
            ))}
          </div>

          {totalPages > 1 && (
            <div className="pagination">
              <button 
                onClick={() => setCurrentPage(prev => Math.max(prev - 1, 1))}
                disabled={currentPage === 1}
                className="pagination-button"
              >
                Previous
              </button>
              
              <span className="page-info">
                Page {currentPage} of {totalPages}
              </span>
              
              <button 
                onClick={() => setCurrentPage(prev => Math.min(prev + 1, totalPages))}
                disabled={currentPage === totalPages}
                className="pagination-button"
              >
                Next
              </button>
            </div>
          )}
        </>
      )}

      {!loading && !error && history.length === 0 && (
        <div className="no-data">No search history available</div>
      )}
    </div>
  );
};

export default SearchHistory; 