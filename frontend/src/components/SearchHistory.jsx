import { useState, useEffect } from 'react';
import { getSearchHistory } from '../api/api'; // Ensure this function is defined in api.jsx

const SearchHistory = ({ history }) => {
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage] = useState(10); // Number of items per page
  const [paginatedHistory, setPaginatedHistory] = useState([]);

  useEffect(() => {
    const indexOfLastItem = currentPage * itemsPerPage;
    const indexOfFirstItem = indexOfLastItem - itemsPerPage;
    setPaginatedHistory(history.slice(indexOfFirstItem, indexOfLastItem));
  }, [currentPage, history]);

  const totalPages = Math.ceil(history.length / itemsPerPage);

  return (
    <div className="history-container">
      <h2>Search History</h2>
      
      <ul className="history-list">
        {paginatedHistory.map((item, index) => (
          <li key={index} className="history-item">
            {item}
          </li>
        ))}
      </ul>

      <div className="pagination">
        {Array.from({ length: totalPages }, (_, i) => (
          <button
            key={i + 1}
            onClick={() => setCurrentPage(i + 1)}
            className={i + 1 === currentPage ? 'active' : ''}
          >
            {i + 1}
          </button>
        ))}
      </div>
    </div>
  );
};

export default SearchHistory; 