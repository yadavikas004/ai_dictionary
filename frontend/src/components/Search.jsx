import React, { useState } from 'react';
import axios from 'axios';
import ReactPaginate from 'react-paginate';

const Search = () => {
  const [query, setQuery] = useState('');
  const [suggestions, setSuggestions] = useState([]);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [currentPage, setCurrentPage] = useState(0);
  const [itemsPerPage] = useState(10);

  const handleChange = async (e) => {
    const value = e.target.value;
    setQuery(value);

    if (value) {
      const response = await axios.get(`${process.env.REACT_APP_API_URL}/suggestions?query=${value}`);
      setSuggestions(response.data);
    } else {
      setSuggestions([]);
    }
  };

  const handleSearch = async () => {
    setLoading(true);
    try {
      const response = await axios.post(`${process.env.REACT_APP_API_URL}/search-history`, { word: query });
      setResult(response.data);
    } catch (error) {
      setError('Failed to fetch data. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handlePageChange = (data) => {
    setCurrentPage(data.selected);
  };

  const downloadHistory = async () => {
    const response = await axios.get(`${process.env.REACT_APP_API_URL}/download-history`, { responseType: 'blob' });
    const url = window.URL.createObjectURL(new Blob([response.data]));
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', 'search_history.csv');
    document.body.appendChild(link);
    link.click();
  };

  if (loading) {
    return <div className="loading">Loading...</div>;
  }

  if (error) {
    return <div className="error">{error}</div>;
  }

  return (
    <div>
      <input type="text" value={query} onChange={handleChange} placeholder="Search..." />
      <button onClick={handleSearch}>Search</button>
      <button onClick={downloadHistory}>Download History</button>
      <ul>
        {suggestions.map((suggestion) => (
          <li key={suggestion.id}>{suggestion.word}</li>
        ))}
      </ul>

      {result && (
        <div>
          <h2>Results for: {query}</h2>
          <ul>
            {result.meanings.map((meaning, index) => (
              <li key={index}>{index + 1 + currentPage * itemsPerPage}. {meaning}</li>
            ))}
          </ul>

          <ReactPaginate
            previousLabel={"← Previous"}
            nextLabel={"Next →"}
            breakLabel={"..."}
            pageCount={Math.ceil(result.totalCount / itemsPerPage)}
            marginPagesDisplayed={2}
            pageRangeDisplayed={3}
            onPageChange={handlePageChange}
            containerClassName={"pagination"}
            activeClassName={"active"}
          />
        </div>
      )}
    </div>
  );
};

export default Search; 