import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { FaSun, FaMoon } from 'react-icons/fa';
import { useTheme } from '../context/ThemeContext';
import './Navbar.css';

const Navbar = () => {
  const location = useLocation();
  const { isDarkMode, toggleTheme } = useTheme();

  return (
    <nav className="navbar">
      <div className="navbar-brand">
        <Link to="/" className="navbar-logo">AI Dictionary</Link>
      </div>
      <div className="navbar-links">
        <Link 
          to="/" 
          className={`nav-link ${location.pathname === '/' ? 'active' : ''}`}
        >
          Search
        </Link>
        {/* <Link 
          to="/analysis" 
          className={`nav-link ${location.pathname === '/analysis' ? 'active' : ''}`}
        >
          Analysis
        </Link> */}
        <Link 
          to="/history" 
          className={`nav-link ${location.pathname === '/history' ? 'active' : ''}`}
        >
          History
        </Link>
        <Link 
          to="/frequency" 
          className={`nav-link ${location.pathname === '/frequency' ? 'active' : ''}`}
        >
          Frequency
        </Link>
        {/* <button 
          className="theme-toggle-nav"
          onClick={toggleTheme}
          aria-label="Toggle theme"
        >
          {isDarkMode ? <FaSun className="theme-icon" /> : <FaMoon className="theme-icon" />}
        </button> */}
      </div>
    </nav>
  );
};

export default Navbar; 