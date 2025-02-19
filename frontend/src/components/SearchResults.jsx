import React from 'react';
import './SearchResults.css'; // Import your CSS file

const SearchResults = ({ word, result }) => {
  console.log('Search Results:', result);

  if (!result) {
    return <div className="loading">
        
    </div>;
  }

  return (
    <div className="results-container">
      <h2 className="word-title">{word}</h2>
      
      <div className="section meanings-section">
        <h3>Meanings</h3>
        <ul className="meanings-list">
          {Array.isArray(result.meanings) && result.meanings.length > 0 ? (
            result.meanings.map((meaning, index) => (
              <li key={index} className="meaning-item">{meaning}</li>
            ))
          ) : (
            <li>No meanings found.</li>
          )}
        </ul>
      </div>

      <div className="section sentiment-section">
        <h3>Sentiment Analysis</h3>
        <div className={`sentiment-container ${result.sentiment.label.toLowerCase()}`}>
          <span className="sentiment-label">{result.sentiment.label}</span>
          <p className="sentiment-score">Confidence: <span>{result.sentiment.score.toFixed(2)}%</span></p>
        </div>
      </div>

      <div className="section similar-words-section">
        <h3>Similar Words</h3>
        <ul className="similar-words-list">
          {Array.isArray(result.similar_words) && result.similar_words.length > 0 ? (
            result.similar_words.map((word, index) => (
              <li key={index} className="similar-word-item">{word}</li>
            ))
          ) : (
            <li>No similar words found.</li>
          )}
        </ul>
      </div>

      <div className="section context-examples-section">
        <h3>Context Examples</h3>
        <ul className="context-examples-list">
          {Array.isArray(result.context_examples) && result.context_examples.length > 0 ? (
            result.context_examples.map((example, index) => (
              <li key={index} className="context-example-item">{example}</li>
            ))
          ) : (
            <li>No context examples found.</li>
          )}
        </ul>
      </div>
    </div>
  );
};

export default SearchResults;
