import React from 'react';
import { Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';
import './SearchResults.css'; // Import your CSS file

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

const SearchResults = ({ word, result }) => {
  console.log('Search Results:', result);

  if (!result) {
    return <div className="loading">Loading...</div>;
  }

  // Helper function for sentiment color
  const getSentimentColor = (score) => {
    if (score > 0.6) return '#28a745';
    if (score < 0.4) return '#dc3545';
    return '#ffc107';
  };

  // Ensure word_vector exists and is an array
  const vectorData = result.word_vector || [];

  // Chart configuration
  const chartData = {
    labels: vectorData.map((_, index) => `D${index + 1}`),
    datasets: [
      {
        label: 'Vector Dimensions',
        data: vectorData,
        backgroundColor: 'rgba(53, 162, 235, 0.5)',
        borderColor: 'rgba(53, 162, 235, 1)',
        borderWidth: 1,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Word Vector Representation',
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: 'Value'
        }
      },
      x: {
        title: {
          display: true,
          text: 'Dimensions'
        }
      }
    }
  };

  return (
    <div className="results-container">
      <h2 className="word-title">{word}</h2>
      
      <div className="section meanings-section">
        <h3>Meanings</h3>
        <div className="meanings-list">
          {result.meanings?.map((meaning, index) => (
            <div key={index} className="meaning-item">
              {index + 1}. {meaning}
            </div>
          ))}
        </div>
      </div>

      <div className="section vector-section">
        <h3>Word Vector Analysis</h3>
        <div className="vector-chart-container">
          <Bar data={chartData} options={chartOptions} />
        </div>
        <div className="vector-values">
          {vectorData.map((value, index) => (
            <div key={index} className="vector-value">
              <span className="dimension">D{index + 1}:</span>
              <span className="value">{value.toFixed(4)}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="section sentiment-section">
        <h3>Sentiment Analysis</h3>
        <div className="sentiment-container" 
             style={{ backgroundColor: `${getSentimentColor(result.sentiment.score)}20` }}>
          <div className="sentiment-label" 
               style={{ color: getSentimentColor(result.sentiment.score) }}>
            {result.sentiment.label}
          </div>
          <div className="sentiment-score">
            Confidence: {(result.sentiment.score * 100).toFixed(2)}%
          </div>
        </div>
      </div>

      <div className="section similar-words-section">
        <h3>Similar Words</h3>
        <div className="similar-words-list">
          {result.similar_words?.map((word, index) => (
            <span key={index} className="similar-word">{word}</span>
          ))}
        </div>
      </div>

      <div className="section context-examples-section">
        <h3>Context Examples</h3>
        <div className="context-examples-list">
          {result.context_examples?.map((example, index) => (
            <div key={index} className="context-example">"{example}"</div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default SearchResults;
