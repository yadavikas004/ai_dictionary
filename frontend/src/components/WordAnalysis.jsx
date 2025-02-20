import React, { useState } from 'react';
import { getAdvancedAnalysis } from '../api/api';
import './WordAnalysis.css';

const WordAnalysis = () => {
    const [word, setWord] = useState('');
    const [analysis, setAnalysis] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleAnalyze = async (e) => {
        e.preventDefault();
        if (!word.trim()) return;

        setLoading(true);
        setError(null);
        setAnalysis(null);

        try {
            const data = await getAdvancedAnalysis(word.trim());
            setAnalysis(data);
        } catch (err) {
            setError(err.message || 'Failed to analyze word. Please try again.');
            console.error('Analysis error:', err);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="word-analysis-container">
            <div className="analysis-header">
                <h2>Advanced Word Analysis</h2>
                <p className="analysis-subtitle">Get detailed linguistic and semantic analysis</p>
                
                <form onSubmit={handleAnalyze} className="analysis-form">
                    <div className="analysis-input-wrapper">
                        <input
                            type="text"
                            value={word}
                            onChange={(e) => setWord(e.target.value)}
                            placeholder="Enter a word to analyze..."
                            className="analysis-input"
                            disabled={loading}
                        />
                        <button 
                            type="submit" 
                            className="analyze-button"
                            disabled={!word.trim() || loading}
                        >
                            {loading ? 'Analyzing...' : 'Analyze'}
                        </button>
                    </div>
                </form>
            </div>

            {error && (
                <div className="error-message">
                    {error}
                </div>
            )}

            {loading && (
                <div className="loading-container">
                    <div className="loader"></div>
                    <p>Analyzing word...</p>
                </div>
            )}

            {analysis && (
                <div className="analysis-results">
                    {/* Basic Analysis */}
                    {analysis.basic_analysis && (
                        <div className="analysis-card">
                            <h3>Basic Analysis</h3>
                            <div className="card-content">
                                <p><strong>Word:</strong> {word}</p>
                                {analysis.basic_analysis.meanings && (
                                    <div className="meanings-section">
                                        <h4>Meanings:</h4>
                                        <ul>
                                            {analysis.basic_analysis.meanings.map((meaning, index) => (
                                                <li key={index}>{meaning}</li>
                                            ))}
                                        </ul>
                                    </div>
                                )}
                            </div>
                        </div>
                    )}

                    {/* ML Analysis */}
                    {analysis.ml_analysis && (
                        <div className="analysis-card">
                            <h3>ML Analysis</h3>
                            <div className="card-content">
                                {analysis.ml_analysis.sentiment && (
                                    <div className="sentiment-section">
                                        <h4>Sentiment Analysis:</h4>
                                        <p>Label: {analysis.ml_analysis.sentiment.label}</p>
                                        <p>Score: {(analysis.ml_analysis.sentiment.score * 100).toFixed(2)}%</p>
                                    </div>
                                )}
                                
                                {analysis.ml_analysis.similar_words && (
                                    <div className="similar-words-section">
                                        <h4>Similar Words:</h4>
                                        <div className="similar-words-list">
                                            {analysis.ml_analysis.similar_words.map((word, index) => (
                                                <span key={index} className="similar-word">{word}</span>
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>
                    )}

                    {/* POS Tags */}
                    {analysis.pos_tags && (
                        <div className="analysis-card">
                            <h3>Parts of Speech</h3>
                            <div className="pos-tags">
                                {analysis.pos_tags.map(([word, tag], index) => (
                                    <div key={index} className="pos-tag">
                                        <span className="word">{word}</span>
                                        <span className="tag">{tag}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

export default WordAnalysis; 