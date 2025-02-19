import React from 'react';

const VoiceSearch = ({ onSearch }) => {
    const startVoiceRecognition = () => {
        const recognition = new window.SpeechRecognition();
        recognition.lang = 'en-US'; // Set the language
        recognition.interimResults = false; // Set to true for interim results

        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            onSearch(transcript); // Trigger search with the recognized text
        };

        recognition.onerror = (event) => {
            console.error('Voice recognition error:', event.error);
        };

        recognition.start();
    };

    return (
        <button onClick={startVoiceRecognition}>
            🎤 Voice Search
        </button>
    );
};

export default VoiceSearch;
