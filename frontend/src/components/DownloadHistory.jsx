import React from 'react';

const DownloadHistory = () => {
    const handleDownload = async () => {
        const response = await fetch('/api/download-history');
        if (!response.ok) {
            console.error('Failed to download history:', response.statusText);
            return;
        }
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'search_history.csv';
        document.body.appendChild(a);
        a.click();
        a.remove();
    };

    return (
        <button onClick={handleDownload}>
            📥 Download Search History
        </button>
    );
};

export default DownloadHistory; 