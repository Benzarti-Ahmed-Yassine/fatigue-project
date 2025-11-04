// frontend/src/App.js
import React, { useRef, useState } from 'react';
import Webcam from 'react-webcam';
import { predictFromBase64 } from './api';

function App() {
  const webcamRef = useRef(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const captureAndSend = async () => {
    const screenshot = webcamRef.current.getScreenshot();
    if (!screenshot) {
      alert('Impossible de capturer la webcam.');
      return;
    }
    const base64 = screenshot.split(',')[1];
    setLoading(true);
    try {
      const res = await predictFromBase64(base64);
      setResult(res);
    } catch (err) {
      console.error(err);
      alert('Erreur: ' + (err?.response?.data?.error || err.message));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: 20, fontFamily: 'Arial, sans-serif' }}>
      <h2>Détection de Fatigue - TPE</h2>
      <Webcam audio={false} ref={webcamRef} screenshotFormat="image/png" width={320} />
      <div style={{ marginTop: 10 }}>
        <button onClick={captureAndSend} disabled={loading} style={{ padding: '10px 20px' }}>
          {loading ? 'Analyse...' : 'Analyser'}
        </button>
      </div>

      {result && (
        <div style={{ marginTop: 20, padding: 15, border: '1px solid #ccc', borderRadius: 8 }}>
          <p><strong>État :</strong> {result.label} ({(result.probability * 100).toFixed(1)}%)</p>
          <p><strong>Recommandation :</strong> {result.recommendation}</p>
          <details>
            <summary>Voir les probabilités détaillées</summary>
            <pre>{JSON.stringify(result.probabilities, null, 2)}</pre>
          </details>
        </div>
      )}
    </div>
  );
}

export default App;