// frontend/src/api.js
import axios from 'axios';

const API_BASE = process.env.NODE_ENV === 'production' 
  ? 'http://backend:5000' 
  : 'http://localhost:5000';

export async function predictFromBase64(base64) {
  const payload = { image: base64 };
  const res = await axios.post(`${API_BASE}/predict`, payload);
  return res.data;
}