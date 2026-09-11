import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';

const ChatWindow = ({ onQueryResult }) => {
  const [query, setQuery] = useState('');
  const [patchId, setPatchId] = useState('patch_001');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  const handleSend = async (e) => {
    e.preventDefault();
    if (!query.trim() || loading) return;

    const userQuery = query;
    setQuery('');
    setMessages((prev) => [...prev, { sender: 'user', text: userQuery }]);
    setLoading(true);

    try {
      const response = await axios.post('http://127.0.0.1:8000/api/v1/query', {
        query: userQuery,
        patch_id: patchId,
      });

      setMessages((prev) => [
        ...prev,
        {
          sender: 'system',
          text: response.data.final_response,
          details: response.data,
        },
      ]);

      if (onQueryResult) {
        onQueryResult(response.data);
      }
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        { sender: 'system', text: 'Error: Unable to connect to satellite backend.' },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', background: '#ffffff', borderRadius: '8px', border: '1px solid #e0e0e0', boxShadow: '0 2px 8px rgba(0,0,0,0.05)', padding: '1rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem', paddingBottom: '0.5rem', borderBottom: '1px solid #f0f0f0' }}>
        <h3 style={{ fontSize: '1.1rem', color: '#1a1a1a' }}>SatQuery Assistant</h3>
        <select value={patchId} onChange={(e) => setPatchId(e.target.value)} style={{ padding: '0.25rem 0.5rem', borderRadius: '4px', border: '1px solid #ccc', fontSize: '0.85rem' }}>
          <option value="patch_001">Patch 001 (SAR/Opt)</option>
          <option value="patch_002">Patch 002 (SAR/Opt)</option>
        </select>
      </div>

      <div style={{ flex: 1, overflowY: 'auto', marginBottom: '0.75rem', paddingRight: '0.25rem' }}>
        {messages.length === 0 && (
          <p style={{ color: '#888', fontSize: '0.9rem', textAlign: 'center', marginTop: '2rem' }}>
            Ask a spatial query (e.g., "Check flood damage under clouds").
          </p>
        )}
        {messages.map((msg, index) => (
          <div key={index} style={{ textAlign: msg.sender === 'user' ? 'right' : 'left', margin: '0.5rem 0' }}>
            <div style={{ display: 'inline-block', maxWidth: '85%', padding: '0.6rem 0.8rem', borderRadius: '8px', background: msg.sender === 'user' ? '#007bff' : '#f1f3f5', color: msg.sender === 'user' ? '#fff' : '#212529', fontSize: '0.9rem', lineHeight: '1.4' }}>
              {msg.text}
            </div>
          </div>
        ))}
        {loading && (
          <div style={{ textAlign: 'left', margin: '0.5rem 0', color: '#6c757d', fontSize: '0.85rem' }}>
            <em>Processing satellite bands...</em>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <form onSubmit={handleSend} style={{ display: 'flex', gap: '0.5rem' }}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Type your query..."
          disabled={loading}
          style={{ flex: 1, padding: '0.6rem 0.8rem', borderRadius: '4px', border: '1px solid #ccc', outline: 'none', fontSize: '0.9rem' }}
        />
        <button type="submit" disabled={loading} style={{ padding: '0.6rem 1.2rem', borderRadius: '4px', border: 'none', background: '#007bff', color: '#fff', fontWeight: 'bold', cursor: loading ? 'not-allowed' : 'pointer' }}>
          Send
        </button>
      </form>
    </div>
  );
};

export default ChatWindow;