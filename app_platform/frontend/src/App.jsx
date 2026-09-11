import React, { useState } from 'react';
import ChatWindow from './components/ChatWindow';
import MapViewer from './components/MapViewer';

function App() {
  const [pipelineResult, setPipelineResult] = useState(null);

  return (
    <div style={{ display: 'flex', height: '100vh', width: '100vw', padding: '1rem', gap: '1rem', boxSizing: 'border-box' }}>
      <div style={{ width: '380px', height: 'calc(100vh - 2rem)', flexShrink: 0 }}>
        <ChatWindow onQueryResult={setPipelineResult} />
      </div>
      <div style={{ flex: 1, height: 'calc(100vh - 2rem)' }}>
        <MapViewer evidenceData={pipelineResult?.evidence} />
      </div>
    </div>
  );
}

export default App;