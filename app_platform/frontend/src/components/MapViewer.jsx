import React from 'react';
import { MapContainer, TileLayer, Rectangle, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

export default function MapViewer({ queryData, evidence }) {
  const primaryEvidence = evidence?.[0] || {};
  const bboxes = primaryEvidence?.results?.bounding_boxes || [];
  const toolName = primaryEvidence?.tool || 'N/A';
  const target = primaryEvidence?.target || 'N/A';
  const mathFormula = primaryEvidence?.math || 'None';
  const confidence = primaryEvidence?.confidence ? (primaryEvidence.confidence * 100).toFixed(1) : '0.0';

  const handleExportReport = () => {
    const reportData = {
      timestamp: new Date().toISOString(),
      query: queryData,
      evidence: evidence,
    };
    const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `satquery_report_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      {/* Audit HUD Badge (Top-Right) */}
      <div style={{
        position: 'absolute', top: 12, right: 12, zIndex: 1000,
        background: 'rgba(15, 23, 42, 0.85)', color: '#fff',
        padding: '10px 14px', borderRadius: '8px', fontSize: '12px',
        backdropFilter: 'blur(4px)', border: '1px solid rgba(255, 255, 255, 0.1)'
      }}>
        <div><strong>Tool:</strong> {toolName}</div>
        <div><strong>Modality/Target:</strong> {target}</div>
        <div><strong>Math:</strong> <code>{mathFormula}</code></div>
        <div><strong>Confidence:</strong> {confidence}%</div>
        <button 
          onClick={handleExportReport}
          style={{
            marginTop: '8px', width: '100%', padding: '4px 8px',
            background: '#2563eb', color: '#fff', border: 'none',
            borderRadius: '4px', cursor: 'pointer', fontWeight: '600'
          }}
        >
          Export Report (.JSON)
        </button>
      </div>

      <MapContainer center={[20.5937, 78.9629]} zoom={5} style={{ height: '100%', width: '100%' }}>
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; OpenStreetMap contributors'
        />
        {/* Render Bounding Box Rectangles */}
        {bboxes.map((box, idx) => {
          // Leaflet bounds format: [[latMin, lonMin], [latMax, lonMax]]
          const bounds = [[box[1], box[0]], [box[3], box[2]]];
          return (
            <Rectangle key={idx} bounds={bounds} pathOptions={{ color: '#ef4444', weight: 2 }}>
              <Popup>Detected Region #{idx + 1}</Popup>
            </Rectangle>
          );
        })}
      </MapContainer>
    </div>
  );
}