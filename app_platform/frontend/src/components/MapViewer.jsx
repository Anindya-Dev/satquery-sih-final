import React, { useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Rectangle, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

// Recenter component to handle dynamic map zooming when new bbox arrives
const MapRecenter = ({ bounds }) => {
  const map = useMap();
  useEffect(() => {
    if (bounds) {
      map.fitBounds(bounds, { padding: [30, 30] });
    }
  }, [bounds, map]);
  return null;
};

const MapViewer = ({ evidenceData }) => {
  const defaultCenter = [20.5937, 78.9629]; // Default: India Center

  let bounds = null;
  let center = defaultCenter;

  // Transform bbox [min_lon, min_lat, max_lon, max_lat] -> Leaflet [[south, west], [north, east]]
  if (evidenceData?.bbox && Array.isArray(evidenceData.bbox) && evidenceData.bbox.length === 4) {
    const [minLon, minLat, maxLon, maxLat] = evidenceData.bbox;
    bounds = [
      [minLat, minLon],
      [maxLat, maxLon]
    ];
    center = [(minLat + maxLat) / 2, (minLon + maxLon) / 2];
  }

  return (
    <div style={{ height: '100%', width: '100%', position: 'relative' }}>
      <MapContainer center={defaultCenter} zoom={5} style={{ height: '100%', width: '100%', borderRadius: '8px' }}>
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        />
        
        {bounds && <MapRecenter bounds={bounds} />}

        {/* Dynamic bounding box overlay */}
        {bounds && (
          <Rectangle
            bounds={bounds}
            pathOptions={{ color: '#d9534f', weight: 2, fillColor: '#d9534f', fillOpacity: 0.35 }}
          />
        )}

        {/* Evidence location marker */}
        {evidenceData && (
          <Marker position={center}>
            <Popup>
              <div>
                <strong>Pipeline Analysis</strong><br />
                Target: {evidenceData.target || 'N/A'}<br />
                Affected Area: {evidenceData.affected_area_pct ?? 'N/A'}%<br />
                Confidence: {evidenceData.confidence ?? 'N/A'}
              </div>
            </Popup>
          </Marker>
        )}
      </MapContainer>
    </div>
  );
};

export default MapViewer;