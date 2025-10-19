import React, { useState, useEffect } from "react";

const CursorOverlay = () => {
  const [cursor, setCursor] = useState({ x: 100, y: 100 });
  const [zoom, setZoom] = useState(100);

  // Dummy cursor movement for now
  useEffect(() => {
    const interval = setInterval(() => {
      setCursor(prev => ({ x: (prev.x + 5) % window.innerWidth, y: (prev.y + 3) % window.innerHeight }));
      setZoom(prev => prev < 200 ? prev + 1 : 100);
    }, 50);
    return () => clearInterval(interval);
  }, []);

  return (
    <>
      <div
        style={{
          position: "fixed",
          left: cursor.x,
          top: cursor.y,
          width: "20px",
          height: "20px",
          backgroundColor: "magenta",
          borderRadius: "50%",
          zIndex: 9999,
          pointerEvents: "none"
        }}
      />
      <div
        style={{
          position: "fixed",
          top: 10,
          left: 10,
          color: "white",
          fontSize: "18px",
          zIndex: 9999
        }}
      >
        Zoom: {zoom}%
      </div>
    </>
  );
};

export default CursorOverlay;