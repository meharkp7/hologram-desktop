import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";

const CursorOverlay = ({ gesture }) => {
  const [cursor, setCursor] = useState({ x: 100, y: 100 });
  const [zoom, setZoom] = useState(100);
  const [zoomDirection, setZoomDirection] = useState(1);
  const [ripple, setRipple] = useState(false);

  useEffect(() => {
    const interval = setInterval(() => {
      setCursor(prev => ({
        x: (prev.x + 7 + Math.random()*3) % window.innerWidth,
        y: (prev.y + 5 + Math.random()*3) % window.innerHeight
      }));
      setZoom(prev => {
        if (prev >= 180) setZoomDirection(-1);
        if (prev <= 100) setZoomDirection(1);
        return prev + zoomDirection*2;
      });
    }, 30);
    return () => clearInterval(interval);
  }, [zoomDirection]);

  useEffect(() => {
    if (gesture === "click") {
      setRipple(true);
      setTimeout(() => setRipple(false), 400);
    }
  }, [gesture]);

  return (
    <>
      <motion.div
        animate={{ scale: [1, 1.4, 1], boxShadow: ["0 0 10px #acd9da", "0 0 25px #7375db", "0 0 10px #acd9da"] }}
        transition={{ duration: 0.8, repeat: Infinity }}
        style={{
          position: "fixed",
          left: cursor.x,
          top: cursor.y,
          width: "25px",
          height: "25px",
          backgroundColor: "#acd9da",
          borderRadius: "50%",
          zIndex: 9999,
          pointerEvents: "none"
        }}
      />
      {ripple && (
        <motion.div
          initial={{ scale: 0, opacity: 0.8 }}
          animate={{ scale: 2.5, opacity: 0 }}
          transition={{ duration: 0.4 }}
          style={{
            position: "fixed",
            left: cursor.x - 10,
            top: cursor.y - 10,
            width: "45px",
            height: "45px",
            borderRadius: "50%",
            border: "2px solid #7375db",
            zIndex: 9998,
            pointerEvents: "none"
          }}
        />
      )}
      <div
        style={{
          position: "fixed",
          top: 15,
          left: 15,
          color: "#ffffff",
          fontSize: "18px",
          zIndex: 9999,
          fontFamily: "monospace"
        }}
      >
        Zoom: {zoom}%
      </div>
    </>
  );
};

export default CursorOverlay;