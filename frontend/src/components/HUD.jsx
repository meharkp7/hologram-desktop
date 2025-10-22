import React from "react";

const HUD = () => (
  <div style={{
    position: "fixed",
    top: "10px",
    right: "10px",
    background: "rgba(16, 13, 40, 0.8)",
    color: "#acd9da",
    padding: "10px 20px",
    borderRadius: "12px",
    fontFamily: "monospace",
    fontSize: "14px",
    zIndex: 9999
  }}>
    Work | Presentation | Media
  </div>
);

export default HUD;