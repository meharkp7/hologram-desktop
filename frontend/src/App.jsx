import React, { useState, useEffect } from "react";
import CursorOverlay from "./components/CursorOverlay";
import HUD from "./components/HUD";
import GestureIcon from "./components/GestureIcon";
import { FaHandPointer, FaArrowsAlt, FaSearchPlus, FaSearchMinus } from "react-icons/fa";

const App = () => {
  const [activeGesture, setActiveGesture] = useState("none");
  const [notification, setNotification] = useState("");
  const gestures = [
    { name: "click", icon: <FaHandPointer /> },
    { name: "scroll", icon: <FaArrowsAlt /> },
    { name: "zoom-in", icon: <FaSearchPlus /> },
    { name: "zoom-out", icon: <FaSearchMinus /> },
  ];

  useEffect(() => {
    const interval = setInterval(() => {
      const randomGesture = gestures[Math.floor(Math.random() * gestures.length)].name;
      setActiveGesture(randomGesture);
      setNotification(
        randomGesture === "click"
          ? "Click detected!"
          : randomGesture === "scroll"
          ? "Scrolling..."
          : randomGesture === "zoom-in"
          ? "Zooming In"
          : "Zooming Out"
      );
      setTimeout(() => setNotification(""), 1200); // notification fades
    }, 1800);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="app-container">
      <HUD />
      <CursorOverlay gesture={activeGesture} />
      <div className="gesture-panel">
        {gestures.map((g) => (
          <GestureIcon key={g.name} icon={g.icon} active={activeGesture === g.name} />
        ))}
      </div>
      {notification && (
        <div className="notification">
          {notification}
        </div>
      )}
    </div>
  );
};

export default App;