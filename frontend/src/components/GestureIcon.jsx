import React from "react";
import { motion } from "framer-motion";

const GestureIcon = ({ icon, active }) => (
  <motion.div
    animate={{ scale: active ? 1.5 : 1, rotate: active ? [0, 10, -10, 0] : 0 }}
    transition={{ duration: 0.6 }}
    style={{
      margin: "0 15px",
      color: active ? "#acd9da" : "#7375db",
      fontSize: "28px",
      cursor: "pointer"
    }}
  >
    {icon}
  </motion.div>
);

export default GestureIcon;