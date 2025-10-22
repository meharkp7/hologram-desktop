import { useEffect, useRef, useState, useCallback } from "react";

/**
 * useWebSocket(url, {reconnectInterval, autoStart})
 * returns { lastMessage, status, sendJson }
 */
export default function useWebSocket(url, opts = {}) {
  const { reconnectInterval = 2000, autoStart = true } = opts;
  const wsRef = useRef(null);
  const [lastMessage, setLastMessage] = useState(null);
  const [status, setStatus] = useState("idle"); // connecting, open, closed, error
  const reconnectTimer = useRef(null);

  const connect = useCallback(() => {
    if (!url) return;
    setStatus("connecting");
    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        setStatus("open");
        if (reconnectTimer.current) {
          clearTimeout(reconnectTimer.current);
          reconnectTimer.current = null;
        }
      };

      ws.onmessage = (ev) => {
        setLastMessage(ev.data);
      };

      ws.onclose = () => {
        setStatus("closed");
        wsRef.current = null;
        // try reconnect
        reconnectTimer.current = setTimeout(() => connect(), reconnectInterval);
      };

      ws.onerror = (e) => {
        setStatus("error");
        // will trigger onclose eventually
      };
    } catch (e) {
      setStatus("error");
      reconnectTimer.current = setTimeout(() => connect(), reconnectInterval);
    }
  }, [url, reconnectInterval]);

  useEffect(() => {
    if (!autoStart) return;
    connect();
    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, [connect, autoStart]);

  const sendJson = useCallback((obj) => {
    try {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify(obj));
        return true;
      }
      return false;
    } catch (e) {
      return false;
    }
  }, []);

  return { lastMessage, status, sendJson };
}