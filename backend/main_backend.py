import threading
import time
import cv2
from hand_tracking_backend import calibrate_interactive, capture_thread, processing_loop

def main():
    print("🖐️ Starting Touchless Interaction Backend...")
    print("Initializing camera and calibration...")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Unable to access camera.")
        return

    # Step 1: Calibrate interactively
    calibration_data = calibrate_interactive(cap)
    print(f"✅ Calibration complete: {calibration_data}")

    # Step 2: Launch background threads
    stop_flag = threading.Event()

    capture_t = threading.Thread(target=capture_thread, args=(cap, stop_flag), daemon=True)
    process_t = threading.Thread(target=processing_loop, args=(stop_flag, calibration_data), daemon=True)

    capture_t.start()
    process_t.start()

    print("🚀 Hand-tracking backend is running. Press 'Ctrl+C' to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping backend gracefully...")
        stop_flag.set()
        capture_t.join()
        process_t.join()
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Shutdown complete.")

if __name__ == "__main__":
    main()