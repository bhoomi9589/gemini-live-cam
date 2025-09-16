from flask import Flask, request, jsonify, Response
import asyncio
import threading
import cv2
import mss
import numpy as np
import time
from live import AudioLoop
import queue # New import for thread-safe queue

app = Flask(__name__)

# -----------------------
# Global state
# -----------------------
audio_loop_instance = None
loop_thread = None
status = "stopped"   # running | paused | stopped
mode = "none"  # current mode
shared_frame_queue = queue.Queue(maxsize=5) # For passing frames to the live thread

# -----------------------
# Rate limiting (no changes)
# -----------------------
last_request_time = {}
REQUEST_INTERVAL = 10.0

def is_rate_limited(endpoint: str) -> bool:
    now = time.time()
    if endpoint not in last_request_time or now - last_request_time[endpoint] >= REQUEST_INTERVAL:
        last_request_time[endpoint] = now
        return False
    return True

# -----------------------
# Session Handling
# -----------------------
def run_event_loop(selected_mode="camera", frame_queue=None):
    global audio_loop_instance, status, mode
    # Pass the shared queue to the AudioLoop instance
    audio_loop_instance = AudioLoop(video_mode=selected_mode, frame_queue=frame_queue)
    status = "running"
    mode = selected_mode
    asyncio.run(audio_loop_instance.run())

@app.route("/")
def index():
    return "API is running"

@app.route("/status", methods=["GET"])
def get_status():
    return jsonify({"status": status, "mode": mode})

@app.route("/start", methods=["POST"])
def start_session():
    global loop_thread, status, mode, shared_frame_queue

    if is_rate_limited("start"):
        return jsonify({"status": "error", "message": "Too many requests"}), 429

    requested_mode = request.json.get("mode", "camera")

    if status == "running":
        return jsonify({"status": "error", "message": f"Session already running in {mode} mode"}), 400

    # Clear any old frames from the queue before starting
    while not shared_frame_queue.empty():
        try:
            shared_frame_queue.get_nowait()
        except queue.Empty:
            break
            
    loop_thread = threading.Thread(target=run_event_loop, args=(requested_mode, shared_frame_queue), daemon=True)
    loop_thread.start()
    status = "running"
    mode = requested_mode
    return jsonify({"status": "success", "message": f"Session started in {mode} mode"})

# --- (pause, resume, and stop endpoints remain largely the same) ---

@app.route("/pause", methods=["POST"])
def pause_session():
    # ... (code is the same as original)
    global audio_loop_instance, status

    if is_rate_limited("pause"):
        return jsonify({"status": "error", "message": "Too many requests"}), 429

    if audio_loop_instance is None:
        return jsonify({"status": "error", "message": "No session running"}), 400

    audio_loop_instance.running = False
    status = "paused"
    return jsonify({"status": "success", "message": "Session paused"})


@app.route("/resume", methods=["POST"])
def resume_session():
    # ... (code is the same as original)
    global loop_thread, status, mode

    if is_rate_limited("resume"):
        return jsonify({"status": "error", "message": "Too many requests"}), 429

    if status != "paused":
        return jsonify({"status": "error", "message": "No paused session"}), 400

    loop_thread = threading.Thread(target=run_event_loop, args=(mode, shared_frame_queue), daemon=True)
    loop_thread.start()
    status = "running"
    return jsonify({"status": "success", "message": "Session resumed"})


@app.route("/stop", methods=["POST"])
def stop_session():
    global audio_loop_instance, loop_thread, status, mode

    if is_rate_limited("stop"):
        return jsonify({"status": "error", "message": "Too many requests"}), 429

    if audio_loop_instance is None:
        return jsonify({"status": "error", "message": "No session running"}), 400

    try:
        audio_loop_instance.stop()
        audio_loop_instance = None
        if loop_thread:
            loop_thread.join(timeout=2)
            loop_thread = None
        status = "stopped"
        mode = "none"
        return jsonify({"status": "success", "message": "Session stopped"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# NEW endpoint for receiving frames from the frontend
@app.route("/upload_frame", methods=["POST"])
def upload_frame():
    global shared_frame_queue
    if status != "running":
        return jsonify({"status": "error", "message": "Session not running"}), 400

    frame_file = request.files.get('frame')
    if not frame_file:
        return jsonify({"status": "error", "message": "No frame received"}), 400

    try:
        if not shared_frame_queue.full():
            np_img = np.frombuffer(frame_file.read(), np.uint8)
            img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
            if img is not None:
                # Put the raw image (NumPy array) into the queue
                shared_frame_queue.put(img)
                return jsonify({"status": "success"})
    except Exception:
        # Queue might be full, or other errors.
        pass
    
    return jsonify({"status": "error", "message": "Could not process frame"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000, host='0.0.0.0')