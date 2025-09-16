from flask import Flask, request, jsonify, Response
import asyncio
import threading
import cv2
import mss
import numpy as np
import time
from live import AudioLoop

app = Flask(__name__)

# -----------------------
# Global state
# -----------------------
audio_loop_instance = None
loop_thread = None
status = "stopped"   # running | paused | stopped
cap = None
mode = "none"  # current mode

# -----------------------
# Rate limiting storage
# -----------------------
last_request_time = {}
# MODIFIED: Changed the request interval to 10 seconds
REQUEST_INTERVAL = 10.0  # 1 request per 10 seconds


def is_rate_limited(endpoint: str) -> bool:
    """Return True if the given endpoint is being called too quickly"""
    now = time.time()
    if endpoint not in last_request_time or now - last_request_time[endpoint] >= REQUEST_INTERVAL:
        last_request_time[endpoint] = now
        return False
    return True


# -----------------------
# Session Handling
# -----------------------
def run_event_loop(selected_mode="camera"):
    global audio_loop_instance, status, cap, mode
    audio_loop_instance = AudioLoop(video_mode=selected_mode)
    status = "running"
    mode = selected_mode

    # Open camera if mode = camera
    if mode == "camera":
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = None

    asyncio.run(audio_loop_instance.run())


@app.route("/")
def index():
    return "API is running"
@app.route("/status", methods=["GET"])
def get_status():
    return jsonify({"status": status, "mode": mode})


@app.route("/start", methods=["POST"])
def start_session():
    global loop_thread, audio_loop_instance, status, mode

    if is_rate_limited("start"):
        return jsonify({"status": "error", "message": "Too many requests, wait 1 minute"}), 429

    requested_mode = request.json.get("mode", "camera")

    # Prevent same-mode multiple sessions
    if status == "running" and mode == requested_mode:
        return jsonify({"status": "error", "message": f"Session already running in {mode} mode"}), 400

    # If another mode is running, block
    if status == "running" and mode != requested_mode:
        return jsonify({"status": "error", "message": f"Another session is running in {mode} mode"}), 400

    loop_thread = threading.Thread(target=run_event_loop, args=(requested_mode,), daemon=True)
    loop_thread.start()
    status = "running"
    mode = requested_mode
    return jsonify({"status": "success", "message": f"Session started in {mode} mode"})


@app.route("/pause", methods=["POST"])
def pause_session():
    global audio_loop_instance, status

    if is_rate_limited("pause"):
        return jsonify({"status": "error", "message": "Too many requests, wait 1 minute"}), 429

    if audio_loop_instance is None:
        return jsonify({"status": "error", "message": "No session running"}), 400

    audio_loop_instance.running = False
    status = "paused"
    return jsonify({"status": "success", "message": "Session paused"})


@app.route("/resume", methods=["POST"])
def resume_session():
    global loop_thread, audio_loop_instance, status, mode

    if is_rate_limited("resume"):
        return jsonify({"status": "error", "message": "Too many requests, wait 1 minute"}), 429

    if status != "paused":
        return jsonify({"status": "error", "message": "No paused session"}), 400

    loop_thread = threading.Thread(target=run_event_loop, args=(mode,), daemon=True)
    loop_thread.start()
    status = "running"
    return jsonify({"status": "success", "message": "Session resumed"})


@app.route("/stop", methods=["POST"])
def stop_session():
    global audio_loop_instance, loop_thread, status, cap, mode

    if is_rate_limited("stop"):
        return jsonify({"status": "error", "message": "Too many requests, wait 1 minute"}), 429

    if audio_loop_instance is None:
        return jsonify({"status": "error", "message": "No session running"}), 400

    try:
        audio_loop_instance.stop()
        audio_loop_instance = None
        if cap:
            cap.release()
            cap = None
        if loop_thread:
            loop_thread.join(timeout=2)
            loop_thread = None
        status = "stopped"
        mode = "none"
        return jsonify({"status": "success", "message": "Session stopped"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/frame", methods=["GET"])
def get_frame():
    """Return the latest frame as JPEG"""
    global cap, status, mode

    # ⚠️ No rate limiting here → allow smooth video

    if status != "running":
        return jsonify({"status": "error", "message": "No running session"}), 400

    frame = None
    if mode == "camera":
        if cap is None or not cap.isOpened():
            return jsonify({"status": "error", "message": "Camera not available"}), 500
        ret, frame = cap.read()
        if not ret:
            return jsonify({"status": "error", "message": "Failed to read camera"}), 500
    elif mode == "screen":
        sct = mss.mss()
        monitor = sct.monitors[0]
        img = sct.grab(monitor)
        frame = np.array(img)[:, :, :3]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if frame is None:
        return jsonify({"status": "error", "message": "No frame captured"}), 500

    _, jpeg = cv2.imencode(".jpg", frame)
    return Response(jpeg.tobytes(), mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(debug=True, port=5000,host='0.0.0.0')