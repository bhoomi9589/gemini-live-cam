import streamlit as st
import requests
from PIL import Image
import io
import time
# --- NEW IMPORTS ---
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import numpy as np
import cv2

API_URL = "https://gemini-live-cam.onrender.com"  # Flask backend URL


def call_api(endpoint, method="GET", data=None, files=None):
    """Modified to handle file uploads."""
    try:
        url = f"{API_URL}{endpoint}"
        if method == "POST":
            response = requests.post(url, json=data, files=files)
        else:
            response = requests.get(url, stream=True if endpoint == "/frame" else False)
        
        response.raise_for_status()
        
        # Handle different response types
        if response.headers['Content-Type'] == 'application/json':
            return response.json()
        return response
    except requests.exceptions.RequestException as e:
        st.error(f"API connection error: {e}")
        return None

# This callback function will be executed for each frame from the webcam
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    """Encodes each frame and sends it to the backend."""
    img = frame.to_ndarray(format="bgr24")

    _, buffer = cv2.imencode(".jpg", img)
    jpeg_bytes = buffer.tobytes()

    try:
        # Send the frame to the backend's new endpoint
        call_api("/upload_frame", method="POST", files={"frame": jpeg_bytes})
    except Exception as e:
        # Silently pass if the backend is busy, to not interrupt the stream
        pass
    
    # Return the frame to display it in the Streamlit UI
    return frame

# -----------------------
# Sidebar Controls
# -----------------------
st.sidebar.title("🎛️ Controls")
mode = st.sidebar.radio(
    "Select Mode",
    options=["camera", "screen", "none"],
    index=0,
    help="Choose session mode: camera, screen capture, or audio-only"
)

if st.sidebar.button("▶️ Start Session"):
    res = call_api("/start", method="POST", data={"mode": mode})
    if res and res.get("status") == "success":
        st.sidebar.success("✅ Session started successfully")
    elif res and "already running" in res.get("message", ""):
        st.sidebar.warning("⚠️ Session is already running")
    elif res:
        st.sidebar.error(f"❌ {res.get('message', 'An unknown error occurred.')}")
    st.rerun()


if st.sidebar.button("⏸️ Pause Session"):
    res = call_api("/pause", method="POST")
    if res and res.get("status") == "success":
        st.sidebar.info("⏸️ Session paused")
    elif res:
        st.sidebar.error(f"❌ {res.get('message', 'An unknown error occurred.')}")
    st.rerun()


if st.sidebar.button("▶️ Resume Session"):
    res = call_api("/resume", method="POST")
    if res and res.get("status") == "success":
        st.sidebar.success("▶️ Session resumed")
    elif res:
        st.sidebar.error(f"❌ {res.get('message', 'An unknown error occurred.')}")
    st.rerun()


if st.sidebar.button("🛑 Stop Session"):
    res = call_api("/stop", method="POST")
    if res and res.get("status") == "success":
        st.sidebar.success("🛑 Session stopped")
    elif res:
        st.sidebar.error(f"❌ {res.get('message', 'An unknown error occurred.')}")
    st.rerun()

# -----------------------
# Custom Styling
# -----------------------
st.markdown("""
    <style>
        body { background-color: #f8fafc; }
        .status-banner { padding: 12px; border-radius: 12px; font-weight: bold; text-align: center; margin-bottom: 20px; }
        .status-running { background-color: #d1fae5; color: #065f46; }
        .status-paused { background-color: #fef3c7; color: #92400e; }
        .status-stopped { background-color: #fee2e2; color: #991b1b; }
        .stImage > img { border-radius: 16px; box-shadow: 0 6px 18px rgba(0,0,0,0.2); }
    </style>
""", unsafe_allow_html=True)


# -----------------------
# Main UI
# -----------------------
st.title("🎥 Gemini Live UI")

status_res = call_api("/status")
if status_res:
    status_value = status_res.get("status", "error")
    mode_value = status_res.get("mode", "none")
else:
    status_value = "error"
    mode_value = "N/A"

status_class_map = {
    "running": "status-running",
    "paused": "status-paused",
    "stopped": "status-stopped",
}
status_class = status_class_map.get(status_value, "status-stopped")

st.markdown(
    f"<div class='status-banner {status_class}'>"
    f"Status: {status_value.upper()} | Mode: {mode_value.upper()}"
    f"</div>",
    unsafe_allow_html=True
)

if status_value == "running" and mode_value == "camera":
    st.info("Webcam feed is active. Your camera data is being sent for processing.")
    webrtc_streamer(
        key="camera-stream",
        mode=WebRtcMode.SENDONLY,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
elif status_value == "stopped":
    st.info("Session is stopped. Start a new session to see the live feed.")
elif status_value == "paused":
    st.info("Session is paused. Resume to see the live feed.")
elif status_value == "error":
    st.error("Could not connect to the backend. Please ensure the Flask server is running.")