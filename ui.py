import streamlit as st
import requests
from PIL import Image
import io
import time

API_URL = "http://localhost:5000"  # Flask backend URL


def call_api(endpoint, method="GET", data=None, stream=False):
    try:
        url = f"{API_URL}{endpoint}"
        if method == "POST":
            response = requests.post(url, json=data, stream=stream)
        else:
            response = requests.get(url, stream=stream)
        
        response.raise_for_status()  # Raise an exception for bad status codes
        
        if stream:
            return response
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": f"API connection error: {e}"}


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
    if res.get("status") == "success":
        st.sidebar.success("✅ Session started successfully")
    elif res.get("status") == "error" and "already running" in res.get("message", ""):
        st.sidebar.warning("⚠️ Session is already running")
    else:
        st.sidebar.error(f"❌ {res.get('message', 'An unknown error occurred.')}")
    st.rerun()


if st.sidebar.button("⏸️ Pause Session"):
    res = call_api("/pause", method="POST")
    if res.get("status") == "success":
        st.sidebar.info("⏸️ Session paused")
    else:
        st.sidebar.error(f"❌ {res.get('message', 'An unknown error occurred.')}")
    st.rerun()


if st.sidebar.button("▶️ Resume Session"):
    res = call_api("/resume", method="POST")
    if res.get("status") == "success":
        st.sidebar.success("▶️ Session resumed")
    else:
        st.sidebar.error(f"❌ {res.get('message', 'An unknown error occurred.')}")
    st.rerun()


if st.sidebar.button("🛑 Stop Session"):
    res = call_api("/stop", method="POST")
    if res.get("status") == "success":
        st.sidebar.success("🛑 Session stopped")
    else:
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
status_value = status_res.get("status", "error")
mode_value = status_res.get("mode", "none")

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

# Create a placeholder for the live video frame
image_placeholder = st.empty()

if status_value == "running" and mode_value in ["camera", "screen"]:
    while True:
        frame_response = call_api("/frame", method="GET", stream=True)
        if isinstance(frame_response, dict) and frame_response.get("status") == "error":
            st.error(f"Error fetching frame: {frame_response.get('message')}")
            break
        
        if frame_response and hasattr(frame_response, 'status_code') and frame_response.status_code == 200:
            try:
                img = Image.open(io.BytesIO(frame_response.content))
                # FIX: Replaced deprecated 'use_container_width' with 'width'
                image_placeholder.image(img, caption="Live Feed", width='stretch')
            except Exception as e:
                st.error(f"Failed to decode image: {e}")
                break
        else:
            st.warning("Waiting for video stream...")
        
        # Control the frame rate to reduce network load
        time.sleep(0.05)
elif status_value == "stopped":
    image_placeholder.info("Session is stopped. Start a new session to see the live feed.")
elif status_value == "paused":
    image_placeholder.info("Session is paused. Resume to see the live feed.")