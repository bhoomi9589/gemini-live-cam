import asyncio
import base64
import io
import traceback
import cv2
import pyaudio
import PIL.Image
import os

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# Audio constants (no changes)
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

MODEL = "models/gemini-2.0-flash-live-001"
DEFAULT_MODE = "camera"

client = genai.Client(http_options={"api_version": "v1alpha"}, api_key=os.getenv("GEMINI_API_KEY"))
tools = [types.Tool(google_search=types.GoogleSearch())]

CONFIG = types.LiveConnectConfig(
    response_modalities=[types.Modality.AUDIO],
    generation_config=types.GenerationConfig(
        max_output_tokens=300,
        temperature=0.7
    ),
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Leda")
        )
    ),
    tools=types.ToolListUnion(tools),
)

pya = pyaudio.PyAudio()


class AudioLoop:
    def __init__(self, video_mode=DEFAULT_MODE, frame_queue=None):
        self.video_mode = video_mode
        self.frame_queue = frame_queue  # The shared queue for receiving frames
        self.audio_in_queue = asyncio.Queue()
        self.audio_out_queue = asyncio.Queue(maxsize=5) # Renamed for clarity

        self.session = None
        self.audio_stream = None
        self.playback_stream = None

        self.running = True
        self.tasks: list[asyncio.Task] = []

        self.received_texts = []
        self.received_audio = []

    async def send_text(self):
        # ... (no changes to this method)
        while self.running:
            text = await asyncio.to_thread(input, "message > ")
            if text.lower() == "q":
                break
            if self.session:
                await self.session.send_client_content(
                    turns=types.Content(
                        role="user",
                        parts=[types.Part(text=text or ".")]
                    )
                )

    def _encode_frame(self, frame):
        """Encodes a single NumPy frame to a base64 JPEG."""
        img = PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img.thumbnail((1024, 1024))
        
        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        return {
            "mime_type": "image/jpeg",
            "data": base64.b64encode(image_io.read()).decode(),
        }

    async def send_realtime(self):
        """Reads from the shared frame_queue and sends to Gemini."""
        while self.running:
            try:
                # Read from the synchronous queue in a non-blocking way
                frame_np = await asyncio.to_thread(self.frame_queue.get, timeout=1.0)
                
                # Encode the frame
                encoded_frame = self._encode_frame(frame_np)

                if self.session:
                    await self.session.send_realtime_input(
                        media=types.Blob(data=encoded_frame["data"], mime_type=encoded_frame["mime_type"])
                    )
            except Exception:
                # Timeout or other error, just continue the loop
                await asyncio.sleep(0.1)


    async def listen_audio(self):
        # ... (no changes to this method)
        mic_info = pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=int(mic_info["index"]),
            frames_per_buffer=CHUNK_SIZE,
        )
        kwargs = {"exception_on_overflow": False}
        while self.running:
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
            # Put audio data into its own queue
            await self.audio_out_queue.put({"data": data, "mime_type": "audio/pcm"})

    # This task is now also responsible for sending audio
    async def send_audio(self):
        while self.running:
            msg = await self.audio_out_queue.get()
            if self.session:
                await self.session.send_realtime_input(
                    media=types.Blob(data=msg["data"], mime_type=msg["mime_type"])
                )

    async def receive_audio(self):
        # ... (no changes to this method)
        while self.running:
            if not self.session:
                await asyncio.sleep(1)
                continue
            turn = self.session.receive()
            async for response in turn:
                if data := response.data:
                    self.audio_in_queue.put_nowait(data)
                    self.received_audio.append(data)

                if hasattr(response, 'parts'):
                    for part in response.parts:
                        if part.text:
                            print(part.text, end="")
                            self.received_texts.append(part.text)
                        elif part.tool_call:
                            print(f"\n[Tool Call: {part.tool_call.name}]")

    async def play_audio(self):
        # ... (no changes to this method)
        self.playback_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        while self.running:
            bytestream = await self.audio_in_queue.get()
            await asyncio.to_thread(self.playback_stream.write, bytestream)

    async def run(self):
        try:
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session
                
                self.tasks.append(tg.create_task(self.send_text()))
                self.tasks.append(tg.create_task(self.listen_audio()))
                self.tasks.append(tg.create_task(self.send_audio()))
                
                # Only run the video task if in camera mode and a queue is provided
                if self.video_mode == "camera" and self.frame_queue:
                    self.tasks.append(tg.create_task(self.send_realtime()))
                
                self.tasks.append(tg.create_task(self.receive_audio()))
                self.tasks.append(tg.create_task(self.play_audio()))

        except asyncio.CancelledError:
            pass
        except Exception as e:
            traceback.print_exception(e)
        finally:
            self._cleanup()

    def stop(self):
        # ... (no changes to this method)
        self.running = False
        for task in list(self.tasks):
            if not task.done():
                task.cancel()
        self.tasks.clear()
        print("✅ AudioLoop stop signal sent")

    def _cleanup(self):
        # ... (no changes to this method)
        try:
            self.session = None
            if self.audio_stream:
                self.audio_stream.close()
                self.audio_stream = None
            if self.playback_stream:
                self.playback_stream.close()
                self.playback_stream = None
        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")

if __name__ == "__main__":
    # This part is now mainly for testing the AudioLoop directly
    # The main execution is handled by app.py
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default=DEFAULT_MODE, choices=["camera", "screen", "none"])
    args = parser.parse_args()
    main = AudioLoop(video_mode=args.mode)
    asyncio.run(main.run())