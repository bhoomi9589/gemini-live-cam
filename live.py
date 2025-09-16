import asyncio
import base64
import io
import traceback
import cv2
import pyaudio
import PIL.Image
import mss
import argparse
import os

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# Audio constants
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
    def __init__(self, video_mode=DEFAULT_MODE):
        self.video_mode = video_mode
        self.audio_in_queue = asyncio.Queue()
        self.out_queue = asyncio.Queue(maxsize=5)

        self.session = None
        self.audio_stream = None
        self.playback_stream = None

        self.running = True
        self.tasks: list[asyncio.Task] = []

        self.received_texts = []
        self.received_audio = []

    async def send_text(self):
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

    def _get_frame(self, cap):
        ret, frame = cap.read()
        if not ret:
            return None
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(frame_rgb)
        img.thumbnail((1024, 1024))

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        return {
            "mime_type": "image/jpeg",
            "data": base64.b64encode(image_io.read()).decode(),
        }

    async def get_frames(self):
        cap = await asyncio.to_thread(cv2.VideoCapture, 0)
        while self.running:
            frame = await asyncio.to_thread(self._get_frame, cap)
            if frame is None:
                break
            await asyncio.sleep(1.0)
            await self.out_queue.put(frame)
        cap.release()

    def _get_screen(self):
        sct = mss.mss()
        monitor = sct.monitors[0]
        img = sct.grab(monitor)
        pil_img = PIL.Image.frombytes("RGB", img.size, img.rgb)

        image_io = io.BytesIO()
        pil_img.save(image_io, format="jpeg")
        image_io.seek(0)

        return {
            "mime_type": "image/jpeg",
            "data": base64.b64encode(image_io.read()).decode(),
        }

    async def get_screen(self):
        while self.running:
            frame = await asyncio.to_thread(self._get_screen)
            if frame is None:
                break
            await asyncio.sleep(1.0)
            await self.out_queue.put(frame)

    async def send_realtime(self):
        while self.running:
            msg = await self.out_queue.get()
            if self.session:
                await self.session.send_realtime_input(
                    media=types.Blob(data=msg["data"], mime_type=msg["mime_type"])
                )

    async def listen_audio(self):
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
            await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

    async def receive_audio(self):
        while self.running:
            if not self.session:
                await asyncio.sleep(1)
                continue
            turn = self.session.receive()
            async for response in turn:
                # Handle audio data if present
                if data := response.data:
                    self.audio_in_queue.put_nowait(data)
                    self.received_audio.append(data)

                # FIX: Safely check if the 'parts' attribute exists before trying to loop
                if hasattr(response, 'parts'):
                    for part in response.parts:
                        if part.text:
                            print(part.text, end="")
                            self.received_texts.append(part.text)
                        elif part.tool_call:
                            # This will print if the model uses a tool like Google Search
                            print(f"\n[Tool Call: {part.tool_call.name}]")

    async def play_audio(self):
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
                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)

                self.tasks.append(tg.create_task(self.send_text()))
                self.tasks.append(tg.create_task(self.send_realtime()))
                self.tasks.append(tg.create_task(self.listen_audio()))
                if self.video_mode == "camera":
                    self.tasks.append(tg.create_task(self.get_frames()))
                elif self.video_mode == "screen":
                    self.tasks.append(tg.create_task(self.get_screen()))
                self.tasks.append(tg.create_task(self.receive_audio()))
                self.tasks.append(tg.create_task(self.play_audio()))

        except asyncio.CancelledError:
            pass
        except Exception as e:
            traceback.print_exception(e)
        finally:
            self._cleanup()

    def stop(self):
        """Gracefully stop audio/video/session without killing Flask"""
        self.running = False
        for task in list(self.tasks):
            if not task.done():
                task.cancel()
        self.tasks.clear()
        # self._cleanup() was REMOVED from here to prevent a race condition.
        # The finally block in the run() method now handles all cleanup safely.
        print("✅ AudioLoop stop signal sent")

    def _cleanup(self):
        """Release resources safely (sync now)"""
        try:
            # ✅ FIXED: Removed redundant and incorrect self.session.close() call
            # The 'async with' block in run() handles this correctly.
            self.session = None

            if self.audio_stream:
                try:
                    self.audio_stream.close()
                except Exception:
                    pass
                self.audio_stream = None

            if self.playback_stream:
                try:
                    self.playback_stream.close()
                except Exception:
                    pass
                self.playback_stream = None
        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default=DEFAULT_MODE, choices=["camera", "screen", "none"])
    args = parser.parse_args()
    main = AudioLoop(video_mode=args.mode)
    asyncio.run(main.run())