import asyncio
import base64
import io
import traceback
import cv2
import PIL.Image
import os

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# The unused audio constants that were causing the error have been removed.

MODEL = "models/gemini-2.0-flash-live-001"
DEFAULT_MODE = "camera"

client = genai.Client(http_options={"api_version": "v1alpha"}, api_key=os.getenv("GEMINI_API_KEY"))
tools = [types.Tool(google_search=types.GoogleSearch())]

CONFIG = types.LiveConnectConfig(
    response_modalities=[types.Modality.AUDIO, types.Modality.TEXT],
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


class AudioLoop:
    def __init__(self, video_mode=DEFAULT_MODE, frame_queue=None):
        self.video_mode = video_mode
        self.frame_queue = frame_queue
        self.audio_in_queue = asyncio.Queue()

        self.session = None
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

    def _encode_frame(self, frame):
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
        while self.running:
            try:
                frame_np = await asyncio.to_thread(self.frame_queue.get, timeout=1.0)
                encoded_frame = self._encode_frame(frame_np)

                if self.session:
                    await self.session.send_realtime_input(
                        media=types.Blob(data=encoded_frame["data"], mime_type=encoded_frame["mime_type"])
                    )
            except Exception:
                await asyncio.sleep(0.1)

    async def receive_audio(self):
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

    async def run(self):
        try:
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session
                
                self.tasks.append(tg.create_task(self.send_text()))
                self.tasks.append(tg.create_task(self.receive_audio()))
                
                if self.video_mode == "camera" and self.frame_queue:
                    self.tasks.append(tg.create_task(self.send_realtime()))

        except asyncio.CancelledError:
            pass
        except Exception as e:
            traceback.print_exception(e)
        finally:
            self._cleanup()

    def stop(self):
        self.running = False
        for task in list(self.tasks):
            if not task.done():
                task.cancel()
        self.tasks.clear()
        print("✅ AudioLoop stop signal sent")

    def _cleanup(self):
        try:
            self.session = None
        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")

if __name__ == "__main__":
    main = AudioLoop(video_mode="none")
    asyncio.run(main.run())