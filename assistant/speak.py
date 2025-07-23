import asyncio
import edge_tts
import threading
import tempfile
import os

def speak_streamed_text(text):
    def _speak():
        asyncio.run(_speak_with_edge(text))
    threading.Thread(target=_speak, daemon=True).start()

async def _speak_with_edge(text):
    try:
        communicate = edge_tts.Communicate(text, voice="en-US-JennyNeural")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    temp_audio.write(chunk["data"])
            temp_audio_path = temp_audio.name

        os.system(f"afplay '{temp_audio_path}'")  # macOS
        os.remove(temp_audio_path)
    except Exception as e:
        print(f"❌ Error in Edge TTS: {e}")

