import os
import whisper
import sounddevice as sd
from scipy.io.wavfile import write
import fitz  # PyMuPDF
import asyncio
import edge_tts
import threading
import tempfile
from ollama import Client

# 🔁 Reuse the same Ollama client across all calls
ollama_client = Client()

class EchoLearnAssistant:
    def __init__(self, pdf_path=None, context=None, model_name="gemma3n:e2b", audio_filename="input.wav", audio_duration=5, audio_fs=44100):
        self.client = ollama_client
        self.model_name = model_name
        self.audio_filename = audio_filename
        self.audio_duration = audio_duration
        self.audio_fs = audio_fs
        self.context = context if context else self.extract_text_from_pdf(pdf_path)
        self.whisper_model = whisper.load_model("tiny")

    def extract_text_from_pdf(self, pdf_path):
        try:
            doc = fitz.open(pdf_path)
            text = "\n".join(page.get_text() for page in doc)
            print(f"✅ Extracted text from PDF ({pdf_path})")
            return text
        except Exception as e:
            print(f"❌ Error extracting PDF text: {e}")
            return ""

    def record_audio(self):
        print(f"🎤 Speak your question (max {self.audio_duration} seconds)...")
        audio = sd.rec(int(self.audio_duration * self.audio_fs), samplerate=self.audio_fs, channels=1)
        sd.wait()
        write(self.audio_filename, self.audio_fs, audio)
        print(f"✅ Audio saved to {self.audio_filename}")

    def transcribe_audio(self):
        try:
            result = self.whisper_model.transcribe(self.audio_filename)
            transcription = result.get("text", "").strip()
            print(f"📝 Transcription: {transcription}")
            return transcription
        except Exception as e:
            print(f"❌ Error transcribing audio: {e}")
            return ""

    def speak_streamed_text(self, text):
        def _speak():
            asyncio.run(self._speak_with_edge(text))
        threading.Thread(target=_speak, daemon=True).start()

    async def _speak_with_edge(self, text):
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

    def generate_summary(self):
        prompt = f"""You are a helpful assistant. Summarize the following notes or textbook into key takeaways, organized into topics with bullet points. Then convert this into a concise narrated explanation as if you're teaching the content.

        Content:
        {self.context}

        Output format:
        1. Summary (bullet points)
        2. Learning podcast script (spoken explanation)
        """
    def generate_quiz(self):
        prompt = f"""Generate 5 short-answer quiz questions with correct answers based on the following study material:

        {self.context}

        Output as JSON:
        [
        {{"question": "...", "answer": "..."}},
        ...
        ]
        """



    def ask_gemma(self, question):
        prompt = f"""You are a helpful study assistant. Use the following textbook context to answer the question concisely.

Context:
{self.context}

Question:
{question}

Answer:"""

        try:
            response_stream = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                stream=True
            )

            buffer = ""
            spoken = ""

            for chunk in response_stream:
                token = chunk.get("response", "")
                print(token, end="", flush=True)
                buffer += token

                # Speak earlier for more natural flow
                if len(buffer) > 30 or any(p in buffer for p in [".", "!", "?"]):
                    new_chunk = buffer.strip().replace(spoken, "")
                    spoken += new_chunk
                    self.speak_streamed_text(new_chunk)
                    buffer = ""

            return spoken

        except Exception as e:
            print(f"❌ Error streaming response from Gemma: {e}")
            return ""

    def run_once(self):
        self.record_audio()
        question = self.transcribe_audio()
        if question:
            answer = self.ask_gemma(question)
            if answer:
                print(f"\n🤖 Done speaking!")
            else:
                print("⚠️ No answer received from Gemma.")
        else:
            print("⚠️ No question transcribed.")

    def run_loop(self):
        while True:
            self.run_once()
            again = input("\n🔁 Ask another? (y/n): ").strip().lower()
            if again != "y":
                print("👋 Goodbye!")
                break

if __name__ == "__main__":
    # 🧠 Either load from a real PDF or from a cached file
    pdf_file = "CMSC320_ Notes.pdf"
    # with open("cached_context.txt") as f:
    #     cached_context = f.read()
    # assistant = EchoLearnAssistant(context=cached_context)
    
    assistant = EchoLearnAssistant(pdf_path=pdf_file)
    assistant.run_loop()
