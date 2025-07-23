import sounddevice as sd
from scipy.io.wavfile import write
import whisper

model = whisper.load_model("tiny")

def record_audio(filename, duration, fs):
    print(f"🎤 Speak your question (max {duration} seconds)...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, audio)
    print(f"✅ Audio saved to {filename}")

def transcribe_audio(filename):
    try:
        result = model.transcribe(filename)
        transcription = result.get("text", "").strip()
        print(f"📝 Transcription: {transcription}")
        return transcription
    except Exception as e:
        print(f"❌ Error transcribing audio: {e}")
        return ""