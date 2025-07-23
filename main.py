from assistant.audio import record_audio, transcribe_audio
from assistant.pdf import extract_text_from_pdf
from assistant.quiz import generate_quiz
from assistant.speak import speak_streamed_text
from assistant.summary import generate_summary
from assistant.gemma import ask_gemma

class EchoLearnAssistant:
    def __init__(self, pdf_path=None, context=None):
        self.audio_filename = "input.wav"
        self.audio_duration = 5
        self.audio_fs = 44100
        self.context = context if context else extract_text_from_pdf(pdf_path)

    def run_once(self):
        record_audio(self.audio_filename, self.audio_duration, self.audio_fs)
        question = transcribe_audio(self.audio_filename)
        if question:
            answer = ask_gemma(question, self.context)
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
   #pdf_file = "CMSC320_ Notes.pdf"
    #assistant = EchoLearnAssistant(pdf_path=pdf_file)
    #assistant.run_loop()
    pdf_file = "CMSC320_ Notes.pdf"
    context = extract_text_from_pdf(pdf_file)

    summary_prompt = generate_summary(context)
    ask_gemma("Summarize this textbook.", context)