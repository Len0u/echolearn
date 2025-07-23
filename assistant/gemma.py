from ollama import Client
from assistant.speak import speak_streamed_text

client = Client()

def ask_gemma(question, context, model_name="gemma3n:e2b"):
    prompt = f"""You are a helpful study assistant. Use the following textbook context to answer the question concisely.

Context:
{context}

Question:
{question}

Answer:"""

    try:
        response_stream = client.generate(
            model=model_name,
            prompt=prompt,
            stream=True
        )

        buffer = ""
        spoken = ""

        for chunk in response_stream:
            token = chunk.get("response", "")
            print(token, end="", flush=True)
            buffer += token

            if any(p in buffer for p in [".", "!", "?"]):
                new_chunk = buffer.strip().replace(spoken, "")
                spoken += new_chunk
                speak_streamed_text(new_chunk)
                buffer = ""

        return spoken

    except Exception as e:
        print(f"❌ Error streaming response from Gemma: {e}")
        return ""
