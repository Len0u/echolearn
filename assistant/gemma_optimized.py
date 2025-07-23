"""
Optimized Gemma client with async support, timeouts, and connection pooling.
"""
import asyncio
import logging
from typing import Optional, AsyncGenerator
from ollama import AsyncClient
import time

logger = logging.getLogger(__name__)


class GemmaClient:
    """Optimized Gemma client with async support and robust error handling."""
    
    def __init__(self, model_name: str = "gemma3n:e2b", host: str = "http://localhost:11434"):
        """Initialize Gemma client with connection pooling."""
        self.model_name = model_name
        self.client = AsyncClient(host=host)
        self.max_retries = 3
        self.retry_delay = 1.0
        
        logger.info(f"GemmaClient initialized with model: {model_name}")
    
    async def ask_question_async(
        self, 
        question: str, 
        context: str, 
        timeout: int = 30,
        stream_callback: Optional[callable] = None
    ) -> str:
        """
        Ask a question to Gemma with timeout and retry logic.
        
        Args:
            question: The question to ask
            context: Context from PDF or other sources
            timeout: Timeout in seconds
            stream_callback: Optional callback for streaming responses
            
        Returns:
            The answer from Gemma or empty string if failed
        """
        prompt = self._build_prompt(question, context)
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"🤖 Asking Gemma (attempt {attempt + 1}/{self.max_retries})")
                
                # Use asyncio.wait_for for timeout
                response = await asyncio.wait_for(
                    self._generate_response(prompt, stream_callback),
                    timeout=timeout
                )
                
                if response:
                    logger.info("✅ Received response from Gemma")
                    return response
                else:
                    logger.warning("Empty response from Gemma")
                    
            except asyncio.TimeoutError:
                logger.error(f"❌ Gemma request timed out after {timeout}s (attempt {attempt + 1})")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                    continue
                else:
                    return "I'm sorry, the request timed out. Please try again."
                    
            except Exception as e:
                logger.error(f"❌ Error with Gemma (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                    continue
                else:
                    return f"I'm sorry, I encountered an error: {str(e)}"
        
        return ""
    
    async def _generate_response(self, prompt: str, stream_callback: Optional[callable] = None) -> str:
        """Generate response from Gemma with streaming support."""
        try:
            response_stream = await self.client.generate(
                model=self.model_name,
                prompt=prompt,
                stream=True
            )
            
            full_response = ""
            buffer = ""
            
            async for chunk in response_stream:
                token = chunk.get("response", "")
                if not token:
                    continue
                
                print(token, end="", flush=True)
                full_response += token
                buffer += token
                
                # Call streaming callback if provided (for TTS)
                if stream_callback and any(p in buffer for p in [".", "!", "?"]):
                    sentence = buffer.strip()
                    if sentence:
                        await stream_callback(sentence)
                    buffer = ""
            
            # Handle any remaining text in buffer
            if stream_callback and buffer.strip():
                await stream_callback(buffer.strip())
            
            return full_response.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def _build_prompt(self, question: str, context: str) -> str:
        """Build optimized prompt for Gemma."""
        # Truncate context if too long to avoid token limits
        max_context_length = 4000
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."
            logger.info(f"Context truncated to {max_context_length} characters")
        
        return f"""You are a helpful study assistant. Use the following textbook context to answer the question concisely and accurately.

Context:
{context}

Question:
{question}

Answer (be concise and direct):"""
    
    async def ask_question_with_streaming_tts(
        self, 
        question: str, 
        context: str, 
        tts_callback: callable,
        timeout: int = 30
    ) -> str:
        """Ask question with integrated streaming TTS."""
        return await self.ask_question_async(
            question=question,
            context=context,
            timeout=timeout,
            stream_callback=tts_callback
        )
    
    async def health_check(self) -> bool:
        """Check if Gemma service is available."""
        try:
            response = await asyncio.wait_for(
                self.client.generate(
                    model=self.model_name,
                    prompt="Hello",
                    stream=False
                ),
                timeout=10
            )
            return bool(response.get("response"))
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def cleanup(self):
        """Clean up client resources."""
        try:
            if hasattr(self.client, 'close'):
                await self.client.close()
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
        logger.info("GemmaClient cleaned up")


# Backward compatibility function
def ask_gemma(question: str, context: str, model_name: str = "gemma3n:e2b") -> str:
    """Legacy function for backward compatibility."""
    client = GemmaClient(model_name=model_name)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(client.ask_question_async(question, context))
    finally:
        loop.close()
