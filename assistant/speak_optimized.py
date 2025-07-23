"""
Optimized speech synthesis with async support and better error handling.
"""
import asyncio
import logging
import tempfile
import os
from typing import Optional
from pathlib import Path
import edge_tts
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class SpeechSynthesizer:
    """Optimized speech synthesizer with async support and resource management."""
    
    def __init__(self, voice: str = "en-US-JennyNeural"):
        """Initialize speech synthesizer."""
        self.voice = voice
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._temp_files = []  # Track temp files for cleanup
        
        logger.info(f"SpeechSynthesizer initialized with voice: {voice}")
    
    async def speak_async(self, text: str, cleanup_delay: float = 2.0) -> bool:
        """
        Speak text asynchronously with better error handling.
        
        Args:
            text: Text to speak
            cleanup_delay: Delay before cleaning up temp files
            
        Returns:
            True if successful, False otherwise
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for speech")
            return False
        
        try:
            logger.info(f"🔊 Speaking text: {text[:50]}...")
            
            # Generate speech audio
            temp_audio_path = await self._generate_audio_async(text)
            if not temp_audio_path:
                return False
            
            # Play audio
            success = await self._play_audio_async(temp_audio_path)
            
            # Schedule cleanup
            if cleanup_delay > 0:
                asyncio.create_task(self._cleanup_temp_file(temp_audio_path, cleanup_delay))
            else:
                await self._cleanup_temp_file(temp_audio_path, 0)
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Error in speech synthesis: {e}")
            return False
    
    async def _generate_audio_async(self, text: str) -> Optional[str]:
        """Generate audio file from text."""
        try:
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            temp_audio_path = temp_file.name
            temp_file.close()
            
            self._temp_files.append(temp_audio_path)
            
            # Generate speech
            communicate = edge_tts.Communicate(text, voice=self.voice)
            
            with open(temp_audio_path, "wb") as audio_file:
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio_file.write(chunk["data"])
            
            logger.info(f"✅ Audio generated: {temp_audio_path}")
            return temp_audio_path
            
        except Exception as e:
            logger.error(f"Error generating audio: {e}")
            return None
    
    async def _play_audio_async(self, audio_path: str) -> bool:
        """Play audio file asynchronously."""
        try:
            if not os.path.exists(audio_path):
                logger.error(f"Audio file not found: {audio_path}")
                return False
            
            # Run audio playback in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._play_audio_sync,
                audio_path
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
            return False
    
    def _play_audio_sync(self, audio_path: str) -> bool:
        """Synchronous audio playback."""
        try:
            # Use afplay on macOS
            result = os.system(f"afplay '{audio_path}' 2>/dev/null")
            return result == 0
        except Exception as e:
            logger.error(f"Error in sync audio playback: {e}")
            return False
    
    async def _cleanup_temp_file(self, file_path: str, delay: float):
        """Clean up temporary file after delay."""
        try:
            if delay > 0:
                await asyncio.sleep(delay)
            
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"Cleaned up temp file: {file_path}")
            
            # Remove from tracking list
            if file_path in self._temp_files:
                self._temp_files.remove(file_path)
                
        except Exception as e:
            logger.warning(f"Error cleaning up temp file {file_path}: {e}")
    
    async def speak_streaming_async(self, text_stream: asyncio.Queue) -> None:
        """
        Speak text from a streaming queue (for real-time TTS during Gemma response).
        
        Args:
            text_stream: AsyncIO queue containing text chunks to speak
        """
        try:
            while True:
                try:
                    # Wait for text chunk with timeout
                    text_chunk = await asyncio.wait_for(text_stream.get(), timeout=1.0)
                    
                    if text_chunk is None:  # End signal
                        break
                    
                    if text_chunk.strip():
                        await self.speak_async(text_chunk, cleanup_delay=1.0)
                    
                    text_stream.task_done()
                    
                except asyncio.TimeoutError:
                    continue  # Continue waiting for more text
                    
        except Exception as e:
            logger.error(f"Error in streaming speech: {e}")
    
    def speak_sync(self, text: str) -> bool:
        """Synchronous wrapper for speech."""
        return asyncio.run(self.speak_async(text))
    
    async def cleanup(self):
        """Clean up all resources and temporary files."""
        try:
            # Clean up all temp files
            cleanup_tasks = [
                self._cleanup_temp_file(file_path, 0) 
                for file_path in self._temp_files.copy()
            ]
            
            if cleanup_tasks:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            
            # Shutdown executor
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=True)
            
            logger.info("SpeechSynthesizer cleaned up")
            
        except Exception as e:
            logger.error(f"Error during speech synthesizer cleanup: {e}")


# Backward compatibility function
def speak_streamed_text(text: str):
    """Legacy function for backward compatibility."""
    synthesizer = SpeechSynthesizer()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(synthesizer.speak_async(text))
    finally:
        loop.close()
