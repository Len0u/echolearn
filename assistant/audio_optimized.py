"""
Optimized audio processing with async support and better error handling.
"""
import asyncio
import logging
import tempfile
import os
from typing import Optional
from pathlib import Path
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Optimized audio processor with async support and resource management."""
    
    def __init__(self, model_size: str = "tiny"):
        """Initialize with lazy loading of Whisper model."""
        self.model_size = model_size
        self._model = None
        self._model_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Audio settings
        self.sample_rate = 44100
        self.channels = 1
        
        logger.info(f"AudioProcessor initialized with model size: {model_size}")
    
    @property
    def model(self):
        """Lazy load Whisper model to save memory."""
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    logger.info(f"Loading Whisper model: {self.model_size}")
                    self._model = whisper.load_model(self.model_size)
                    logger.info("Whisper model loaded successfully")
        return self._model
    
    async def record_audio_async(self, duration: int = 5, filename: Optional[str] = None) -> Optional[str]:
        """
        Record audio asynchronously with better error handling.
        
        Args:
            duration: Recording duration in seconds
            filename: Optional filename, creates temp file if None
            
        Returns:
            Path to recorded audio file or None if failed
        """
        if filename is None:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            filename = temp_file.name
            temp_file.close()
        
        try:
            logger.info(f"🎤 Recording audio for {duration} seconds...")
            
            # Run recording in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            audio_data = await loop.run_in_executor(
                self.executor,
                self._record_audio_sync,
                duration
            )
            
            if audio_data is None:
                return None
            
            # Save audio file
            write(filename, self.sample_rate, audio_data)
            logger.info(f"✅ Audio saved to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"❌ Error recording audio: {e}")
            if os.path.exists(filename):
                os.remove(filename)
            return None
    
    def _record_audio_sync(self, duration: int):
        """Synchronous audio recording."""
        try:
            print(f"🎤 Speak your question (max {duration} seconds)...")
            audio = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='float64'
            )
            sd.wait()  # Wait for recording to complete
            return audio
        except Exception as e:
            logger.error(f"Error in sync recording: {e}")
            return None
    
    async def transcribe_audio_async(self, audio_file: str) -> str:
        """
        Transcribe audio file asynchronously with robust error handling.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Transcribed text or empty string if failed
        """
        try:
            if not os.path.exists(audio_file):
                logger.error(f"Audio file not found: {audio_file}")
                return ""
            
            logger.info("📝 Transcribing audio...")
            
            # Run transcription in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._transcribe_sync,
                audio_file
            )
            
            if result:
                transcription = result.get("text", "").strip()
                logger.info(f"📝 Transcription: {transcription}")
                return transcription
            else:
                logger.warning("No transcription result")
                return ""
                
        except Exception as e:
            logger.error(f"❌ Error transcribing audio: {e}")
            return ""
        finally:
            # Clean up temp audio file
            try:
                if os.path.exists(audio_file) and audio_file.startswith(tempfile.gettempdir()):
                    os.remove(audio_file)
            except Exception as e:
                logger.warning(f"Could not clean up audio file {audio_file}: {e}")
    
    def _transcribe_sync(self, audio_file: str):
        """Synchronous transcription."""
        try:
            return self.model.transcribe(audio_file)
        except Exception as e:
            logger.error(f"Error in sync transcription: {e}")
            return None
    
    def record_and_transcribe_sync(self, duration: int = 5) -> str:
        """Synchronous wrapper for record + transcribe."""
        return asyncio.run(self._record_and_transcribe_async(duration))
    
    async def _record_and_transcribe_async(self, duration: int) -> str:
        """Combined record and transcribe operation."""
        audio_file = await self.record_audio_async(duration)
        if not audio_file:
            return ""
        return await self.transcribe_audio_async(audio_file)
    
    async def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        logger.info("AudioProcessor cleaned up")


# Backward compatibility functions
def record_audio(filename: str, duration: int, fs: int):
    """Legacy function for backward compatibility."""
    processor = AudioProcessor()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(processor.record_audio_async(duration, filename))
        return result is not None
    finally:
        loop.close()


def transcribe_audio(filename: str) -> str:
    """Legacy function for backward compatibility."""
    processor = AudioProcessor()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(processor.transcribe_audio_async(filename))
    finally:
        loop.close()
