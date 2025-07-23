"""
Optimized EchoLearn Assistant with improved performance, error handling, and modularity.
"""
import asyncio
import logging
import time
from typing import Optional, Dict, Any
from pathlib import Path

from assistant.audio_optimized import AudioProcessor
from assistant.pdf_optimized import PDFProcessor
from assistant.gemma_optimized import GemmaClient
from assistant.speak_optimized import SpeechSynthesizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EchoLearnAssistant:
    """Optimized EchoLearn Assistant with async processing and robust error handling."""
    
    def __init__(self, pdf_path: Optional[str] = None, context: Optional[str] = None):
        """Initialize the assistant with optimized components."""
        self.audio_processor = AudioProcessor()
        self.pdf_processor = PDFProcessor()
        self.gemma_client = GemmaClient()
        self.speech_synthesizer = SpeechSynthesizer()
        
        # Load context
        self.context = context
        if not self.context and pdf_path:
            self.context = self.pdf_processor.extract_text(pdf_path)
        
        logger.info("EchoLearn Assistant initialized successfully")
    
    async def process_question_async(self, audio_duration: int = 5) -> Dict[str, Any]:
        """
        Process a single question asynchronously with optimized performance.
        
        Returns:
            Dict containing success status, question, answer, and timing info
        """
        start_time = time.time()
        result = {
            'success': False,
            'question': '',
            'answer': '',
            'timing': {},
            'errors': []
        }
        
        try:
            # Step 1: Record audio
            logger.info("🎤 Starting audio recording...")
            record_start = time.time()
            audio_file = await self.audio_processor.record_audio_async(duration=audio_duration)
            result['timing']['recording'] = time.time() - record_start
            
            if not audio_file:
                result['errors'].append("Failed to record audio")
                return result
            
            # Step 2: Transcribe audio (can be done while preparing other components)
            logger.info("📝 Transcribing audio...")
            transcribe_start = time.time()
            question = await self.audio_processor.transcribe_audio_async(audio_file)
            result['timing']['transcription'] = time.time() - transcribe_start
            result['question'] = question
            
            if not question or len(question.strip()) < 3:
                result['errors'].append("No valid question transcribed")
                return result
            
            # Step 3: Get answer from Gemma with timeout
            logger.info(f"🤖 Processing question: {question[:50]}...")
            gemma_start = time.time()
            answer = await self.gemma_client.ask_question_async(
                question=question,
                context=self.context,
                timeout=30  # 30 second timeout
            )
            result['timing']['gemma'] = time.time() - gemma_start
            result['answer'] = answer
            
            if not answer:
                result['errors'].append("No answer received from Gemma")
                return result
            
            # Step 4: Speak answer (non-blocking)
            logger.info("🔊 Starting speech synthesis...")
            speech_start = time.time()
            await self.speech_synthesizer.speak_async(answer)
            result['timing']['speech'] = time.time() - speech_start
            
            result['success'] = True
            result['timing']['total'] = time.time() - start_time
            
            logger.info(f"✅ Question processed successfully in {result['timing']['total']:.2f}s")
            
        except asyncio.TimeoutError:
            error_msg = "Request timed out"
            result['errors'].append(error_msg)
            logger.error(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            result['errors'].append(error_msg)
            logger.error(error_msg, exc_info=True)
        
        return result
    
    def process_question_sync(self, audio_duration: int = 5) -> Dict[str, Any]:
        """Synchronous wrapper for async processing."""
        return asyncio.run(self.process_question_async(audio_duration))
    
    async def run_interactive_loop(self):
        """Run the interactive question-answer loop with optimized performance."""
        logger.info("🚀 Starting EchoLearn interactive session")
        
        while True:
            try:
                print("\n" + "="*50)
                result = await self.process_question_async()
                
                # Display results
                if result['success']:
                    print(f"\n✅ Question: {result['question']}")
                    print(f"🤖 Answer delivered successfully!")
                    print(f"⏱️  Total time: {result['timing']['total']:.2f}s")
                    
                    # Show timing breakdown
                    timing = result['timing']
                    print(f"   📊 Breakdown: Record({timing.get('recording', 0):.1f}s) + "
                          f"Transcribe({timing.get('transcription', 0):.1f}s) + "
                          f"Gemma({timing.get('gemma', 0):.1f}s) + "
                          f"Speech({timing.get('speech', 0):.1f}s)")
                else:
                    print(f"\n❌ Failed to process question")
                    for error in result['errors']:
                        print(f"   • {error}")
                
                # Ask to continue
                print("\n🔁 Ask another question?")
                continue_choice = input("   Enter 'y' to continue, any other key to exit: ").strip().lower()
                
                if continue_choice != 'y':
                    print("\n👋 Goodbye!")
                    break
                    
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error in interactive loop: {e}", exc_info=True)
                print(f"\n❌ Session error: {e}")
                break
    
    def run_interactive_sync(self):
        """Synchronous wrapper for interactive loop."""
        asyncio.run(self.run_interactive_loop())
    
    async def cleanup(self):
        """Clean up resources."""
        await self.audio_processor.cleanup()
        await self.gemma_client.cleanup()
        await self.speech_synthesizer.cleanup()
        logger.info("Resources cleaned up")


async def main():
    """Main entry point with error handling."""
    pdf_file = "CMSC320_ Notes.pdf"
    
    try:
        # Check if PDF exists
        if not Path(pdf_file).exists():
            logger.warning(f"PDF file {pdf_file} not found. Running without context.")
            assistant = EchoLearnAssistant()
        else:
            assistant = EchoLearnAssistant(pdf_path=pdf_file)
        
        # Run interactive session
        await assistant.run_interactive_loop()
        
    except Exception as e:
        logger.error(f"Failed to start assistant: {e}", exc_info=True)
    finally:
        if 'assistant' in locals():
            await assistant.cleanup()


if __name__ == "__main__":
    # For testing individual components
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Quick test mode
        async def quick_test():
            assistant = EchoLearnAssistant(pdf_path="CMSC320_ Notes.pdf")
            result = await assistant.process_question_async(audio_duration=3)
            print(f"Test result: {result}")
            await assistant.cleanup()
        
        asyncio.run(quick_test())
    else:
        # Normal interactive mode
        asyncio.run(main())
