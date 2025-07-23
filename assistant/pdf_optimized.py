"""
Optimized PDF processing with async support and better error handling.
"""
import asyncio
import logging
from typing import Optional
from pathlib import Path
import fitz  # PyMuPDF
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Optimized PDF processor with async support and caching."""
    
    def __init__(self):
        """Initialize PDF processor with thread pool."""
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._cache = {}  # Simple in-memory cache
        logger.info("PDFProcessor initialized")
    
    async def extract_text_async(self, pdf_path: str) -> str:
        """
        Extract text from PDF asynchronously with caching.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text or empty string if failed
        """
        pdf_path = str(Path(pdf_path).resolve())
        
        # Check cache first
        if pdf_path in self._cache:
            logger.info(f"📄 Using cached text for {pdf_path}")
            return self._cache[pdf_path]
        
        try:
            if not Path(pdf_path).exists():
                logger.error(f"PDF file not found: {pdf_path}")
                return ""
            
            logger.info(f"📄 Extracting text from PDF: {pdf_path}")
            
            # Run extraction in thread pool
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(
                self.executor,
                self._extract_text_sync,
                pdf_path
            )
            
            if text:
                # Cache the result
                self._cache[pdf_path] = text
                logger.info(f"✅ Extracted {len(text)} characters from PDF")
                return text
            else:
                logger.warning("No text extracted from PDF")
                return ""
                
        except Exception as e:
            logger.error(f"❌ Error extracting PDF text: {e}")
            return ""
    
    def _extract_text_sync(self, pdf_path: str) -> str:
        """Synchronous PDF text extraction."""
        try:
            doc = fitz.open(pdf_path)
            text_parts = []
            
            for page_num, page in enumerate(doc):
                try:
                    page_text = page.get_text()
                    if page_text.strip():
                        text_parts.append(page_text)
                except Exception as e:
                    logger.warning(f"Error extracting text from page {page_num}: {e}")
                    continue
            
            doc.close()
            return "\n".join(text_parts)
            
        except Exception as e:
            logger.error(f"Error in sync PDF extraction: {e}")
            return ""
    
    def extract_text(self, pdf_path: str) -> str:
        """Synchronous wrapper for text extraction."""
        return asyncio.run(self.extract_text_async(pdf_path))
    
    async def get_pdf_info_async(self, pdf_path: str) -> dict:
        """Get PDF metadata asynchronously."""
        try:
            if not Path(pdf_path).exists():
                return {}
            
            loop = asyncio.get_event_loop()
            info = await loop.run_in_executor(
                self.executor,
                self._get_pdf_info_sync,
                pdf_path
            )
            return info or {}
            
        except Exception as e:
            logger.error(f"Error getting PDF info: {e}")
            return {}
    
    def _get_pdf_info_sync(self, pdf_path: str) -> dict:
        """Synchronous PDF info extraction."""
        try:
            doc = fitz.open(pdf_path)
            info = {
                'page_count': doc.page_count,
                'metadata': doc.metadata,
                'file_size': Path(pdf_path).stat().st_size
            }
            doc.close()
            return info
        except Exception as e:
            logger.error(f"Error in sync PDF info: {e}")
            return {}
    
    def clear_cache(self):
        """Clear the text cache."""
        self._cache.clear()
        logger.info("PDF cache cleared")
    
    async def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        self.clear_cache()
        logger.info("PDFProcessor cleaned up")


# Backward compatibility function
def extract_text_from_pdf(pdf_path: str) -> str:
    """Legacy function for backward compatibility."""
    processor = PDFProcessor()
    return processor.extract_text(pdf_path)
