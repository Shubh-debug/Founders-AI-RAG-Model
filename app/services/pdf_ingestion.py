"""
PDF document ingestion service for startup case study extraction and knowledge base integration.

Provides PDF text extraction, intelligent chunking, and startup document ingestion
into the Founders AI knowledge base with error handling and metadata management.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from PyPDF2 import PdfReader
from ..core.exceptions import DocumentProcessingError
from ..core.utils import text_processor
from ..core.config import settings
from .lightweight_llm_rag_founders_ai import lightweight_llm_rag
from .enhanced_metadata_processor_founders_ai import enhanced_metadata_processor

logger = logging.getLogger(__name__)


# -------------------------------------------------------
# TEXT EXTRACTION
# -------------------------------------------------------
class PDFTextExtractor:
    """Extracts and cleans text from PDF documents with error handling."""

    def __init__(self):
        self.max_text_length = getattr(settings, "max_text_length", 10000)

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file with enhanced cleaning.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            str: Extracted and cleaned text

        Raises:
            DocumentProcessingError
        """
        try:
            if not os.path.exists(pdf_path):
                raise DocumentProcessingError(
                    f"PDF file not found: {pdf_path}",
                    document_id=os.path.basename(pdf_path),
                    processing_stage="file_validation"
                )

            try:
                reader = PdfReader(pdf_path)
            except Exception as e:
                raise DocumentProcessingError(
                    f"Error loading PDF file: {str(e)}",
                    document_id=os.path.basename(pdf_path),
                    processing_stage="pdf_loading"
                )

            extracted_text_parts = []
            for page_number, page in enumerate(reader.pages, start=1):
                try:
                    page_text = page.extract_text()
                    if not page_text:
                        continue
                    cleaned_text = self._clean_extracted_text(page_text)
                    if cleaned_text.strip():
                        extracted_text_parts.append(f"\n--- Page {page_number} ---\n{cleaned_text}")
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_number}: {e}")

            full_text = "".join(extracted_text_parts).strip()
            if not full_text:
                raise DocumentProcessingError(
                    f"No text extracted from PDF: {pdf_path}",
                    document_id=os.path.basename(pdf_path),
                    processing_stage="text_extraction"
                )

            # Truncate large files safely
            if len(full_text) > self.max_text_length:
                half = self.max_text_length // 2
                full_text = f"{full_text[:half]}\n\n...[truncated]...\n\n{full_text[-half:]}"

            return self._clean_extracted_text(full_text)

        except DocumentProcessingError:
            raise
        except Exception as e:
            raise DocumentProcessingError(
                f"Failed to read PDF {pdf_path}: {str(e)}",
                document_id=os.path.basename(pdf_path),
                processing_stage="pdf_reading"
            )

    def _clean_extracted_text(self, text: str) -> str:
        """Clean extracted text safely using text_processor."""
        if not text:
            return ""
        try:
            return text_processor.clean_text_comprehensive(text)
        except Exception as e:
            logger.warning(f"Text cleaning failed: {e}, using fallback cleaning.")
            return " ".join(text.replace("\r", " ").replace("\t", " ").split())


# -------------------------------------------------------
# TEXT CHUNKER
# -------------------------------------------------------
class TextChunker:
    """Splits text into manageable chunks while preserving sentence boundaries."""

    def __init__(self, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None):
        self.chunk_size = int(chunk_size or getattr(settings, "pdf_chunk_size", 1000))
        self.chunk_overlap = int(chunk_overlap or getattr(settings, "pdf_chunk_overlap", 200))
        if self.chunk_overlap >= self.chunk_size:
            self.chunk_overlap = max(0, self.chunk_size // 10)

    def split_text_into_chunks(self, text: str) -> List[str]:
        """Split text into chunks while preserving natural sentence boundaries."""
        if not text:
            return []
        text = text.strip()
        n = len(text)
        if n <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < n:
            end = min(start + self.chunk_size, n)
            end = self._find_sentence_boundary(text, start, end)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = max(end - self.chunk_overlap, end)
            if start >= n:
                break
        return chunks

    def _find_sentence_boundary(self, text: str, start: int, end: int) -> int:
        """Find best boundary near chunk limit (sentence end or newline)."""
        for i in range(end - 1, start, -1):
            if text[i] in ".!?":
                return i + 1
            if text[i] == "\n":
                return i
        return end


# -------------------------------------------------------
# MAIN INGESTION SERVICE
# -------------------------------------------------------
class PDFIngestionService:
    """Service for ingesting startup PDFs into the Founders AI knowledge base."""

    def __init__(self, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None):
        self.text_extractor = PDFTextExtractor()
        self.text_chunker = TextChunker(chunk_size, chunk_overlap)

    async def ingest_single_pdf(self, pdf_path: str, source: str = "startup_case_study") -> str:
        """
        Ingest a single startup case study PDF into Founders AI.

        Args:
            pdf_path: Path to PDF file
            source: Source label

        Returns:
            Success message
        """
        try:
            # Extract text
            extracted_text = self.text_extractor.extract_text_from_pdf(pdf_path)

            # Split text into chunks
            text_chunks = self.text_chunker.split_text_into_chunks(extracted_text)
            if not text_chunks:
                raise DocumentProcessingError(
                    f"No chunks produced from {pdf_path}",
                    document_id=os.path.basename(pdf_path),
                    processing_stage="chunking"
                )

            # Prepare documents for ingestion
            documents = self._prepare_documents_for_ingestion(text_chunks, pdf_path, source)

            # Send to knowledge base
            document_ids = await lightweight_llm_rag.add_case_studies_bulk(documents)
            if not document_ids:
                raise DocumentProcessingError(
                    f"No document IDs returned for {pdf_path}",
                    document_id=os.path.basename(pdf_path),
                    processing_stage="database_response"
                )

            msg = f"✅ Successfully ingested {len(document_ids)} chunks from {os.path.basename(pdf_path)}"
            logger.info(msg)
            return msg

        except DocumentProcessingError:
            raise
        except Exception as e:
            logger.exception(f"Error ingesting {pdf_path}: {e}")
            raise DocumentProcessingError(
                f"Unexpected ingestion failure: {str(e)}",
                document_id=os.path.basename(pdf_path),
                processing_stage="ingestion"
            )

    async def ingest_multiple_pdfs(self, pdf_paths: List[str], source: str = "startup_case_study") -> List[str]:
        """Ingest multiple PDFs with robust error handling."""
        results = []
        for pdf_path in pdf_paths:
            try:
                res = await self.ingest_single_pdf(pdf_path, source)
                results.append(res)
            except DocumentProcessingError as e:
                err_msg = f"❌ Failed to ingest {pdf_path}: {getattr(e, 'message', str(e))}"
                logger.error(err_msg)
                results.append(err_msg)
            except Exception as e:
                err_msg = f"❌ Unexpected error for {pdf_path}: {str(e)}"
                logger.error(err_msg)
                results.append(err_msg)
        return results

    def _prepare_documents_for_ingestion(
        self, text_chunks: List[str], pdf_path: str, source: str
    ) -> List[Dict[str, Any]]:
        """Prepare startup case study document data with metadata."""
        documents = []
        filename = os.path.basename(pdf_path)

        for chunk_index, chunk in enumerate(text_chunks):
            chunk = chunk.strip()
            if not chunk:
                continue

            try:
                enhanced_metadata = enhanced_metadata_processor.process_document_metadata(
                    content=chunk,
                    filename=filename,
                    chunk_index=chunk_index,
                    total_chunks=len(text_chunks),
                    file_path=pdf_path,
                    source=source
                )
            except Exception as e:
                logger.warning(f"Metadata processing failed for {filename} chunk {chunk_index}: {e}")
                enhanced_metadata = {
                    "filename": filename,
                    "chunk_index": chunk_index,
                    "total_chunks": len(text_chunks),
                    "source": source,
                    "enhanced": False,
                    "error": str(e)
                }

            documents.append({
                "content": chunk,
                "company": self._extract_company_from_filename(filename),
                "metadata": enhanced_metadata
            })

        return documents

    def _extract_company_from_filename(self, filename: str) -> str:
        """Extract probable company name from filename."""
        known_companies = [
            "PolicyBazaar", "Groww", "CoinDCX", "ShareChat",
            "Fractal", "FirstCry", "Good Glamm", "Glamm"
        ]
        f_lower = filename.lower()
        for company in known_companies:
            if company.lower() in f_lower:
                return company
        return os.path.splitext(filename)[0] or "Unknown"


# -------------------------------------------------------
# GLOBAL INSTANCE & WRAPPER
# -------------------------------------------------------
pdf_ingestion_service = PDFIngestionService()


async def ingest_pdfs(pdf_paths: List[str], source: str = "startup_case_study") -> List[str]:
    """Convenience wrapper for batch PDF ingestion."""
    return await pdf_ingestion_service.ingest_multiple_pdfs(pdf_paths, source)
