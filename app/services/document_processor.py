import os
import logging
import tiktoken
from typing import List, Dict, Optional
from dataclasses import dataclass
import re
from io import BytesIO
from docx import Document
import PyPDF2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """Text segment for embedding and storage."""
    id: str
    document_id: int
    content: str
    overlap_content: str
    chunk_index: int
    token_count: int
    start_position: int
    end_position: int


class DocumentProcessor:
    """Extract text from supported files and split into token-aware chunks."""

    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.max_tokens_per_chunk = 300
        self.chunk_tokens = 250
        self.overlap_tokens = 50
        self.supported_extensions = {
            '.txt': self._extract_text_from_txt,
            '.docx': self._extract_text_from_docx,
            '.pdf': self._extract_text_from_pdf
        }

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    @staticmethod
    def _sanitize_text_for_db(text: str) -> str:
        if not text or "\x00" not in text:
            return text
        cleaned = text.replace("\x00", "")
        logger.warning(
            "Stripped NUL (0x00) characters from extracted text (common in some PDFs)"
        )
        return cleaned

    def extract_text_from_file(self, file_content: bytes, file_name: str) -> Optional[str]:
        logger.info("=== TEXT EXTRACTION START ===")
        logger.info("File: %s", file_name)
        logger.info("Size: %s bytes", len(file_content))

        try:
            file_extension = os.path.splitext(file_name.lower())[1]
            logger.info("Detected extension: %s", file_extension)

            if file_extension not in self.supported_extensions:
                logger.warning("Unsupported extension: %s", file_extension)
                return None

            extractor = self.supported_extensions[file_extension]
            text = extractor(file_content)

            if text:
                text = self._sanitize_text_for_db(text)
                logger.info("Extracted text from %s: %s characters", file_name, len(text))
                return text
            logger.warning("Could not extract text from %s", file_name)
            return None

        except Exception as e:
            logger.error("Error extracting text from %s: %s", file_name, e)
            import traceback
            logger.error("Traceback: %s", traceback.format_exc())
            return None

    def _extract_text_from_txt(self, file_content: bytes) -> str:
        try:
            encodings = ['utf-8', 'latin-1', 'cp1252']
            for encoding in encodings:
                try:
                    return file_content.decode(encoding)
                except UnicodeDecodeError:
                    continue
            return file_content.decode('utf-8', errors='ignore')
        except Exception as e:
            logger.error("Error extracting text from TXT: %s", e)
            return ""

    def _extract_text_from_docx(self, file_content: bytes) -> str:
        try:
            doc = Document(BytesIO(file_content))
            text_parts = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_parts.append(paragraph.text)
            return '\n\n'.join(text_parts)
        except Exception as e:
            logger.error("Error extracting text from DOCX: %s", e)
            return ""

    def _extract_text_from_pdf(self, file_content: bytes) -> str:
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
            text_parts = []
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text_parts.append(page_text)
            return '\n\n'.join(text_parts)
        except Exception as e:
            logger.error("Error extracting text from PDF: %s", e)
            return ""

    def split_text_into_paragraphs(self, text: str) -> List[str]:
        paragraphs = re.split(r'\n\s*\n|\r\n\s*\r\n', text)
        cleaned_paragraphs = []
        for paragraph in paragraphs:
            cleaned = paragraph.strip()
            if cleaned:
                cleaned_paragraphs.append(cleaned)
        return cleaned_paragraphs

    def split_paragraph_into_chunks(self, paragraph: str, chunk_index: int, document_id: int) -> List[TextChunk]:
        chunks = []
        current_position = 0
        current_chunk_index = chunk_index
        paragraph_tokens = self.count_tokens(paragraph)

        if paragraph_tokens <= self.max_tokens_per_chunk:
            chunk = TextChunk(
                id=f"doc_{document_id}_chunk_{current_chunk_index}",
                document_id=document_id,
                content=paragraph,
                overlap_content="",
                chunk_index=current_chunk_index,
                token_count=paragraph_tokens,
                start_position=current_position,
                end_position=current_position + len(paragraph)
            )
            chunks.append(chunk)
        else:
            first_chunk_text = self._get_text_for_token_count(paragraph, self.max_tokens_per_chunk)
            first_chunk_tokens = self.count_tokens(first_chunk_text)

            first_chunk = TextChunk(
                id=f"doc_{document_id}_chunk_{current_chunk_index}",
                document_id=document_id,
                content=first_chunk_text,
                overlap_content="",
                chunk_index=current_chunk_index,
                token_count=first_chunk_tokens,
                start_position=current_position,
                end_position=current_position + len(first_chunk_text)
            )
            chunks.append(first_chunk)

            current_position += len(first_chunk_text)
            current_chunk_index += 1
            previous_chunk_text = first_chunk_text
            remaining_text = paragraph[current_position:].strip()

            while remaining_text:
                overlap_text = self._get_text_for_token_count(
                    previous_chunk_text,
                    self.overlap_tokens,
                    from_end=True
                )
                overlap_tokens = self.count_tokens(overlap_text)
                new_tokens_available = self.chunk_tokens - overlap_tokens

                if new_tokens_available <= 0:
                    break

                new_text = self._get_text_for_token_count(remaining_text, new_tokens_available)
                new_tokens = self.count_tokens(new_text)

                chunk = TextChunk(
                    id=f"doc_{document_id}_chunk_{current_chunk_index}",
                    document_id=document_id,
                    content=new_text,
                    overlap_content=overlap_text,
                    chunk_index=current_chunk_index,
                    token_count=new_tokens,
                    start_position=current_position,
                    end_position=current_position + len(new_text)
                )
                chunks.append(chunk)

                current_position += len(new_text)
                current_chunk_index += 1
                previous_chunk_text = new_text
                remaining_text = remaining_text[len(new_text):].strip()

        return chunks

    def _get_text_for_token_count(self, text: str, target_tokens: int, from_end: bool = False) -> str:
        if not text:
            return ""
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= target_tokens:
            return text
        if from_end:
            selected_tokens = tokens[-target_tokens:]
        else:
            selected_tokens = tokens[:target_tokens]
        return self.tokenizer.decode(selected_tokens)

    def process_document(self, file_content: bytes, file_name: str, document_id: int) -> List[TextChunk]:
        try:
            text = self.extract_text_from_file(file_content, file_name)
            if not text:
                logger.warning("Could not extract text from %s", file_name)
                return []

            paragraphs = self.split_text_into_paragraphs(text)
            if not paragraphs:
                logger.warning("No paragraphs found in %s", file_name)
                return []

            all_chunks = []
            chunk_index = 0
            for paragraph in paragraphs:
                paragraph_chunks = self.split_paragraph_into_chunks(paragraph, chunk_index, document_id)
                all_chunks.extend(paragraph_chunks)
                chunk_index += len(paragraph_chunks)

            logger.info("Processed document %s: %s chunks created", file_name, len(all_chunks))
            return all_chunks

        except Exception as e:
            logger.error("Error processing document %s: %s", file_name, e)
            return []


document_processor = DocumentProcessor()
