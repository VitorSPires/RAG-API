import os
import logging
import tiktoken
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import re
from io import BytesIO
from docx import Document
import PyPDF2

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TextChunk:
    """Representa um chunk de texto para vetorização"""
    id: str
    document_id: int
    content: str                    # Texto principal (até 250 tokens)
    overlap_content: str            # Overlap (até 50 tokens, apenas quando necessário)
    chunk_index: int
    token_count: int
    start_position: int
    end_position: int

class DocumentProcessor:
    """Processador de documentos para extração de texto e divisão em chunks"""
    
    def __init__(self):
        # Configurar tokenizer para text-embedding-3-small
        # Usar cl100k_base que é o encoding padrão para modelos recentes da OpenAI
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Configurações de chunks
        self.max_tokens_per_chunk = 300      # Máximo para primeiro chunk
        self.chunk_tokens = 250              # Tamanho padrão dos chunks
        self.overlap_tokens = 50             # Overlap quando necessário
        
        # Extensões suportadas
        self.supported_extensions = {
            '.txt': self._extract_text_from_txt,
            '.docx': self._extract_text_from_docx,
            '.pdf': self._extract_text_from_pdf
        }
    
    def count_tokens(self, text: str) -> int:
        """Contar tokens em um texto usando tiktoken"""
        return len(self.tokenizer.encode(text))

    @staticmethod
    def _sanitize_text_for_db(text: str) -> str:
        """
        Remove bytes NUL (0x00) do texto.

        PostgreSQL não aceita \\x00 em colunas TEXT; PyPDF2 às vezes os inclui
        na extração de PDFs gerados por certas ferramentas.
        """
        if not text or "\x00" not in text:
            return text
        cleaned = text.replace("\x00", "")
        logger.warning(
            "Removidos caracteres NUL (0x00) do texto extraído — comum em alguns PDFs"
        )
        return cleaned

    def extract_text_from_file(self, file_content: bytes, file_name: str) -> Optional[str]:
        """Extrair texto de um arquivo baseado na extensão"""
        logger.info(f"=== INICIANDO EXTRAÇÃO DE TEXTO ===")
        logger.info(f"Arquivo: {file_name}")
        logger.info(f"Tamanho: {len(file_content)} bytes")
        
        try:
            # Obter extensão do arquivo
            file_extension = os.path.splitext(file_name.lower())[1]
            logger.info(f"Extensão detectada: {file_extension}")
            logger.info(f"Extensões suportadas: {list(self.supported_extensions.keys())}")
            
            if file_extension not in self.supported_extensions:
                logger.warning(f"Extensão não suportada: {file_extension}")
                return None
            
            # Extrair texto usando o método apropriado
            extractor = self.supported_extensions[file_extension]
            logger.info(f"Usando extrator: {extractor.__name__}")
            
            text = extractor(file_content)

            if text:
                text = self._sanitize_text_for_db(text)
                logger.info(f"Texto extraído de {file_name}: {len(text)} caracteres")
                return text
            else:
                logger.warning(f"Não foi possível extrair texto de {file_name}")
                return None
                
        except Exception as e:
            logger.error(f"Erro ao extrair texto de {file_name}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _extract_text_from_txt(self, file_content: bytes) -> str:
        """Extrair texto de arquivo .txt"""
        try:
            # Tentar diferentes encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    text = file_content.decode(encoding)
                    return text
                except UnicodeDecodeError:
                    continue
            
            # Se nenhum encoding funcionar, usar utf-8 com tratamento de erro
            return file_content.decode('utf-8', errors='ignore')
            
        except Exception as e:
            logger.error(f"Erro ao extrair texto de arquivo TXT: {e}")
            return ""
    
    def _extract_text_from_docx(self, file_content: bytes) -> str:
        """Extrair texto de arquivo .docx"""
        try:
            doc = Document(BytesIO(file_content))
            text_parts = []
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_parts.append(paragraph.text)
            
            return '\n\n'.join(text_parts)
            
        except Exception as e:
            logger.error(f"Erro ao extrair texto de arquivo DOCX: {e}")
            return ""
    
    def _extract_text_from_pdf(self, file_content: bytes) -> str:
        """Extrair texto de arquivo .pdf"""
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
            text_parts = []
            
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text_parts.append(page_text)
            
            return '\n\n'.join(text_parts)
            
        except Exception as e:
            logger.error(f"Erro ao extrair texto de arquivo PDF: {e}")
            return ""
    
    def split_text_into_paragraphs(self, text: str) -> List[str]:
        """Dividir texto em parágrafos"""
        # Dividir por quebras de linha duplas ou simples
        paragraphs = re.split(r'\n\s*\n|\r\n\s*\r\n', text)
        
        # Filtrar parágrafos vazios e limpar espaços
        cleaned_paragraphs = []
        for paragraph in paragraphs:
            cleaned = paragraph.strip()
            if cleaned:
                cleaned_paragraphs.append(cleaned)
        
        return cleaned_paragraphs
    
    def split_paragraph_into_chunks(self, paragraph: str, chunk_index: int, document_id: int) -> List[TextChunk]:
        """Dividir um parágrafo em chunks conforme as regras especificadas"""
        chunks = []
        current_position = 0
        current_chunk_index = chunk_index
        
        # Se o parágrafo tem 300 tokens ou menos, criar apenas um chunk
        paragraph_tokens = self.count_tokens(paragraph)
        
        if paragraph_tokens <= self.max_tokens_per_chunk:
            # Parágrafo cabe em um chunk
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
            # Parágrafo precisa ser dividido
            # Primeiro chunk: até 300 tokens
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
            
            # Chunks subsequentes: 250 tokens com overlap de 50 tokens
            remaining_text = paragraph[current_position:].strip()
            
            while remaining_text:
                # Pegar os últimos 50 tokens do chunk anterior como overlap
                overlap_text = self._get_text_for_token_count(
                    previous_chunk_text,
                    self.overlap_tokens, 
                    from_end=True
                )
                overlap_tokens = self.count_tokens(overlap_text)
                
                # Calcular quantos tokens novos podemos adicionar
                new_tokens_available = self.chunk_tokens - overlap_tokens
                
                if new_tokens_available <= 0:
                    break
                
                # Pegar o texto novo que cabe
                new_text = self._get_text_for_token_count(remaining_text, new_tokens_available)
                new_tokens = self.count_tokens(new_text)
                
                # Criar chunk
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
        """Obter texto que corresponde a um número específico de tokens"""
        if not text:
            return ""
        
        # Codificar o texto completo
        tokens = self.tokenizer.encode(text)
        
        if len(tokens) <= target_tokens:
            return text
        
        # Pegar os tokens necessários
        if from_end:
            selected_tokens = tokens[-target_tokens:]
        else:
            selected_tokens = tokens[:target_tokens]
        
        # Decodificar de volta para texto
        return self.tokenizer.decode(selected_tokens)
    
    def process_document(self, file_content: bytes, file_name: str, document_id: int) -> List[TextChunk]:
        """Processar documento completo: extrair texto e dividir em chunks"""
        try:
            # Extrair texto do arquivo
            text = self.extract_text_from_file(file_content, file_name)
            
            if not text:
                logger.warning(f"Não foi possível extrair texto de {file_name}")
                return []
            

            
            # Dividir em parágrafos
            paragraphs = self.split_text_into_paragraphs(text)
            
            if not paragraphs:
                logger.warning(f"Nenhum parágrafo encontrado em {file_name}")
                return []
            
            # Processar cada parágrafo
            all_chunks = []
            chunk_index = 0
            
            for i, paragraph in enumerate(paragraphs):
                paragraph_chunks = self.split_paragraph_into_chunks(paragraph, chunk_index, document_id)
                all_chunks.extend(paragraph_chunks)
                chunk_index += len(paragraph_chunks)
            
            logger.info(f"Documento {file_name} processado: {len(all_chunks)} chunks criados")
            return all_chunks
            
        except Exception as e:
            logger.error(f"Erro ao processar documento {file_name}: {e}")
            return []

# Instância global
document_processor = DocumentProcessor() 