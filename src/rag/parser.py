#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Parser for various document types (PDF, CSV, TXT).
Implements structured chunking for PDFs (Korean Insurance Documents)
and general chunking for other types.
Supports "Small-to-Big" chunking strategy preparation.
"""

import os
import re
import json
import uuid
import fitz  # PyMuPDF
import logging
import csv
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path

# LangChain Text Splitter
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.warning("langchain not available. RecursiveCharacterTextSplitter will not be used for advanced chunking.")

# Tokenizer for length function (optional, but recommended for token-based splitting)
try:
    from transformers import AutoTokenizer
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False
    logging.warning("transformers library not available. Token-based length function for splitting will rely on character count.")

# Logger setup
try:
    from src.utils.logger import get_logger
    logger = get_logger("document_parser") # Changed logger name
except ImportError:
    logger = logging.getLogger("document_parser")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.warning("src.utils.logger not found, using basic logging.")


@dataclass
class TextChunk:
    """텍스트 청크를 저장하는 클래스"""
    chunk_id: str
    doc_id: str
    text: str
    parent_chunk_id: Optional[str] = None
    page_num: Optional[int] = None # Page number for PDFs, row number for CSVs, line block for TXT
    bbox: Optional[Tuple[float, float, float, float]] = None
    chunk_type: str = "content" # e.g., "article", "clause", "item", "row", "text_block"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Document:
    """문서 정보를 저장하는 클래스"""
    doc_id: str
    source: str # File path
    title: str  # Document title
    chunks: List[TextChunk]
    parent_chunks: Optional[List[TextChunk]] = None # For "big" chunks in small-to-big
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self):
        data = {
            "doc_id": self.doc_id,
            "source": self.source,
            "title": self.title,
            "chunks": [asdict(chunk) for chunk in self.chunks],
            "metadata": self.metadata
        }
        if self.parent_chunks is not None: # Ensure None is handled correctly
            data["parent_chunks"] = [asdict(chunk) for chunk in self.parent_chunks]
        else:
            data["parent_chunks"] = None
        return data
    
    @classmethod
    def from_dict(cls, data):
        chunks = [TextChunk(**chunk) for chunk in data["chunks"]]
        parent_chunks = None
        if "parent_chunks" in data and data.get("parent_chunks") is not None:
            parent_chunks = [TextChunk(**chunk) for chunk in data["parent_chunks"]]
        
        return cls(
            doc_id=data["doc_id"],
            source=data["source"],
            title=data["title"],
            chunks=chunks,
            parent_chunks=parent_chunks,
            metadata=data.get("metadata", {})
        )

class DocumentParser: # Renamed from InsurancePDFParser for generality
    """
    다양한 문서 형식을 파싱하고 청킹하는 클래스.
    PDF의 경우 보험 약관에 특화된 구조적 청킹 (조항목) 및 Small-to-Big 준비.
    CSV, TXT의 경우 일반적인 청킹 적용.
    """
    
    def __init__(self, config: Optional[Any] = None):
        self.config_data = {}
        if config:
            if hasattr(config, 'config_data'): # If it's our Config object
                self.config_data = config.config_data.get('parser', {})
            elif isinstance(config, dict): # If it's a raw dict
                self.config_data = config.get('parser', config) # Allow passing parser config directly or full config
            else: # Try to access attributes if it's some other object
                try:
                    self.config_data = {
                        'child_chunk_target_size_tokens': getattr(config.chunking, 'child_chunk_size', 250),
                        'child_chunk_overlap_tokens': getattr(config.chunking, 'child_overlap', 30),
                        'parent_chunk_target_size_tokens': getattr(config.chunking, 'parent_chunk_size', 1500),
                        'parent_chunk_overlap_tokens': getattr(config.chunking, 'parent_overlap', 0),
                        'generic_chunk_size_tokens': getattr(config.chunking, 'generic_chunk_size', 500),
                        'generic_chunk_overlap_tokens': getattr(config.chunking, 'generic_overlap', 50),
                        'embedding_model_name': getattr(config.embedding, 'model', "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
                    }
                except AttributeError:
                    logger.warning("Config object provided, but expected attributes (chunking, embedding) not found. Using defaults.")
                    self.config_data = {}


        # PDF (Insurance specific) Small-to-Big Chunking Parameters
        self.child_chunk_target_size_tokens = self.config_data.get('child_chunk_target_size_tokens', 250)
        self.child_chunk_overlap_tokens = self.config_data.get('child_chunk_overlap_tokens', 30)
        self.parent_chunk_target_size_tokens = self.config_data.get('parent_chunk_target_size_tokens', 1500)
        self.parent_chunk_overlap_tokens = self.config_data.get('parent_chunk_overlap_tokens', 0)

        # Generic Chunking Parameters (for TXT, potentially CSV rows if they are long)
        self.generic_chunk_size_tokens = self.config_data.get('generic_chunk_size_tokens', 500)
        self.generic_chunk_overlap_tokens = self.config_data.get('generic_chunk_overlap_tokens', 50)
        
        self.embedding_model_name = self.config_data.get('embedding_model_name', "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")


        # Regex for Korean legal/insurance documents structure
        self.article_pattern = re.compile(r"^\s*제\s?\d+\s?조(?:\s*\(.*?\))?") 
        self.clause_pattern = re.compile(r"^\s*(?:①|②|③|④|⑤|⑥|⑦|⑧|⑨|⑩|⑪|⑫|⑬|⑭|⑮|⑯|⑰|⑱|⑲|⑳)")
        self.item_pattern = re.compile(r"^\s*(?:\d{1,2}\.|[가나다라마바사아자차카타파하]\.|[⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽⑾⑿⒀⒁⒂]\s?|[ⓐⓑⓒⓓⓔ]\s?)") # Added optional space

        self.current_doc_id = None
        self.tokenizer = None
        global TOKENIZER_AVAILABLE  # 전역 변수로 처리
        if TOKENIZER_AVAILABLE and LANGCHAIN_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
                logger.info(f"Tokenizer {self.embedding_model_name} loaded for length calculation.")
            except Exception as e:
                logger.warning(f"Could not load tokenizer {self.embedding_model_name}: {e}. Length function will use character count.")
                TOKENIZER_AVAILABLE = False # Override if loading failed
        
        logger.info(
            f"Parser initialized. PDF Child tokens: {self.child_chunk_target_size_tokens}, overlap: {self.child_chunk_overlap_tokens}. "
            f"Generic tokens: {self.generic_chunk_size_tokens}, overlap: {self.generic_chunk_overlap_tokens}."
        )

    def _token_length_function(self, text: str) -> int:
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return len(text) # Fallback to character length

    def _get_text_blocks_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extracts text blocks with bbox and page number from PDF."""
        all_text_blocks = []
        try:
            doc = fitz.open(pdf_path)
            for page_num_fitz, page in enumerate(doc):
                blocks = page.get_text("dict")["blocks"]
                for b in blocks:
                    if b["type"] == 0: 
                        block_text = ""
                        for line in b["lines"]:
                            for span in line["spans"]:
                                block_text += span["text"]
                            block_text += "\n" 
                        
                        if block_text.strip():
                            all_text_blocks.append({
                                "text": block_text.strip(),
                                "bbox": b["bbox"],
                                "page_num": page_num_fitz + 1 
                            })
            doc.close()
        except Exception as e:
            logger.error(f"Error reading PDF {pdf_path}: {e}")
            raise
        all_text_blocks.sort(key=lambda b: (b["page_num"], b["bbox"][1]))
        return all_text_blocks

    def _is_article_start(self, text: str) -> Optional[str]:
        match = self.article_pattern.match(text.lstrip()) # Use lstrip for matching
        return text.lstrip() if match else None

    def _is_clause_start(self, text: str) -> Optional[str]:
        match = self.clause_pattern.match(text.lstrip())
        return text.lstrip() if match else None

    def _is_item_start(self, text: str) -> Optional[str]:
        match = self.item_pattern.match(text.lstrip())
        return text.lstrip() if match else None

    def _create_pdf_structured_chunks(self, raw_text_blocks: List[Dict[str, Any]], doc_id: str, source_path: str, doc_title: str) -> Tuple[List[TextChunk], List[TextChunk]]:
        small_chunks = [] 
        large_chunks = [] 

        current_article_text_content = ""
        current_article_full_title = ""
        current_article_page_start = -1
        current_article_id = None

        current_clause_marker = ""
        
        temp_item_buffer = [] # Stores {"text": ..., "page_num": ..., "bbox": ...} for the current item

        def finalize_item_chunk():
            nonlocal temp_item_buffer, current_article_id, current_article_full_title, current_clause_marker
            if temp_item_buffer:
                item_text = "\n".join([b["text"] for b in temp_item_buffer]).strip()
                if item_text:
                    first_block_in_item = temp_item_buffer[0]
                    # Extract item marker (e.g., "1.", "가.") from the first line of the item text
                    item_marker_match = self.item_pattern.match(item_text.lstrip())
                    item_marker = item_marker_match.group(0).strip() if item_marker_match else ""

                    chunk_metadata = {
                        "source_file": Path(source_path).name, # Keep original filename
                        "document_title": doc_title,
                        "article_title": current_article_full_title,
                        "clause_marker": current_clause_marker,
                        "item_marker": item_marker,
                    }
                    
                    # Sub-chunking for very long items
                    if LANGCHAIN_AVAILABLE:
                        item_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=self.child_chunk_target_size_tokens,
                            chunk_overlap=self.child_chunk_overlap_tokens,
                            length_function=self._token_length_function,
                            separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""], # More granular separators
                            keep_separator=False
                        )
                        # Check if item_text needs splitting
                        if self._token_length_function(item_text) > self.child_chunk_target_size_tokens:
                            sub_texts = item_splitter.split_text(item_text)
                            for i, sub_text_content in enumerate(sub_texts):
                                small_chunks.append(TextChunk(
                                    chunk_id=str(uuid.uuid4()),
                                    doc_id=doc_id,
                                    parent_chunk_id=current_article_id,
                                    text=sub_text_content,
                                    page_num=first_block_in_item["page_num"],
                                    bbox=first_block_in_item["bbox"], # Bbox less accurate for sub-chunks
                                    chunk_type="item_sub_chunk",
                                    metadata={**chunk_metadata, "sub_chunk_index": i, "original_item_text_start": item_text[:50]+"..."}
                                ))
                        else: # Item is short enough
                            small_chunks.append(TextChunk(
                                chunk_id=str(uuid.uuid4()),
                                doc_id=doc_id,
                                parent_chunk_id=current_article_id,
                                text=item_text,
                                page_num=first_block_in_item["page_num"],
                                bbox=first_block_in_item["bbox"],
                                chunk_type="item",
                                metadata=chunk_metadata
                            ))
                    else: # Langchain not available, create one chunk per item
                         small_chunks.append(TextChunk(
                            chunk_id=str(uuid.uuid4()),
                            doc_id=doc_id,
                            parent_chunk_id=current_article_id,
                            text=item_text,
                            page_num=first_block_in_item["page_num"],
                            bbox=first_block_in_item["bbox"],
                            chunk_type="item",
                            metadata=chunk_metadata
                        ))
                temp_item_buffer = []

        def finalize_article_chunk():
            nonlocal current_article_text_content, large_chunks, current_article_full_title, current_article_page_start, current_article_id
            finalize_item_chunk() # Finalize any pending item within the article
            if current_article_text_content.strip() and current_article_full_title:
                # Use the already generated current_article_id
                chunk_metadata = {
                    "source_file": Path(source_path).name,
                    "document_title": doc_title,
                    "article_full_title": current_article_full_title, 
                }
                large_chunks.append(TextChunk(
                    chunk_id=current_article_id, # Use the ID established at article start
                    doc_id=doc_id,
                    text=current_article_text_content.strip(),
                    page_num=current_article_page_start,
                    chunk_type="article",
                    metadata=chunk_metadata
                ))
            current_article_text_content = ""
            current_article_full_title = ""
            current_article_id = None # Reset for next article
            current_clause_marker = ""


        for block in raw_text_blocks:
            # Iterate through lines within a block, as structure markers often start on new lines
            current_block_text_lines = block["text"].split('\n')
            
            for line_idx, line_text_original in enumerate(current_block_text_lines):
                line_text = line_text_original.strip()
                if not line_text:
                    continue

                is_art_start = self._is_article_start(line_text)
                is_cls_start = self._is_clause_start(line_text) if not is_art_start else None
                is_itm_start = self._is_item_start(line_text) if not is_art_start and not is_cls_start else None
                
                if is_art_start:
                    finalize_article_chunk() # Finalize previous article and its items
                    current_article_full_title = is_art_start 
                    current_article_text_content = line_text_original + "\n" # Start accumulating text for new article
                    current_article_page_start = block["page_num"]
                    current_article_id = str(uuid.uuid4()) # Generate ID for this new article
                    current_clause_marker = "" # Reset clause marker
                    # Item buffer is cleared by finalize_article_chunk via finalize_item_chunk
                
                elif is_cls_start:
                    finalize_item_chunk() # Finalize previous item before new clause
                    current_clause_marker = is_cls_start.split(' ')[0].strip() # e.g., "①"
                    current_article_text_content += line_text_original + "\n"
                    # Add clause line to item buffer as it might be start of an item content
                    temp_item_buffer.append({"text": line_text_original, "page_num": block["page_num"], "bbox": block["bbox"]})

                elif is_itm_start:
                    finalize_item_chunk() # Finalize previous item
                    current_article_text_content += line_text_original + "\n"
                    # Start new item buffer
                    temp_item_buffer.append({"text": line_text_original, "page_num": block["page_num"], "bbox": block["bbox"]})
                
                else: # Continuation text
                    if current_article_full_title: # Only append if we are within an article context
                        current_article_text_content += line_text_original + "\n"
                        temp_item_buffer.append({"text": line_text_original, "page_num": block["page_num"], "bbox": block["bbox"]})
                    else:
                        # Text before the first article, could be a preface or ToC.
                        # For now, we might ignore it or collect it as a separate type of chunk if needed.
                        logger.debug(f"Orphan text block (page {block['page_num']}): {line_text[:100]}...")
                        # Optionally, create a 'preface' chunk if this text is substantial
                        # if len(line_text) > 100: # Arbitrary threshold
                        #     small_chunks.append(TextChunk(... chunk_type="preface" ...))
        
        finalize_article_chunk() # Finalize the last article and its items

        return small_chunks, large_chunks

    def process_pdf(self, pdf_path: str, output_dir: Optional[str] = None) -> Optional[Document]:
        logger.info(f"PDF 처리 시작: {pdf_path}")
        if not os.path.exists(pdf_path):
            logger.error(f"PDF 파일을 찾을 수 없음: {pdf_path}")
            return None
        
        doc_uuid = str(uuid.uuid4())
        file_name = os.path.basename(pdf_path)
        doc_title = os.path.splitext(file_name)[0]

        if not output_dir:
            output_dir = Path(pdf_path).parent
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        try:
            raw_text_blocks = self._get_text_blocks_from_pdf(pdf_path)
            if not raw_text_blocks:
                logger.warning(f"텍스트 블록을 추출하지 못했습니다: {pdf_path}")
                return None

            item_chunks, article_chunks = self._create_pdf_structured_chunks(raw_text_blocks, doc_uuid, pdf_path, doc_title)
            
            document = Document(
                doc_id=doc_uuid,
                source=pdf_path,
                title=doc_title,
                chunks=item_chunks,
                parent_chunks=article_chunks,
                metadata={"original_filename": file_name, "parser_type": "pdf_insurance_v2"}
            )
            
            output_filename = f"{doc_title}_parsed.json" # Changed suffix
            output_path = Path(output_dir) / output_filename
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(document.to_dict(), f, ensure_ascii=False, indent=2)
            logger.info(f"PDF 파싱 결과 저장 완료: {output_path} (Small: {len(item_chunks)}, Large: {len(article_chunks)})")
            return document
        except Exception as e:
            logger.error(f"PDF 처리 중 오류 발생 {pdf_path}: {e}", exc_info=True)
            return None

    def process_csv(self, csv_path: str, output_dir: Optional[str] = None) -> Optional[Document]:
        logger.info(f"CSV 처리 시작: {csv_path}")
        if not os.path.exists(csv_path):
            logger.error(f"CSV 파일을 찾을 수 없음: {csv_path}")
            return None

        doc_uuid = str(uuid.uuid4())
        file_name = os.path.basename(csv_path)
        doc_title = os.path.splitext(file_name)[0]
        
        if not output_dir:
            output_dir = Path(csv_path).parent
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        item_chunks = []
        try:
            with open(csv_path, 'r', encoding='utf-8-sig') as f: # utf-8-sig for BOM
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    # Combine all columns into a single text string for each row
                    # Or select specific columns if preferred
                    row_text = ". ".join([f"{col_name}: {col_val}" for col_name, col_val in row.items() if col_val])
                    if not row_text.strip():
                        continue

                    chunk_metadata = {
                        "source_file": file_name,
                        "document_title": doc_title,
                        "row_number": i + 1, # 1-indexed row
                        "csv_headers": list(row.keys())
                    }
                    item_chunks.append(TextChunk(
                        chunk_id=str(uuid.uuid4()),
                        doc_id=doc_uuid,
                        parent_chunk_id=None, # No parent chunk concept for simple CSV row chunking
                        text=row_text,
                        page_num=i + 1, # Use row number as page_num
                        chunk_type="csv_row",
                        metadata=chunk_metadata
                    ))
            
            # For CSVs, parent_chunks will be None or a single chunk representing the whole file if needed.
            # For simplicity here, we'll set it to None.
            document = Document(
                doc_id=doc_uuid,
                source=csv_path,
                title=doc_title,
                chunks=item_chunks,
                parent_chunks=None, 
                metadata={"original_filename": file_name, "parser_type": "csv_basic"}
            )

            output_filename = f"{doc_title}_parsed.json"
            output_path = Path(output_dir) / output_filename
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(document.to_dict(), f, ensure_ascii=False, indent=2)
            logger.info(f"CSV 파싱 결과 저장 완료: {output_path} (Chunks: {len(item_chunks)})")
            return document

        except Exception as e:
            logger.error(f"CSV 처리 중 오류 발생 {csv_path}: {e}", exc_info=True)
            return None

    def process_text(self, txt_path: str, output_dir: Optional[str] = None) -> Optional[Document]:
        logger.info(f"TXT 처리 시작: {txt_path}")
        if not os.path.exists(txt_path):
            logger.error(f"TXT 파일을 찾을 수 없음: {txt_path}")
            return None

        doc_uuid = str(uuid.uuid4())
        file_name = os.path.basename(txt_path)
        doc_title = os.path.splitext(file_name)[0]

        if not output_dir:
            output_dir = Path(txt_path).parent
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        item_chunks = []
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                full_text_content = f.read()

            if not full_text_content.strip():
                logger.warning(f"TXT 파일이 비어있습니다: {txt_path}")
                return None

            if LANGCHAIN_AVAILABLE:
                txt_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.generic_chunk_size_tokens,
                    chunk_overlap=self.generic_chunk_overlap_tokens,
                    length_function=self._token_length_function,
                    separators=["\n\n", "\n", ". ", " ", ""], # Standard separators
                    keep_separator=False
                )
                split_texts = txt_splitter.split_text(full_text_content)
                
                for i, text_part in enumerate(split_texts):
                    chunk_metadata = {
                        "source_file": file_name,
                        "document_title": doc_title,
                        "text_block_index": i
                    }
                    item_chunks.append(TextChunk(
                        chunk_id=str(uuid.uuid4()),
                        doc_id=doc_uuid,
                        parent_chunk_id=None, # No parent chunk for simple TXT chunking initially
                        text=text_part,
                        page_num=1, # TXT files don't have pages in the PDF sense
                        chunk_type="text_block",
                        metadata=chunk_metadata
                    ))
            else: # Fallback if LangChain is not available
                # Simple splitting by paragraph or fixed length
                # This is a very basic fallback
                paragraphs = full_text_content.split("\n\n")
                for i, para_text in enumerate(paragraphs):
                    if para_text.strip():
                        item_chunks.append(TextChunk(
                            chunk_id=str(uuid.uuid4()),
                            doc_id=doc_uuid,
                            text=para_text.strip(),
                            page_num=1,
                            chunk_type="paragraph_fallback",
                            metadata={"source_file": file_name, "document_title": doc_title}
                        ))
            
            document = Document(
                doc_id=doc_uuid,
                source=txt_path,
                title=doc_title,
                chunks=item_chunks,
                parent_chunks=None, 
                metadata={"original_filename": file_name, "parser_type": "text_generic"}
            )

            output_filename = f"{doc_title}_parsed.json"
            output_path = Path(output_dir) / output_filename
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(document.to_dict(), f, ensure_ascii=False, indent=2)
            logger.info(f"TXT 파싱 결과 저장 완료: {output_path} (Chunks: {len(item_chunks)})")
            return document

        except Exception as e:
            logger.error(f"TXT 처리 중 오류 발생 {txt_path}: {e}", exc_info=True)
            return None

    def process_directory(self, input_dir: str, output_dir: Optional[str] = None) -> List[Document]:
        if not os.path.isdir(input_dir):
            logger.error(f"입력 경로가 디렉토리가 아닙니다: {input_dir}")
            raise NotADirectoryError(f"디렉토리가 아님: {input_dir}")
        
        if not output_dir:
            output_dir = input_dir 
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        all_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        logger.info(f"디렉토리 '{input_dir}'에서 {len(all_files)}개의 파일 발견.")
        
        processed_documents = []
        for file_name in all_files:
            file_path = os.path.join(input_dir, file_name)
            document = None
            try:
                if file_name.lower().endswith('.pdf'):
                    document = self.process_pdf(file_path, output_dir)
                elif file_name.lower().endswith('.csv'):
                    document = self.process_csv(file_path, output_dir)
                elif file_name.lower().endswith('.txt'):
                    document = self.process_text(file_path, output_dir)
                else:
                    logger.warning(f"지원하지 않는 파일 형식: {file_name}. 건너뜁니다.")
                
                if document:
                    processed_documents.append(document)
            except Exception as e:
                logger.error(f"파일 처리 중 일반 오류 발생: {file_path}, 오류: {e}", exc_info=True)

        logger.info(f"총 {len(processed_documents)}개의 문서 처리 완료.")
        return processed_documents


# Example Usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("사용법: python parser.py <파일 경로 또는 디렉토리 경로> [출력 디렉토리]")
        sys.exit(1)
    
    input_path_arg = sys.argv[1]
    output_dir_arg = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Create a dummy config for testing, or load from your actual config system
    class DummyConfig:
        def __init__(self):
            self.parser = { # Corresponds to parser section in config
                'child_chunk_target_size_tokens': 250, 
                'child_chunk_overlap_tokens': 30,
                'parent_chunk_target_size_tokens': 1500,
                'parent_chunk_overlap_tokens': 0,
                'generic_chunk_size_tokens': 300, # Smaller for general text
                'generic_chunk_overlap_tokens': 30,
                'embedding_model_name': "sentence-transformers/paraphrase-multilingual-mpnet-base-v2" # Ensure this matches your embedder
            }
            # Mocking other parts of config if parser tries to access them
            self.embedding = {'model': self.parser['embedding_model_name']}


    parser_instance = DocumentParser(config=DummyConfig()) 
    
    if os.path.isdir(input_path_arg):
        logger.info(f"디렉토리 처리 시작: {input_path_arg}")
        documents = parser_instance.process_directory(input_path_arg, output_dir_arg)
        print(f"총 {len(documents)}개의 문서 처리 완료.")
        # for doc_obj in documents:
        #     print(f" - {doc_obj.title}: {len(doc_obj.chunks)} small chunks, {len(doc_obj.parent_chunks) if doc_obj.parent_chunks else 0} parent chunks")
    elif os.path.isfile(input_path_arg):
        logger.info(f"단일 파일 처리 시작: {input_path_arg}")
        document_obj = None
        if input_path_arg.lower().endswith('.pdf'):
            document_obj = parser_instance.process_pdf(input_path_arg, output_dir_arg)
        elif input_path_arg.lower().endswith('.csv'):
            document_obj = parser_instance.process_csv(input_path_arg, output_dir_arg)
        elif input_path_arg.lower().endswith('.txt'):
            document_obj = parser_instance.process_text(input_path_arg, output_dir_arg)
        else:
            print(f"지원하지 않는 파일 형식입니다: {input_path_arg}")
            sys.exit(1)

        if document_obj:
            small_chunk_count = len(document_obj.chunks)
            parent_chunk_count = len(document_obj.parent_chunks) if document_obj.parent_chunks is not None else 0
            print(f"문서 처리 완료: '{document_obj.title}', Small Chunks: {small_chunk_count}, Parent Chunks: {parent_chunk_count}")
        else:
            print(f"문서 처리 실패: {input_path_arg}")
    else:
        logger.error(f"잘못된 입력 경로입니다: {input_path_arg}")
        sys.exit(1)