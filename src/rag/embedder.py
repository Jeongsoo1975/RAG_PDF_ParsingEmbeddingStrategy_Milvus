#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Document Embedder module for generating and storing vector embeddings.
Supports different embedding models with focus on Korean language support.
"""

import os
import json
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
from pathlib import Path
import re
from tqdm import tqdm
import uuid
import time
import argparse

from dotenv import load_dotenv

# .env 파일 로드
try:
    current_path = Path(__file__).resolve()
    project_root_candidates = [current_path.parent.parent.parent, current_path.parent.parent, current_path.parent]
    env_path_found = None
    for candidate_path in project_root_candidates:
        potential_env_path = candidate_path / '.env'
        if potential_env_path.exists():
            env_path_found = potential_env_path
            break
    if env_path_found:
        load_dotenv(dotenv_path=env_path_found)
        print(f"Loaded .env file from: {env_path_found}")
    else:
        load_dotenv()
        print("Attempting to load .env from current working directory or default locations.")
except Exception as e:
    print(f"Error determining .env path: {e}. Will try default load_dotenv().")
    load_dotenv()

# 라이브러리 임포트 (선택적 로딩)
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    print("sentence_transformers successfully imported")
except ImportError as e:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print(f"sentence_transformers not available: {e}")
except Exception as e:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print(f"Error importing sentence_transformers: {e}")

try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
    print("transformers successfully imported")
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    print(f"transformers not available: {e}")
except Exception as e:
    TRANSFORMERS_AVAILABLE = False
    print(f"Error importing transformers: {e}")

try:
    import pymilvus
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False

try:
    from src.vectordb import MilvusClient
    VECTORDB_CLIENT_AVAILABLE = True
except ImportError:
    VECTORDB_CLIENT_AVAILABLE = False

from src.utils.config import Config
from src.utils.logger import get_logger
from src.rag.parser import Document, TextChunk
# Import Document and TextChunk classes from parser module

logger = get_logger("embedder")

if not SENTENCE_TRANSFORMERS_AVAILABLE: logger.warning("sentence-transformers not available.")
if not TRANSFORMERS_AVAILABLE: logger.warning("transformers not available.")
if not MILVUS_AVAILABLE: logger.warning("pymilvus not available.")
if not VECTORDB_CLIENT_AVAILABLE: logger.warning("vectordb client not available. MilvusClient will not be used.")


class DocumentEmbedder:
    """Class to generate and store embeddings for document chunks using Milvus"""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.logger = logger

        # 임베딩 파라미터
        self.embedding_model_name = self.config.embedding.get('model', "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        self.embedding_dimension = self.config.embedding.get('dimension', None)
        self.batch_size = self.config.embedding.get('batch_size', 16)
        self.normalize_embeddings = self.config.embedding.get('normalize', True)

        # 디바이스 설정
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            try:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                self.logger.info(f"GPU detected: {gpu_name} ({gpu_memory:.2f} GB)")
                torch.cuda.set_device(0)
            except Exception as e:
                self.logger.error(f"Error accessing GPU: {e}. Switching to CPU.")
                self.device = "cpu"
        else:
            self.logger.warning("GPU not available, using CPU.")

        # 모델 로드
        self.model = self._load_model()

        # 임베딩 차원 결정
        if self.model is not None and self.embedding_dimension is None:
            try:
                if isinstance(self.model, SentenceTransformer):
                    self.embedding_dimension = self.model.get_sentence_embedding_dimension()
                else:
                    test_embedding = self.embed_text("Dimension check")
                    self.embedding_dimension = len(test_embedding)
                self.logger.info(f"Determined dimension: {self.embedding_dimension}")
            except Exception as e:
                self.logger.error(f"Could not determine dimension: {e}. Using 768.")
                self.embedding_dimension = 768
        elif self.embedding_dimension is None:
             self.embedding_dimension = 768
             self.logger.warning(f"Using default dimension: {self.embedding_dimension}")
        else:
            self.logger.info(f"Using configured dimension: {self.embedding_dimension}")

        # Milvus 클라이언트 초기화
        self.milvus_client = None
        self.collections = []
        
        if MILVUS_AVAILABLE and VECTORDB_CLIENT_AVAILABLE:
            self._init_milvus()
            if self.milvus_client:
                self.collections = self.milvus_client.list_collections()
                self.logger.info(f"Available Milvus collections: {self.collections}")
        else:
            self.logger.error("Milvus not available. Please install pymilvus.")

    def _init_milvus(self):
        """Milvus 클라이언트 초기화."""
        try:
            if not MILVUS_AVAILABLE or not VECTORDB_CLIENT_AVAILABLE:
                self.logger.error("pymilvus 또는 MilvusClient가 설치되지 않았습니다.")
                return
                
            self.milvus_client = MilvusClient(self.config)
            if self.milvus_client.is_connected():
                self.logger.info("Milvus 서버에 성공적으로 연결되었습니다.")
            else:
                self.logger.error("Milvus 서버 연결에 실패했습니다.")
                self.milvus_client = None
                
        except Exception as e:
            self.logger.error(f"Milvus 초기화 오류: {e}")
            self.milvus_client = None

    def _load_model(self) -> Any:
        """Loads the embedding model."""
        model_to_load = self.embedding_model_name
        self.logger.info(f"Loading model: {model_to_load} on {self.device}")
        
        # 라이브러리가 모두 없는 경우 None 반환 (fallback 모드)
        if not SENTENCE_TRANSFORMERS_AVAILABLE and not TRANSFORMERS_AVAILABLE:
            self.logger.warning("Neither sentence_transformers nor transformers available. Using fallback mode.")
            return None
            
        try:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                try:
                    model = SentenceTransformer(model_to_load, device=self.device)
                    self.logger.info("Loaded with SentenceTransformers.")
                    return model
                except Exception as e_st:
                    self.logger.warning(f"ST load failed: {e_st}. Trying Transformers.")
            if TRANSFORMERS_AVAILABLE:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_to_load)
                    hf_model = AutoModel.from_pretrained(model_to_load).to(self.device)
                    self.logger.info("Loaded with HuggingFace Transformers.")
                    return {"tokenizer": tokenizer, "model": hf_model}
                except Exception as e_hf:
                    self.logger.error(f"HF load failed: {e_hf}")
            self.logger.error("Could not load model with any library.")
            return None
        except Exception as e:
            self.logger.error(f"General error loading model: {e}")
            return None

    def _mean_pooling(self, model_output: Any, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean Pooling for HuggingFace outputs."""
        token_embeddings = model_output[0] if isinstance(model_output, tuple) else model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def embed_text(self, text: str) -> np.ndarray:
        """Embeds a single string."""
        default_dim = self.embedding_dimension or 768
        if self.model is None: return np.zeros(default_dim)
        text_cleaned = re.sub(r'\s+', ' ', text).strip()
        if not text_cleaned: return np.zeros(default_dim)
        try:
            if isinstance(self.model, SentenceTransformer):
                return self.model.encode(text_cleaned, convert_to_numpy=True, normalize_embeddings=self.normalize_embeddings)
            elif isinstance(self.model, dict):
                inputs = self.model["tokenizer"](text_cleaned, return_tensors="pt", padding=True, truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad(): outputs = self.model["model"](**inputs)
                embedding = self._mean_pooling(outputs, inputs["attention_mask"])
                if self.normalize_embeddings: embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
                return embedding[0].cpu().numpy()
            self.logger.error(f"Unsupported model type: {type(self.model)}.")
            return np.zeros(default_dim)
        except Exception as e:
            self.logger.error(f"Error embedding text '{text_cleaned[:30]}...': {e}")
            return np.zeros(default_dim)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embeds a list of strings."""
        default_dim = self.embedding_dimension or 768
        if self.model is None: return np.zeros((len(texts), default_dim))
        cleaned_texts = [re.sub(r'\s+', ' ', text).strip() for text in texts if text and text.strip()]
        if not cleaned_texts: return np.array([])

        if self.device == "cuda": torch.cuda.empty_cache()
        try:
            if isinstance(self.model, SentenceTransformer):
                self.logger.info(f"Embedding {len(cleaned_texts)} texts (ST, batch: {self.batch_size})")
                return self.model.encode(cleaned_texts, batch_size=self.batch_size, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=self.normalize_embeddings)
            elif isinstance(self.model, dict):
                self.logger.info(f"Embedding {len(cleaned_texts)} texts (HF, batch: {self.batch_size})")
                all_embs = []
                for i in tqdm(range(0, len(cleaned_texts), self.batch_size), desc="Embedding batches"):
                    batch = cleaned_texts[i:i + self.batch_size]
                    inputs = self.model["tokenizer"](batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    with torch.no_grad(): outputs = self.model["model"](**inputs)
                    embs = self._mean_pooling(outputs, inputs["attention_mask"])
                    if self.normalize_embeddings: embs = torch.nn.functional.normalize(embs, p=2, dim=1)
                    all_embs.append(embs.cpu().numpy())
                return np.vstack(all_embs) if all_embs else np.array([])
            self.logger.error(f"Unsupported model type: {type(self.model)}.")
            return np.zeros((len(cleaned_texts), default_dim))
        except Exception as e:
            self.logger.error(f"Error batch embedding: {e}")
            return np.zeros((len(cleaned_texts), default_dim))

    def embed_document(self, document: Document) -> Document:
        """Generates embeddings for all valid chunks in a document."""
        if not document or not document.chunks: return document
        self.logger.info(f"Embedding doc: {document.source or document.doc_id}")
        
        # Process child chunks first (main chunks)
        valid_chunks_indices = [(i, chunk.text) for i, chunk in enumerate(document.chunks) if chunk.text and chunk.text.strip()]
        if valid_chunks_indices:
            indices, texts = zip(*valid_chunks_indices)
            embeddings = self.embed_texts(list(texts))
            if len(embeddings) != len(indices):
                self.logger.error("Embedding count mismatch in embed_document.")
                return document
            for i, original_index in enumerate(indices):
                if not hasattr(document.chunks[original_index], 'metadata') or document.chunks[original_index].metadata is None:
                    document.chunks[original_index].metadata = {}
                document.chunks[original_index].metadata["embedding"] = embeddings[i].tolist()
        
        # Process parent chunks if they exist
        if document.parent_chunks and len(document.parent_chunks) > 0:
            self.logger.info(f"Embedding {len(document.parent_chunks)} parent chunks for doc: {document.source or document.doc_id}")
            valid_parent_indices = [(i, chunk.text) for i, chunk in enumerate(document.parent_chunks) if chunk.text and chunk.text.strip()]
            if valid_parent_indices:
                parent_indices, parent_texts = zip(*valid_parent_indices)
                parent_embeddings = self.embed_texts(list(parent_texts))
                if len(parent_embeddings) != len(parent_indices):
                    self.logger.error("Embedding count mismatch in embed_document for parent chunks.")
                    return document
                for i, original_index in enumerate(parent_indices):
                    if not hasattr(document.parent_chunks[original_index], 'metadata') or document.parent_chunks[original_index].metadata is None:
                        document.parent_chunks[original_index].metadata = {}
                    document.parent_chunks[original_index].metadata["embedding"] = parent_embeddings[i].tolist()
        
        return document

    def embed_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """Generates embeddings for a list of valid chunks."""
        if not chunks: return []
        self.logger.info(f"Embedding {len(chunks)} chunks.")
        valid_chunks_indices = [(i, chunk.text) for i, chunk in enumerate(chunks) if chunk.text and chunk.text.strip()]
        if not valid_chunks_indices: return chunks
        indices, texts = zip(*valid_chunks_indices)
        embeddings = self.embed_texts(list(texts))
        if len(embeddings) != len(indices):
            self.logger.error("Embedding count mismatch in embed_chunks.")
            return chunks
        for i, original_index in enumerate(indices):
            if not hasattr(chunks[original_index], 'metadata') or chunks[original_index].metadata is None:
                 chunks[original_index].metadata = {}
            chunks[original_index].metadata["embedding"] = embeddings[i].tolist()
        return chunks

    def store_document_in_db(self, document: Document, collection_name_override: Optional[str] = None) -> bool:
        """
        Store a document in the Milvus vector database.
        
        Args:
            document: Document to store.
            collection_name_override: Override the collection name. If None, use document title or ID.
            
        Returns:
            True if successful, False otherwise.
        """
        if not self.milvus_client:
            self.logger.error("Milvus 클라이언트가 초기화되지 않았습니다.")
            return False
        
        # 수정된 부분: 임베딩 확인 방식 변경
        # embeddings 속성 대신 chunks의 metadata 확인
        has_embeddings = all(
            hasattr(chunk, 'metadata') and 
            chunk.metadata and 
            'embedding' in chunk.metadata 
            for chunk in document.chunks
        ) if document.chunks else False
        
        if not has_embeddings:
            self.logger.warning("Document has no embeddings, embedding before storing")
            document = self.embed_document(document)
            # 다시 확인
            has_embeddings = all(
                hasattr(chunk, 'metadata') and 
                chunk.metadata and 
                'embedding' in chunk.metadata 
                for chunk in document.chunks
            ) if document.chunks else False
            
            if not has_embeddings:
                self.logger.error("Failed to generate embeddings for document")
                return False
        
        # 수정된 부분: 컬렉션 이름 미리 정제
        if collection_name_override:
            collection_name = self.milvus_client.sanitize_name(collection_name_override)
        else:
            if document.title:
                collection_name = self.milvus_client.sanitize_name(document.title)
            else:
                collection_name = self.milvus_client.sanitize_name(document.doc_id)
        
        self.logger.info(f"Using sanitized collection name: {collection_name}")
        
        return self._store_document_in_milvus(document, collection_name)
            
    def _store_document_in_milvus(self, document: Document, collection_name: str) -> bool:
        """Milvus 벡터 데이터베이스에 문서 저장 (자식 및 부모 청크 모두)"""
        self.logger.info(f"Milvus 컬렉션에 문서 저장: {collection_name}")
        
        # 이전에 생성된 컬렉션 삭제 (스키마 변경을 위해)
        if self.milvus_client.has_collection(collection_name):
            self.logger.info(f"기존 컬렉션 '{collection_name}'을 삭제하고 다시 생성합니다.")
            self.milvus_client.drop_collection(collection_name)
        
        # 컬렉션 생성
        if not self.milvus_client.has_collection(collection_name):
            self.logger.info(f"컬렉션 '{collection_name}'이 존재하지 않아 생성합니다.")
            if not self.milvus_client.create_collection(collection_name, self.embedding_dimension):
                self.logger.error(f"컬렉션 '{collection_name}' 생성 실패")
                return False
        
        # 파티션 이름도 정제
        partition_raw = document.title if document.title else document.doc_id
        partition = self.milvus_client.sanitize_name(partition_raw)
        
        if not self.milvus_client.has_partition(collection_name, partition):
            self.logger.info(f"파티션 '{partition}'이 존재하지 않아 생성합니다.")
            if not self.milvus_client.create_partition(collection_name, partition):
                self.logger.warning(f"파티션 '{partition}' 생성 실패. 기본 파티션 사용")
                partition = None
        
        # 메인 (자식) 청크 저장
        child_success = self._store_chunks_in_milvus(document.chunks, collection_name, partition, "child")
        
        # 부모 청크가 있는 경우 별도 저장
        parent_success = True
        if document.parent_chunks and len(document.parent_chunks) > 0:
            # 부모 청크 파티션 생성 (자식과 구분하기 위해 접미사 추가)
            parent_partition_name = f"{partition}_parent" if partition else "parent_chunks"
            
            if not self.milvus_client.has_partition(collection_name, parent_partition_name):
                self.logger.info(f"부모 청크용 파티션 '{parent_partition_name}'을 생성합니다.")
                if not self.milvus_client.create_partition(collection_name, parent_partition_name):
                    self.logger.warning(f"부모 청크 파티션 '{parent_partition_name}' 생성 실패. 기본 파티션 사용")
                    parent_partition_name = None
            
            parent_success = self._store_chunks_in_milvus(document.parent_chunks, collection_name, parent_partition_name, "parent")
            self.logger.info(f"부모 청크 {len(document.parent_chunks)}개 저장 완료 ({parent_success})")
        
        return child_success and parent_success
        
    def _store_chunks_in_milvus(self, chunks: List[TextChunk], collection_name: str, partition: Optional[str] = None, chunk_type: str = "child") -> bool:
        """Milvus에 청크 저장"""
        if not chunks or not self.milvus_client:
            return False
            
        self.logger.info(f"Milvus 컬렉션 '{collection_name}'에 {len(chunks)} {chunk_type} 청크 저장 (파티션: {partition or '_default'})")
        
        ids = []
        vectors = []
        metadata_list = []
        
        for chunk in chunks:
            if not hasattr(chunk, 'metadata') or chunk.metadata is None or \
               "embedding" not in chunk.metadata or not isinstance(chunk.metadata["embedding"], list):
                self.logger.warning(f"청크 {getattr(chunk, 'chunk_id', 'N/A')}에 유효한 임베딩이 없습니다. 건너뜁니다.")
                continue
                
            chunk_id = str(getattr(chunk, 'chunk_id', uuid.uuid4().hex))
            ids.append(chunk_id)
            vectors.append(chunk.metadata["embedding"])
            
            # chunk는 parser.py의 TextChunk 객체여야 합니다.
            meta_for_milvus = {
                "text": getattr(chunk, 'text', ""),
                "chunk_type": chunk_type,  # 청크 타입 저장 (parent, child)
                "doc_id": getattr(chunk, 'doc_id', ""),
                "parent_chunk_id": getattr(chunk, 'parent_chunk_id', None) or "",  # None이면 빈 문자열
                "original_chunk_type": getattr(chunk, 'chunk_type', "unknown"),
                "page_num": getattr(chunk, 'page_num', -1),  # -1은 페이지 번호가 없음을 의미
                
                # TextChunk.metadata에서 가져올 추가 필드들
                "article_title": getattr(chunk.metadata, 'get', lambda k, d: d)("article_title", ""),
                "clause_marker": getattr(chunk.metadata, 'get', lambda k, d: d)("clause_marker", ""),
                "item_marker": getattr(chunk.metadata, 'get', lambda k, d: d)("item_marker", ""),
                "source_file": getattr(chunk.metadata, 'get', lambda k, d: d)("source_file", 
                               getattr(chunk, 'source', ""))
            }
            
            # 동적 필드 지원을 위한 메타데이터 처리
            if hasattr(chunk, 'metadata') and isinstance(chunk.metadata, dict):
                for k, v in chunk.metadata.items():
                    if k != "embedding" and k not in meta_for_milvus:  # 이미 추가된 필드는 중복 추가 방지
                        # 복잡한 구조(dict, list)는 JSON 문자열로 변환
                        if isinstance(v, (dict, list)):
                            try:
                                meta_for_milvus[k] = json.dumps(v, ensure_ascii=False)
                            except (TypeError, OverflowError):
                                meta_for_milvus[k] = str(v)  # 직렬화 실패 시 문자열 변환
                        else:
                            meta_for_milvus[k] = v
                        
            metadata_list.append(meta_for_milvus)
        
        if not ids:
            self.logger.warning(f"저장할 유효한 {chunk_type} 청크가 없습니다.")
            return False
            
        try:
            # 수정된 부분: partition 매개변수 제거
            upserted_count = self.milvus_client.insert(
                collection_name=collection_name,
                ids=ids,
                vectors=vectors,
                metadata=metadata_list,
                # partition 매개변수 제거
                force_flush=True
            )
            
            self.logger.info(f"Milvus에 {upserted_count}개 {chunk_type} 벡터 저장 성공 (총 시도: {len(ids)})")
            return upserted_count > 0
            
        except Exception as e:
            self.logger.error(f"Milvus에 {chunk_type} 청크 저장 중 오류 발생: {e}")
            return False
    
    def embed_directory(self, input_dir_str: str, output_dir_str: Optional[str] = None, store_in_db: bool = True) -> List[Document]:
        """Processes all JSON chunk files in a directory."""
        input_dir = Path(input_dir_str)
        if not input_dir.is_dir(): self.logger.error(f"Not a dir: {input_dir_str}"); return []
        self.logger.info(f"Processing dir: {input_dir_str}")
        output_dir = Path(output_dir_str) if output_dir_str else None
        if output_dir: output_dir.mkdir(parents=True, exist_ok=True)

        chunk_files = [f for f in input_dir.iterdir() if f.is_file() and f.name.endswith('_chunks.json')]
        self.logger.info(f"Found {len(chunk_files)} chunk files.")
        processed_docs = []
        for chunk_f_path in chunk_files:
            out_f_path_str = None
            if output_dir: out_f_path_str = str(output_dir / chunk_f_path.name.replace('_chunks.json', '_embeddings.json'))
            doc = self.embed_file(str(chunk_f_path), out_f_path_str, store_in_db)
            if doc: processed_docs.append(doc)
        return processed_docs

    def embed_file(self, input_file_str: str, output_file_str: Optional[str] = None, store_in_db: bool = True) -> Optional[Document]:
        """Processes a single JSON chunk file."""
        input_file = Path(input_file_str)
        if not input_file.is_file(): self.logger.error(f"Not a file: {input_file_str}"); return None
        self.logger.info(f"Processing file: {input_file_str}")
        try:
            with open(input_file, 'r', encoding='utf-8') as f: doc_dict = json.load(f)
            doc = Document.from_dict(doc_dict)
            self.logger.info(f"Loaded {len(doc.chunks)} chunks from {input_file_str}")
            doc_w_embeds = self.embed_document(doc)

            if output_file_str:
                out_file = Path(output_file_str)
                out_file.parent.mkdir(parents=True, exist_ok=True)
                with open(out_file, 'w', encoding='utf-8') as f: json.dump(doc_w_embeds.to_dict(), f, ensure_ascii=False, indent=2)
                self.logger.info(f"Saved embeddings to {output_file_str}")
            
            if store_in_db:
                coll_name = input_file.stem.replace('_chunks', '')
                self.store_document_in_db(doc_w_embeds, coll_name)
            return doc_w_embeds
        except FileNotFoundError: self.logger.error(f"File not found (FNFE): {input_file_str}"); return None
        except json.JSONDecodeError: self.logger.error(f"JSON decode error: {input_file_str}"); return None
        except Exception as e: self.logger.error(f"Error processing file {input_file_str}: {e}", exc_info=True); return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed document chunks and store them.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input JSON chunk file or directory.")
    parser.add_argument("--output_path", type=str, help="Path to save output JSON with embeddings, or directory.")
    parser.add_argument("--model_name", type=str, help="Embedding model name (overrides config).")
    parser.add_argument("--store_in_db", action=argparse.BooleanOptionalAction, default=True, help="Store in DB (default: True). Use --no-store_in_db to disable.")

    args = parser.parse_args()
    config_obj = Config()
    if args.model_name:
        logger.info(f"Overriding model with: {args.model_name}")
        if 'embedding' not in config_obj.config_data: config_obj.config_data['embedding'] = {}
        config_obj.config_data['embedding']['model'] = args.model_name

    embedder_instance = DocumentEmbedder(config_obj)
    logger.info(f"Store in DB setting: {args.store_in_db}")
    input_path = Path(args.input_path)

    if input_path.is_dir():
        logger.info(f"Input is directory: {args.input_path}")
        out_dir_str = args.output_path
        if out_dir_str: Path(out_dir_str).mkdir(parents=True, exist_ok=True); logger.info(f"Output dir: {out_dir_str}")
        else: logger.info("No output dir for directory input.")
        docs = embedder_instance.embed_directory(str(input_path), out_dir_str, args.store_in_db)
        print(f"Processed {len(docs)} docs from {args.input_path}")

    elif input_path.is_file():
        logger.info(f"Input is file: {args.input_path}")
        out_file_str = None
        if args.output_path:
            out_path = Path(args.output_path)
            if not out_path.suffix or out_path.is_dir():
                out_path.mkdir(parents=True, exist_ok=True)
                name = input_path.name.replace('_chunks.json', '_embeddings.json') if '_chunks.json' in input_path.name else f"{input_path.stem}_embeddings.json"
                out_file_str = str(out_path / name)
                logger.info(f"Output dir specified. Output file: {out_file_str}")
            else:
                out_file_str = str(out_path)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Output file path: {out_file_str}")
        else: logger.info("No output path for file input.")
        
        doc = embedder_instance.embed_file(str(input_path), out_file_str, args.store_in_db)
        if doc: print(f"Successfully processed file: {args.input_path}")
        else: print(f"Failed to process file: {args.input_path}")
    else:
        logger.error(f"Invalid input path: {args.input_path}")
        parser.print_help()