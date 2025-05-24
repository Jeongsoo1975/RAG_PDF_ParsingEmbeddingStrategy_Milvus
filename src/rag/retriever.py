#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Document Retriever module for searching and retrieving relevant documents.
Implements vector similarity search and hybrid retrieval strategies using Milvus.
Supports Small-to-Big retrieval strategy for enhanced context.
"""

import os
import numpy as np
import json
import re
import time
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path # Path 추가

# Milvus 임포트
try:
    import pymilvus
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    logging.warning("pymilvus not available, Milvus vector database search will not work")

# VectorDB 클라이언트 임포트
try:
    from src.vectordb import MilvusClient # 가정: MilvusClient는 config를 인자로 받음
    VECTORDB_CLIENT_AVAILABLE = True
except ImportError:
    VECTORDB_CLIENT_AVAILABLE = False
    logging.warning("vectordb client (MilvusClient) not available. MilvusClient will not be used.")

from src.utils.config import Config
from src.utils.logger import get_logger
from src.rag.embedder import DocumentEmbedder
# Re-ranker 임포트
try:
    from src.rag.reranker import DocumentReranker
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    logging.warning("DocumentReranker not available, re-ranking will be disabled")
# parser.py의 Document, TextChunk 클래스를 임포트하여 부모 청크를 로드할 때 사용 가능하도록 준비
try:
    from src.rag.parser import Document as ParsedDocument, TextChunk as ParsedTextChunk
    PARSER_CLASSES_AVAILABLE = True
except ImportError:
    PARSER_CLASSES_AVAILABLE = False
    logging.warning("parser.py Document/TextChunk classes not found. Parent chunk loading from JSON will be limited.")


logger = get_logger("retriever")

class DocumentRetriever:
    """Class to search and retrieve documents based on queries using Milvus"""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the document retriever with configuration."""
        self.config = config or Config()
        self.logger = logger
        
        self.retrieval_config = self.config.get_section('retrieval') if hasattr(self.config, 'get_section') else self.config.retrieval if hasattr(self.config, 'retrieval') else {}
        
        self.top_k = self.retrieval_config.get('top_k', 5)
        self.similarity_threshold = self.retrieval_config.get('similarity_threshold', 0.5) # 디버깅을 위해 기본 임계값을 낮춤 (예: 0.5 또는 더 낮게)
        self.similarity_metric = self.retrieval_config.get('similarity_metric', 'COSINE') # MilvusClient와 일치 필요
        self.hybrid_search_enabled_by_default = self.retrieval_config.get('hybrid_search', False) # 기본적으로 hybrid는 False로 시작 가능
        
        # collections는 Milvus에서 직접 읽어오므로 config에서는 제거하거나 기본값으로만 활용
        self.db_collections = [] 
        self.config_collection_name = self.config.milvus.get('collection_name', 'insurance-embeddings') if hasattr(self.config, 'milvus') else 'insurance-embeddings'
        
        
        self.offline_mode = self.retrieval_config.get('offline_mode', False)
        
        self.embedder = DocumentEmbedder(self.config) # config 전달
        self.milvus_client = None
        
        # Re-ranker 초기화
        self.reranker = None
        if RERANKER_AVAILABLE and self.retrieval_config.get('reranking', True):
            try:
                self.reranker = DocumentReranker(self.config)
                logger.info("DocumentReranker initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize DocumentReranker: {e}")
                self.reranker = None
        else:
            logger.info("Re-ranking disabled or DocumentReranker not available")
        
        # Small-to-Big: 부모 청크가 저장된 JSON 파일 경로 (parser.py의 output_dir과 연관)
        # 이 부분은 config에서 읽어오거나, 문서 ID 기반으로 동적으로 경로를 구성해야 함
        self.parent_chunk_data_dir = self.retrieval_config.get('parent_chunk_data_dir', './data/parsed_output') # 예시 경로
        
        if not self.offline_mode:
            if MILVUS_AVAILABLE and VECTORDB_CLIENT_AVAILABLE:
                self._init_milvus()
                if self.milvus_client and self.milvus_client.is_connected():
                    try:
                        self.db_collections = self.milvus_client.list_collections()
                        logger.info(f"Available Milvus collections: {self.db_collections}")
                        if not self.db_collections:
                            logger.warning("No collections found in Milvus.")
                    except Exception as e:
                        logger.error(f"Failed to list Milvus collections: {e}")
                        self.db_collections = []
                else:
                    logger.warning("Milvus client not connected. Retriever might operate in a limited or offline capacity.")
                    self.offline_mode = True # 연결 실패 시 오프라인으로 강제 전환 가능
            else:
                logger.warning("Milvus or MilvusClient not available. Switching to offline mode.")
                self.offline_mode = True
        
        if self.offline_mode:
            logger.info("Retriever running in OFFLINE mode.")
            self.db_collections = ["offline_dummy_collection"] # 오프라인 모드용 기본 컬렉션

    @property
    def collections(self) -> List[str]: # main.py에서 retriever.collections를 사용하므로 프로퍼티로 유지
        if self.offline_mode:
            return ["offline_dummy_collection"]
        if self.milvus_client and self.milvus_client.is_connected():
            try:
                # 필요시 캐싱 로직 추가 가능
                self.db_collections = self.milvus_client.list_collections()
            except Exception as e:
                logger.error(f"Error refreshing Milvus collections: {e}")
                return self.db_collections # 이전 값 반환 또는 빈 리스트
        return self.db_collections


    def _init_milvus(self):
        """Initialize Milvus client and find matching collection."""
        try:
            self.milvus_client = MilvusClient(self.config) # config 전달
            if self.milvus_client.is_connected():
                logger.info("Successfully connected to Milvus server for retriever.")
                
                # 사용 가능한 콜렉션 목록 가져오기
                self.db_collections = self.milvus_client.list_collections()
                logger.info(f"Available Milvus collections: {self.db_collections}")
                
                # config에 설정된 콜렉션 이름과 일치하는 콜렉션이 있는지 확인
                if self.config_collection_name in self.db_collections:
                    logger.info(f"Using collection '{self.config_collection_name}' as configured in config.")
                else:
                    # 기본 콜렉션 이름이 없을 경우 자동으로 다른 콜렉션 찾기
                    if not self.db_collections:
                        logger.warning("No collections found in Milvus.")
                    else:
                        logger.warning(f"Collection '{self.config_collection_name}' not found. Trying to use another available collection.")
                        # CI 관련 콜렉션 우선 사용
                        ci_collections = [c for c in self.db_collections if 'ci' in c.lower()]
                        if ci_collections:
                            logger.info(f"Found CI-related collections: {ci_collections}. Using '{ci_collections[0]}' as default.")
                            # 첫 번째 CI 관련 콜렉션을 설정
                            self.config_collection_name = ci_collections[0]
            else:
                logger.error("Failed to connect to Milvus server (retriever init).")
                self.milvus_client = None
                self.offline_mode = True
        except Exception as e:
            logger.error(f"Error initializing Milvus for retriever: {e}", exc_info=True)
            self.milvus_client = None
            self.offline_mode = True
    
    def _optimize_query(self, query: str, enable_optimization: bool = True) -> str:
        """
        쿼리 최적화 (디버깅을 위해 이 부분을 단순화하거나 비활성화 가능)
        """
        if not enable_optimization:
            logger.info(f"Query optimization DISABLED. Using raw query: '{query}'")
            return query.strip()

        original_query = query
        optimized = query.strip()
        
        # 단계 1: 불필요한 조사, 어미, 특수문자 제거 (매우 보수적으로)
        # "제1보험기간이란 무엇인가요?" -> "제1보험기간"
        # "OOO에 대해 알려줘" -> "OOO"
        # 주의: 과도하게 제거하면 의미가 손실될 수 있음
        optimized = re.sub(r'(이란|이란 무엇인가요|이란 뭔가요|이란 뭐죠|은 무엇인가요|는 무엇인가요|란|은|는|이|가|에 대해|에 대해서|대하여|알려줘|궁금합니다|설명해주세요)\s*\?*$', '', optimized).strip()
        optimized = re.sub(r'\?$', '', optimized).strip() # 마지막 물음표 제거

        # "정의" 추가 로직은 매우 신중해야 함. 원본 쿼리의 의도를 바꿀 수 있음.
        # 여기서는 "정의"를 자동으로 추가하지 않도록 주석 처리 또는 제거.
        # if re.search(r'[가-힣]+', optimized) and not any(word in optimized for word in ["정의", "개념", "뜻", "의미"]) \
        #    and optimized and not optimized.endswith("정의"):
        #     if re.search(r'[가-힣0-9]$', optimized):
        #         logger.debug(f"Potentially adding ' 정의' to '{optimized}'")
        #         # optimized += " 정의" # <<-- 이 부분은 검색 실패의 주요 원인이 될 수 있음
        
        logger.info(f"Query optimization (enable_optimization={enable_optimization}): '{original_query}' -> '{optimized}'")
        return optimized

    def _extract_keywords(self, query: str, min_keyword_length: int = 2) -> List[str]:
        # 한국어 형태소 분석기(예: Okt, Mecab - konlpy)를 사용하는 것이 더 좋음
        # 여기서는 간단한 공백 기반 및 불용어 제거
        # (이전 코드와 유사하게 유지)
        stop_words = set([
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
            "in", "on", "at", "by", "for", "with", "about", "to", "from", "of", "and", "or", "but", "if", "as", "what", "when", "where", "how", "why", "who", "which",
            "이", "가", "은", "는", "을", "를", "의", "에", "에서", "에게", "께", "한테", "더러", "와", "과", "랑", "이랑", "고", "하고",
            "면", "이면", "며", "이며", "란", "이란", "것", "무엇", "뭔가", "뭐", "언제", "어디", "누가", "어떻게", "왜", "좀", "등", "및"
        ])
        # 연속된 한글, 영어, 숫자 단어 추출
        words = re.findall(r'[a-zA-Z0-9가-힣]+', query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) >= min_keyword_length]
        
        # 원본 쿼리의 중요한 용어가 포함되도록 (예: "제1보험기간")
        # 원본 쿼리에서 명사형 단어들을 추가하는 로직이 필요할 수 있음 (형태소 분석기 활용)
        # 임시로, 공백 제거된 쿼리 자체도 키워드로 간주 (매우 단순한 접근)
        processed_query_terms = query.replace(" ", "").lower()
        if processed_query_terms not in keywords and len(processed_query_terms) >= min_keyword_length :
             if query.strip().lower() not in stop_words : # 전체 쿼리가 불용어가 아니라면
                keywords.append(query.strip().lower()) # 원본 형태도 추가 시도

        unique_keywords = sorted(list(set(keywords)), key=keywords.index) # 순서 유지하며 중복 제거
        logger.debug(f"Extracted keywords from '{query}': {unique_keywords}")
        return unique_keywords

    def retrieve(self, 
                query: str, 
                top_k: Optional[int] = None, 
                threshold: Optional[float] = None,
                target_collections: Optional[List[str]] = None, # 명확한 이름으로 변경
                use_parent_chunks: bool = False, # Small-to-Big 플래그
                enable_query_optimization: bool = False, # 디버깅: 기본적으로 False로 설정하여 테스트
                force_filter_expr: Optional[str] = "default" 
                ) -> List[Dict[str, Any]]:
        
        _top_k = top_k if top_k is not None else self.top_k
        _threshold = threshold if threshold is not None else self.similarity_threshold
        
        # 검색 대상 콜렉션 결정
        _collections_to_search = target_collections
        if not _collections_to_search: # 인자로 안넘어오면
            if self.config_collection_name in self.db_collections:
                # config에서 설정한 콜렉션이 있을 경우 그것을 사용
                _collections_to_search = [self.config_collection_name]
                logger.info(f"Using collection from config: {self.config_collection_name}")
            elif self.collections: # CI 관련 콜렉션 등 자동 감지된 콜렉션이 있으면 그것을 사용
                _collections_to_search = self.collections
                logger.info(f"Using automatically detected collections: {self.collections}")
            else: # 그것도 없으면 경고 후 빈 리스트 (오프라인 모드는 아래에서 처리)
                logger.warning("No collections specified or configured for search.")
                _collections_to_search = []
        
        
        if self.offline_mode or not self.milvus_client or not self.milvus_client.is_connected():
            logger.warning("Milvus client not available/connected or in offline mode. Using offline_retrieve.")
            return self.offline_retrieve(query, _top_k)

        if not _collections_to_search:
            logger.error("Online mode but no collections to search. Please check Milvus and configuration.")
            return []
        
        logger.info(f"Retrieving documents for raw query: '{query}'")
        logger.info(f"Parameters: top_k={_top_k}, threshold={_threshold}, collections={_collections_to_search}, use_parent_chunks={use_parent_chunks}, query_opt={enable_query_optimization}, filter_override='{force_filter_expr}'")

        # 1. 쿼리 처리 (최적화 또는 원본 사용)
        processed_query = self._optimize_query(query, enable_optimization=enable_query_optimization)
        query_embedding_np = self.embedder.embed_text(processed_query)
        
        if query_embedding_np is None or query_embedding_np.size == 0:
            logger.error(f"Failed to generate embedding for query: '{processed_query}'")
            return []
        query_embedding_list = query_embedding_np.tolist()

        # 2. 필터 표현식 결정
        final_filter_expr = None
        if force_filter_expr == 'default':
            # Small-to-Big 전략: 기본적으로 자식 청크(item, item_sub_chunk 등)만 검색하도록 필터링
            # 이 필터는 parser.py에서 TextChunk.chunk_type에 설정한 값과 일치해야 함
            # milvus_client.py의 스키마에도 chunk_type이 VARCHAR로 정의되어 있어야 함
            # 그리고 embedder.py가 이 값을 Milvus에 올바르게 저장해야 함
            small_chunk_types = self.config.retrieval.get('small_chunk_types', ["item", "item_sub_chunk", "csv_row", "text_block"])
            if small_chunk_types: # 설정에 정의된 경우에만 필터 적용
                 filter_parts = [f'chunk_type == "{ct}"' for ct in small_chunk_types]
                 final_filter_expr = " or ".join(filter_parts)
                 # 예: 'chunk_type == "item" or chunk_type == "item_sub_chunk"'
        elif force_filter_expr: # 'default'도 아니고 None도 아닌 경우, 주어진 필터 사용
            final_filter_expr = force_filter_expr
        # else: force_filter_expr is None, so final_filter_expr remains None

        logger.info(f"Using Milvus filter expression: {final_filter_expr if final_filter_expr else 'None'}")
        
        # 3. Milvus에서 검색 실행
        child_results_raw = []
        for collection_name in _collections_to_search:
            # Sanitize collection name before use if your MilvusClient doesn't do it internally for every call
            if hasattr(self.milvus_client, 'sanitize_name'):
                safe_collection_name = self.milvus_client.sanitize_name(collection_name)
            else:
                safe_collection_name = collection_name
                
            try:
                has_collection = hasattr(self.milvus_client, 'has_collection') and self.milvus_client.has_collection(safe_collection_name)
            except Exception as e:
                logger.error(f"Error checking for collection '{safe_collection_name}': {e}")
                has_collection = False
                
            if not has_collection:
                logger.warning(f"Collection '{safe_collection_name}' (original: '{collection_name}') does not exist in Milvus. Skipping.")
                continue
            
            try:
                logger.debug(f"Searching in collection: '{safe_collection_name}' for query: '{processed_query}'")
                # MilvusClient.search의 partition_names 인자 사용 (list of strings)
                # output_fields도 MilvusClient.search에서 기본값을 갖거나 설정 가능하게
                search_output_fields = self.retrieval_config.get('milvus_output_fields', 
                    ["id", "text", "doc_id", "source", "page_num", "chunk_id", "parent_chunk_id", "chunk_type", "article_title", "item_marker"]
                )

                # 임계값 낮추기: 검색 시에는 임계값을 대폭 낮춰서 더 많은 결과를 가져올 수 있도록 함
                actual_search_threshold = max(0.01, _threshold * 0.5)  # 최소 0.01, 또는 지정 임계값의 50%
                
                results_from_collection = self.milvus_client.search(
                    collection_name=safe_collection_name,
                    query_vector=query_embedding_list,
                    top_k=_top_k * 4, # 부모 청크 대체를 위해 더 많이 가져옴 (필요시 조정)
                    filter_expr=final_filter_expr, 
                    partition_names=None, # 특정 파티션만 검색하려면 여기에 리스트 전달
                    output_fields=search_output_fields
                )
                
                for res_dict in results_from_collection:
                    # Milvus search 결과는 'score' 또는 'distance'를 포함할 수 있음. 'similarity'로 통일
                    similarity_score = res_dict.get('score', res_dict.get('distance', 0.0))
                    
                    # 실제 임계값 적용
                    if similarity_score >= actual_search_threshold:
                        child_results_raw.append({
                            "id": res_dict.get("id"),
                            "similarity": similarity_score,
                            "content": res_dict.get("text", ""),
                            "collection": safe_collection_name, # 정제된 컬렉션 이름 사용
                            "metadata": { # 필요한 모든 메타데이터를 여기에 포함
                                "doc_id": res_dict.get("doc_id"),
                                "source_file": res_dict.get("source"), # parser.py 출력과 일관성
                                "page_num": res_dict.get("page_num"),
                                "chunk_id": res_dict.get("chunk_id"),
                                "parent_chunk_id": res_dict.get("parent_chunk_id"),
                                "chunk_type": res_dict.get("chunk_type"),
                                "article_title": res_dict.get("article_title"),
                                "item_marker": res_dict.get("item_marker"),
                                # Milvus에서 반환된 다른 모든 메타데이터 필드도 추가 가능
                                **{k:v for k,v in res_dict.items() if k not in ["id", "score", "distance", "text", "vector"]}
                            }
                        })
            except Exception as e:
                logger.error(f"Error searching collection '{safe_collection_name}': {e}", exc_info=True)
        
        # 중복 제거 및 유사도 기준으로 정렬
        # ID를 기준으로 중복 제거 (다른 컬렉션에서 같은 ID가 나올 수 있으므로 (collection, id) 튜플 사용)
        unique_results_map = {}
        for r in child_results_raw:
            unique_key = (r['collection'], r['id'])
            if unique_key not in unique_results_map or r['similarity'] > unique_results_map[unique_key]['similarity']:
                unique_results_map[unique_key] = r
        
        sorted_child_results = sorted(list(unique_results_map.values()), key=lambda x: x["similarity"], reverse=True)
        
        # top_k 만큼만 선택 (이 단계에서 이미 임계값은 적용됨)
        retrieved_child_chunks = sorted_child_results[:_top_k]
        logger.info(f"Retrieved {len(retrieved_child_chunks)} child chunks after filtering and sorting.")

        # 4. Small-to-Big: 부모 청크 로딩 (use_parent_chunks가 True일 경우)
        processed_results = retrieved_child_chunks
        if use_parent_chunks and retrieved_child_chunks:
            logger.info("Attempting to fetch parent chunks (Small-to-Big)...")
            parent_results = self._get_parent_chunks_from_retrieved_children(retrieved_child_chunks)
            if parent_results:
                logger.info(f"Using {len(parent_results)} parent chunks (or original if parent not found).")
                processed_results = parent_results
            else: # 부모 청크를 가져오지 못했으면, 그냥 자식 청크 결과라도 반환
                logger.warning("Failed to fetch any parent chunks, using original child chunks.")
                processed_results = retrieved_child_chunks
        
        # 5. Re-ranking 적용 (최종 단계)
        if self.reranker and processed_results:
            logger.info("Applying re-ranking to search results...")
            try:
                reranked_results = self.reranker.rerank_results(query, processed_results, _top_k)
                logger.info(f"Re-ranking completed. Final result count: {len(reranked_results)}")
                return reranked_results
            except Exception as e:
                logger.error(f"Re-ranking failed: {e}. Returning original results.")
                return processed_results
        else:
            if not processed_results and query: # 초기 검색 결과가 아예 없는 경우
                logger.warning(f"No documents found for query '{processed_query}' with threshold {_threshold}. Trying offline retrieve as fallback.")
                return self.offline_retrieve(query, _top_k) # 원본 쿼리로 오프라인 재시도
            return processed_results

    def _load_parsed_document_for_parent_retrieval(self, doc_id: str, source_file_hint: Optional[str] = None) -> Optional[ParsedDocument]:
        """
        주어진 doc_id에 해당하는 파싱된 Document 객체를 로드합니다.
        parser.py가 생성한 JSON 파일에서 로드합니다.
        이 함수는 _get_parent_chunks_from_retrieved_children 내부에서 사용됩니다.
        """
        if not PARSER_CLASSES_AVAILABLE:
            logger.error("Parser_CLASSES_AVAILABLE is False. Cannot load parent documents from JSON.")
            return None

        # JSON 파일 경로 추론: self.parent_chunk_data_dir 사용
        # source_file_hint (예: "CI_20060401.pdf")를 기반으로 파일명 생성
        if source_file_hint:
            base_name = Path(source_file_hint).stem
            parsed_json_filename = f"{base_name}_parsed.json" # parser.py의 출력 파일명 규칙과 일치
            json_file_path = Path(self.parent_chunk_data_dir) / parsed_json_filename
            
            logger.debug(f"Attempting to load parent document from: {json_file_path}")
            if json_file_path.exists():
                try:
                    with open(json_file_path, 'r', encoding='utf-8') as f:
                        doc_dict = json.load(f)
                    # doc_id가 일치하는지 확인 (선택적이지만, 파일명 규칙이 깨졌을 경우 대비)
                    if doc_dict.get("doc_id") == doc_id:
                        return ParsedDocument.from_dict(doc_dict)
                    else:
                        logger.warning(f"Doc ID mismatch in {json_file_path}: expected {doc_id}, found {doc_dict.get('doc_id')}")
                except Exception as e:
                    logger.error(f"Error loading or parsing document JSON {json_file_path}: {e}")
            else:
                logger.warning(f"Parsed document JSON file not found: {json_file_path}")
        else:
            logger.warning(f"Cannot determine parent document JSON path without source_file_hint for doc_id: {doc_id}")
        
        # TODO: 만약 source_file_hint가 없고 doc_id만 있다면, self.parent_chunk_data_dir 전체를 스캔해야 할 수도 있음 (비효율적)
        # 이 경우, doc_id와 파일명을 매핑하는 별도의 메타데이터 저장소가 필요할 수 있음.
        return None


    def _get_parent_chunks_from_retrieved_children(self, child_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        검색된 자식 청크 목록을 기반으로 부모 청크의 내용을 가져옵니다.
        부모 청크 내용은 parser.py가 저장한 JSON 파일에서 로드합니다.
        """
        if not child_chunks:
            return []

        parent_chunk_results = []
        # doc_id별로 그룹화하여 파일 I/O 최소화
        docs_to_load_map: Dict[str, List[Dict[str,Any]]] = {}
        for child_chunk_dict in child_chunks:
            metadata = child_chunk_dict.get("metadata", {})
            doc_id = metadata.get("doc_id")
            parent_chunk_id = metadata.get("parent_chunk_id")
            
            if doc_id and parent_chunk_id:
                if doc_id not in docs_to_load_map:
                    docs_to_load_map[doc_id] = []
                docs_to_load_map[doc_id].append(child_chunk_dict)
            else: # 부모 정보가 없는 자식 청크는 그대로 결과에 포함
                logger.debug(f"Child chunk {child_chunk_dict.get('id')} lacks doc_id or parent_chunk_id. Using child chunk itself.")
                parent_chunk_results.append(child_chunk_dict)

        loaded_documents_cache: Dict[str, Optional[ParsedDocument]] = {}

        for doc_id, relevant_child_dicts in docs_to_load_map.items():
            parsed_doc = loaded_documents_cache.get(doc_id)
            if parsed_doc is None: # 캐시에 없으면 로드
                source_file_hint = relevant_child_dicts[0].get("metadata",{}).get("source_file")
                parsed_doc = self._load_parsed_document_for_parent_retrieval(doc_id, source_file_hint)
                loaded_documents_cache[doc_id] = parsed_doc # None일지라도 캐시에 저장하여 반복 로드 방지

            if parsed_doc and parsed_doc.parent_chunks:
                parent_id_to_chunk_map = {p_chunk.chunk_id: p_chunk for p_chunk in parsed_doc.parent_chunks}
                for child_dict in relevant_child_dicts:
                    parent_id = child_dict.get("metadata", {}).get("parent_chunk_id")
                    parent_chunk_obj = parent_id_to_chunk_map.get(parent_id)
                    if parent_chunk_obj:
                        # 부모 청크를 찾았으면, 부모 청크의 내용으로 대체
                        # 자식 청크의 유사도 점수와 ID는 유지하거나 메타데이터에 추가
                        parent_context_dict = {
                            "id": parent_chunk_obj.chunk_id, # 부모 청크의 ID
                            "similarity": child_dict["similarity"], # 자식 청크의 유사도 점수 사용
                            "content": parent_chunk_obj.text,
                            "collection": child_dict["collection"], # 원본 컬렉션 정보 유지
                            "metadata": {
                                **parent_chunk_obj.metadata, # 부모 청크의 메타데이터
                                "original_child_id": child_dict["id"],
                                "original_child_similarity": child_dict["similarity"],
                                "is_parent_context": True,
                                "chunk_type": parent_chunk_obj.chunk_type # 부모 청크의 타입 ("article")
                            }
                        }
                        parent_chunk_results.append(parent_context_dict)
                        logger.debug(f"Replaced child {child_dict['id']} with parent {parent_id} for doc {doc_id}")
                    else:
                        logger.warning(f"Parent chunk with ID '{parent_id}' not found in loaded document '{doc_id}'. Using child chunk {child_dict['id']}.")
                        parent_chunk_results.append(child_dict) # 부모 못찾으면 자식 그대로 사용
            else: # 파싱된 문서 로드 실패 또는 부모 청크 없음
                logger.warning(f"Could not load parsed document or no parent chunks for doc_id '{doc_id}'. Using original child chunks.")
                parent_chunk_results.extend(relevant_child_dicts)
        
        # 최종 결과를 자식 청크의 원래 유사도 점수 기준으로 다시 정렬
        parent_chunk_results.sort(key=lambda x: x.get("metadata", {}).get("original_child_similarity", x.get("similarity", 0.0)), reverse=True)
        # top_k 제한은 이미 retrieve 메소드에서 적용되었으므로 여기서는 불필요
        return parent_chunk_results


    def hybrid_retrieve(self, 
                       query: str, 
                       top_k: Optional[int] = None, 
                       threshold: Optional[float] = None,
                       target_collections: Optional[List[str]] = None, # 명확한 이름
                       use_parent_chunks: bool = False, # Small-to-Big 플래그
                       enable_query_optimization: bool = False, # 디버깅
                       force_filter_expr: Optional[str] = 'default'
                       ) -> List[Dict[str, Any]]:
        
        _top_k = top_k if top_k is not None else self.top_k
        _threshold = threshold if threshold is not None else self.similarity_threshold
        _collections_to_search = target_collections or self.collections

        logger.info(f"Performing hybrid retrieval for raw query: {query}")
        logger.info(f"Parameters: top_k={_top_k}, threshold={_threshold}, collections={_collections_to_search}, use_parent_chunks={use_parent_chunks}, query_opt={enable_query_optimization}, filter_override='{force_filter_expr}'")


        if not self.hybrid_search_enabled_by_default and not os.environ.get("FORCE_HYBRID_SEARCH"): # 환경변수 등으로 강제 활성화 가능
             logger.info("Hybrid search is not enabled by default configuration. Performing standard vector retrieval.")
             return self.retrieve(query, _top_k, _threshold, _collections_to_search, use_parent_chunks, enable_query_optimization, force_filter_expr)

        if self.offline_mode or not self.milvus_client or not self.milvus_client.is_connected():
            logger.warning("Milvus not available or in offline mode. Hybrid retrieval falls back to offline_retrieve.")
            return self.offline_retrieve(query, _top_k)
        
        # 1. 쿼리 처리 (최적화 또는 원본)
        # 벡터 검색용 쿼리와 키워드 검색용 쿼리를 다르게 가져갈 수도 있음
        vector_query = self._optimize_query(query, enable_optimization=enable_query_optimization)
        keyword_query = query # 키워드 검색은 원본 쿼리 또는 다른 형태의 최적화된 쿼리 사용 가능
        
        # 2. 벡터 검색 (Small-to-Big 로직은 retrieve 메소드 내에서 처리)
        # 벡터 검색은 더 많은 후보군을 가져오기 위해 top_k를 늘릴 수 있음 (예: top_k * 3)
        # threshold는 약간 낮춰서 더 많은 후보를 포함시킬 수 있음
        vector_search_top_k = _top_k * 3 
        vector_search_threshold = _threshold * 0.9 # 예: 원래 임계값의 90%
        
        logger.debug(f"Hybrid Step 1: Vector search with top_k={vector_search_top_k}, threshold={vector_search_threshold}")
        # retrieve 메소드는 use_parent_chunks 플래그에 따라 부모 또는 자식 청크를 반환할 것임.
        vector_results = self.retrieve(vector_query, 
                                       vector_search_top_k, 
                                       vector_search_threshold, 
                                       _collections_to_search, 
                                       use_parent_chunks, 
                                       enable_query_optimization=False, # retrieve 내에서 이미 처리하므로 중복 방지
                                       force_filter_expr=force_filter_expr)

        logger.debug(f"Hybrid Step 1: Got {len(vector_results)} results from vector search.")

        # 3. 키워드 추출 (키워드 검색용 쿼리 사용)
        keywords = self._extract_keywords(keyword_query)
        if not keywords:
            logger.warning(f"No keywords extracted from '{keyword_query}'. Hybrid search might behave like vector search.")
            # 키워드가 없으면 벡터 검색 결과만 사용 (이미 유사도 정렬됨)
            return sorted(vector_results, key=lambda x: x.get("similarity", 0.0), reverse=True)[:_top_k]

        # 4. BM25 또는 다른 키워드 검색 (Milvus는 직접 지원 안함, 외부 라이브러리 또는 직접 구현 필요)
        #    여기서는 vector_results를 키워드로 재점수화 하는 방식을 사용 (간단한 하이브리드)
        
        reranked_results_for_hybrid = []
        for result_dict in vector_results:
            content_lower = result_dict.get("content", "").lower()
            # content가 없을 경우 metadata의 다른 필드를 확인해볼 수도 있음
            if not content_lower and result_dict.get("metadata"):
                 # 예시: metadata의 title, item_marker 등도 검색 대상에 포함
                 meta_text = " ".join(str(v) for k,v in result_dict["metadata"].items() if isinstance(v, str)).lower()
                 content_lower += " " + meta_text


            keyword_match_score = 0
            matched_keywords = []
            for kw in keywords:
                if kw in content_lower:
                    keyword_match_score += 1 # 단순 매칭 카운트
                    matched_keywords.append(kw)
            
            # BM25 유사 점수 대신 간단한 가중치 부여 (0.0 ~ 1.0 범위)
            # 키워드 일치 비율에 기반한 점수. 모든 키워드가 매칭되면 1.0
            keyword_boost_score = (keyword_match_score / len(keywords)) if keywords else 0.0
            
            # 최종 점수: 벡터 유사도와 키워드 점수 결합 (가중치 조절 가능)
            # 예: 70% 벡터 유사도, 30% 키워드 부스트
            # 유사도 점수는 보통 0~1 (코사인) 또는 더 큰 값 (L2) 일 수 있음. 정규화 필요.
            # Milvus의 코사인 유사도는 score가 높을수록 유사함 (0~1)
            # L2 거리는 score가 낮을수록 유사함. 이 경우 변환 필요: 1 / (1 + distance) 또는 exp(-distance)
            
            vector_similarity = result_dict.get("similarity", 0.0)
            # 현재 MilvusClient search결과에서 score가 distance를 의미할 수 있으므로 확인 필요.
            # 만약 distance라면 (낮을수록 좋음), similarity = 1.0 / (1.0 + vector_similarity) 등으로 변환.
            # 여기서는 score가 높을수록 좋다고 가정 (코사인 유사도 등)
            
            # Hybrid score 계산 (단순 가중합 예시)
            # W_vector * S_vector + W_keyword * S_keyword
            # S_keyword를 0~1로 정규화 (keyword_boost_score 사용)
            hybrid_score = (0.7 * vector_similarity) + (0.3 * keyword_boost_score)
            
            # 결과에 추가 정보 저장
            result_dict["hybrid_score"] = hybrid_score
            result_dict["keyword_matches"] = matched_keywords # 디버깅용
            result_dict["original_vector_similarity"] = vector_similarity # 디버깅용
            reranked_results_for_hybrid.append(result_dict)

        # 5. 하이브리드 점수로 최종 정렬 및 top_k 선택
        reranked_results_for_hybrid.sort(key=lambda x: x["hybrid_score"], reverse=True)
        hybrid_results = reranked_results_for_hybrid[:_top_k]
        
        # 6. Re-ranking 적용 (최종 단계)
        if self.reranker and hybrid_results:
            logger.info("Applying re-ranking to hybrid search results...")
            try:
                # 하이브리드 점수를 유지하면서 re-ranking 점수도 추가
                final_reranked_results = self.reranker.rerank_results(query, hybrid_results, _top_k)
                
                # 하이브리드 정보를 메타데이터에 보존
                for result in final_reranked_results:
                    if 'hybrid_score' not in result:
                        # 기존 hybrid_results에서 hybrid_score 찾기
                        matching_hybrid = next((h for h in hybrid_results if h.get('id') == result.get('id')), None)
                        if matching_hybrid:
                            result['hybrid_score'] = matching_hybrid.get('hybrid_score', result.get('similarity', 0.0))
                            result['keyword_matches'] = matching_hybrid.get('keyword_matches', [])
                            result['original_vector_similarity'] = matching_hybrid.get('original_vector_similarity', result.get('similarity', 0.0))
                
                logger.info(f"Hybrid retrieval with re-ranking completed. Returning {len(final_reranked_results)} documents.")
                return final_reranked_results
            except Exception as e:
                logger.error(f"Re-ranking failed in hybrid search: {e}. Returning hybrid results without re-ranking.")
                return hybrid_results
        else:
            logger.info(f"Hybrid retrieval completed. Returning {len(hybrid_results)} documents.")
            return hybrid_results

    def offline_retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        logger.info(f"Performing OFFLINE retrieval for: '{query}' with top_k={top_k}")
        
        # 오프라인 모드의 샘플 데이터를 확장하고 개선
        dummy_docs_content = {
            "doc1_item1": "제1보험기간은 계약체결일로부터 80세 계약해당일 전일까지입니다. 이 기간 동안 주요 보장이 제공됩니다.",
            "doc1_item2": "제2보험기간은 80세 계약해당일부터 종신까지이며, 주로 연금 지급이나 건강 관련 특정 보장이 이루어집니다.",
            "doc1_article1_parent": "제1조(목적) 이 약관은 어쩌고 저쩌고... 제1보험기간은 계약체결일로부터 80세 계약해당일 전일까지입니다. 이 기간 동안 주요 보장이 제공됩니다. 제2보험기간은 80세 계약해당일부터 종신까지이며, 주로 연금 지급이나 건강 관련 특정 보장이 이루어집니다.",
            "doc2_general": "보험료 납입은 매월 자동이체로 이루어지며, 미납 시 계약이 해지될 수 있습니다.",
            "doc3_faq": "자주 묻는 질문: 제1보험기간이 끝나면 어떻게 되나요? 답변: 제2보험기간으로 넘어가거나 계약 조건에 따라 만기 보험금이 지급될 수 있습니다.",
            "doc4_ci_info": "CI보험금은 '중대한 질병', '중대한 수술', '중대한 화상' 등 약관에서 정한 특정 CI 발생 시 사망보험금의 일부(50% 또는 80%)를 선지급 받는 개념입니다. CI보험금 지급 후 계약은 종료되지 않으며, 차회 이후 보험료 납입이 면제됩니다.",
            "doc5_special_clause": "제18조에 명시된 주요 보험금의 종류는 사망보험금과 CI보험금입니다. CI보험금 중 '중대한 질병' 및 '중대한 수술'에 대한 보장개시일은 제1회 보험료를 받은 날부터 그 날을 포함하여 90일이 지난 날의 다음날입니다.",
            "doc6_prepayment_service": "선지급서비스특약은 피보험자의 여명이 6개월 이내로 판단될 경우 피보험자의 신청에 따라 주계약 사망보험금의 일부 또는 전부를 미리 지급받는 서비스입니다. 반면 CI보험금은 특정 질병/수술 발생 시 지급되는 보험금으로, 두 제도는 목적과 조건이 다릅니다.",
            "doc7_premium_exemption": "보험료 납입기간 중 피보험자가 장해분류표상 동일한 재해 또는 재해 이외의 동일한 원인으로 합산장해 지급률이 50% 이상 80% 미만인 장해상태가 되거나, 제18조 제1항 제2호의 CI보험금이 지급된 경우 차회 이후의 보험료 납입이 면제됩니다.",
            "doc8_special_contract": "주계약의 보험료 납입이 면제된 경우에는 무배당 정기특약의 보험료 납입도 면제하여 드립니다. 단, 보험료 납입이 면제된 이후 갱신시에도 보험료 납입을 면제하여 드립니다.",
            "doc9_prepaid_info": "선지급서비스특약은 의료법 제3조에서 정한 종합병원의 전문의 자격을 갖은 자가 실시한 진단결과 보험대상자(피보험자)의 잔여수명(여명)이 6개월 이내라고 판단한 경우에 보험대상자(피보험자)의 신청에 따라 주계약 사망보험금액의 일부 또는 전부를 선지급 사망보험금으로 보험대상자(피보험자)에게 지급합니다."
        }
        
        results = []
        keywords_in_query = self._extract_keywords(query)

        # 더 낮은 임계값 적용 (더 많은 결과 포함)
        min_score_threshold = 0.01

        for chunk_id, text_content in dummy_docs_content.items():
            # 향상된 점수 계산 로직
            score = 0.0
            
            # 1. 단순 키워드 매칭
            matched_keywords = []
            for kw in keywords_in_query:
                if kw.lower() in text_content.lower():
                    score += 0.15  # 키워드당 점수 부여
                    matched_keywords.append(kw)
            
            # 2. 특정 중요 키워드 가중치 부여
            important_terms = [
                ("CI보험금", 0.5), ("선지급", 0.5), ("선지급서비스특약", 0.6), 
                ("제18조", 0.4), ("제8조", 0.4), ("무배당 정기특약", 0.5),
                ("보험료 납입", 0.3), ("면제", 0.3), ("갱신", 0.3)
            ]
            
            for term, weight in important_terms:
                if term.lower() in text_content.lower() and term.lower() in query.lower():
                    score += weight
            
            # 3. 정확한 구문 일치에 가중치 부여
            phrases_in_query = self._extract_phrases(query)
            for phrase in phrases_in_query:
                if len(phrase) >= 4 and phrase.lower() in text_content.lower():
                    score += 0.3
            
            # 최대 점수 제한
            score = min(score, 1.0)

            # 임계값 이상인 경우 결과에 추가
            if score > min_score_threshold:
                results.append({
                    "id": chunk_id,
                    "similarity": score,
                    "content": text_content,
                    "collection": "offline_dummy_collection",
                    "metadata": {
                        "source_file": "dummy_document.pdf" if "doc1" in chunk_id else "generic_info.txt",
                        "page_num": 1,
                        "chunk_id": chunk_id,
                        "parent_chunk_id": "doc1_article1_parent" if "item" in chunk_id else None,
                        "chunk_type": "item" if "item" in chunk_id else "article" if "parent" in chunk_id else "text_block"
                    }
                })
        
        results.sort(key=lambda x: x["similarity"], reverse=True)
        final_offline_results = results[:top_k]
        logger.info(f"Offline retrieval found {len(final_offline_results)} results for query '{query}'.")
        return final_offline_results
    
    def _extract_phrases(self, text: str) -> List[str]:
        """
        텍스트에서 의미 있는 구문을 추출하는 헬퍼 메서드
        """
        # 단순 처리: 2-4 단어로 구성된 구문 추출
        words = text.split()
        phrases = []
        
        # 2단어 구문
        for i in range(len(words) - 1):
            phrases.append(f"{words[i]} {words[i+1]}")
        
        # 3단어 구문
        for i in range(len(words) - 2):
            phrases.append(f"{words[i]} {words[i+1]} {words[i+2]}")
        
        # 4단어 구문
        for i in range(len(words) - 3):
            phrases.append(f"{words[i]} {words[i+1]} {words[i+2]} {words[i+3]}")
        
        return phrases
    
    def highlight_keywords(self, text: str, keywords: List[str], context_size: int = 50) -> str:
        if not text or not keywords:
            return text
        
        highlighted_parts = []
        lower_text = text.lower()
        processed_indices = set()

        # Find all occurrences of all keywords
        positions = []
        for kw in set(k.lower() for k in keywords if k): # Use unique, lowercased keywords
            start_idx = 0
            while start_idx < len(lower_text):
                pos = lower_text.find(kw, start_idx)
                if pos == -1:
                    break
                if not any(existing_start <= pos < existing_end for existing_start, existing_end in positions):
                     # Add if not overlapping with an already found longer keyword match that might encompass this one
                    positions.append((pos, pos + len(kw)))
                start_idx = pos + len(kw)
        
        positions.sort()

        # Merge overlapping/adjacent positions to create context windows
        merged_windows = []
        for start, end in positions:
            # Check if this position is already covered by the last window
            if merged_windows and start < merged_windows[-1][1] + context_size // 2 : # Allow some proximity to merge windows
                merged_windows[-1] = (merged_windows[-1][0], max(merged_windows[-1][1], end))
            else:
                merged_windows.append((start, end))
        
        if not merged_windows: # No keywords found
            return text[:context_size * 2] + "..." if len(text) > context_size*2 else text


        for start, end in merged_windows:
            context_start = max(0, start - context_size)
            context_end = min(len(text), end + context_size)
            
            # Extract the snippet
            snippet = text[context_start:context_end]
            
            # Highlight keywords within this snippet
            temp_snippet = snippet
            for kw in sorted(set(k for k in keywords if k), key=len, reverse=True): # Highlight longer keywords first
                # Case-insensitive highlighting
                pattern = re.compile(re.escape(kw), re.IGNORECASE)
                temp_snippet = pattern.sub(f"[{kw}]", temp_snippet)
            
            prefix = "..." if context_start > 0 else ""
            suffix = "..." if context_end < len(text) else ""
            highlighted_parts.append(f"{prefix}{temp_snippet}{suffix}")
            
        return "\n...\n".join(highlighted_parts) if highlighted_parts else text[:context_size*2]+"..."


# Entry point for direct execution (for testing retriever.py independently)
if __name__ == "__main__":
    # .env 파일 로드를 위해 main.py 상단의 로직 일부 사용
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).resolve().parent.parent.parent / '.env' # 프로젝트 루트의 .env 가정
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            print(f"(Retriever Test) Loaded .env file from: {env_path}")
        else:
            print(f"(Retriever Test) .env file not found at {env_path}, ensure environment variables are set.")
    except Exception as e:
        print(f"(Retriever Test) Error loading .env: {e}")

    # 로깅 설정 (main.py의 setup_logger와 유사하게)
    log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


    config_obj = Config() # Config가 기본 설정 파일 (예: default_config.yaml)을 로드한다고 가정
    retriever_instance = DocumentRetriever(config_obj)
    
    if len(sys.argv) > 1:
        test_query = sys.argv[1]
        test_top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        test_collections = [sys.argv[3]] if len(sys.argv) > 3 else None # 예: CI_20060401
        use_parents = True if len(sys.argv) <=4 or sys.argv[4].lower() == 'true' else False


        print(f"Retriever Test Mode - Available Collections: {retriever_instance.collections}")
        if not test_collections:
             test_collections = retriever_instance.collections # 설정된 컬렉션이 없으면 사용 가능한 모든 컬렉션 사용
             if not test_collections and not retriever_instance.offline_mode : # 오프라인 모드가 아닌데 컬렉션이 없으면 경고
                  print("WARNING: No collections specified and no collections found in Milvus. Testing may use offline mode.")


        print(f"Searching for: '{test_query}', top_k={test_top_k}, collections={test_collections}, use_parent_chunks={use_parents}")
        
        # 테스트 시 쿼리 최적화와 필터 표현식을 다양하게 시도해볼 수 있도록 플래그 추가
        # 1. 원본 쿼리, 필터 없음 (가장 기본적인 벡터 검색)
        print("\n--- Test 1: Raw Query, No Filter, Small Chunks ---")
        results1 = retriever_instance.retrieve(test_query, top_k=test_top_k, target_collections=test_collections, 
                                            use_parent_chunks=False, enable_query_optimization=False, force_filter_expr=None)
        for i, r in enumerate(results1): print(f"  Res1.{i+1}: ID={r['id']}, Sim={r['similarity']:.4f}, Text='{r['content'][:50]}...', Meta={r.get('metadata',{}).get('chunk_type')}")

        # 2. 최적화된 쿼리, 기본 small_chunk 필터, Small Chunks
        print("\n--- Test 2: Optimized Query, Default Filter (small chunks), Small Chunks ---")
        results2 = retriever_instance.retrieve(test_query, top_k=test_top_k, target_collections=test_collections, 
                                            use_parent_chunks=False, enable_query_optimization=True, force_filter_expr='default')
        for i, r in enumerate(results2): print(f"  Res2.{i+1}: ID={r['id']}, Sim={r['similarity']:.4f}, Text='{r['content'][:50]}...', Meta={r.get('metadata',{}).get('chunk_type')}")

        # 3. 최적화된 쿼리, 기본 small_chunk 필터, 부모 청크 요청 (Small-to-Big)
        #    이것이 main.py의 기본 동작과 가장 유사
        if use_parents:
            print("\n--- Test 3: Optimized Query, Default Filter (small chunks searched), Parent Chunks Returned (Small-to-Big) ---")
            results3 = retriever_instance.retrieve(test_query, top_k=test_top_k, target_collections=test_collections, 
                                                use_parent_chunks=True, enable_query_optimization=True, force_filter_expr='default')
            for i, r in enumerate(results3): print(f"  Res3.{i+1}: ID={r['id']}, Sim(orig child)={r['similarity']:.4f}, Text='{r['content'][:70]}...', Meta={r.get('metadata',{}).get('chunk_type')}, ParentContext={r.get('metadata',{}).get('is_parent_context')}")

        # 4. Hybrid 검색 (내부적으로 retrieve 호출, use_parent_chunks=True 기본값으로 설정)
        print("\n--- Test 4: Hybrid Retrieve (Small-to-Big ON by default in test) ---")
        hybrid_results = retriever_instance.hybrid_retrieve(test_query, top_k=test_top_k, target_collections=test_collections, use_parent_chunks=use_parents, enable_query_optimization=True, force_filter_expr='default')
        print(f"Found {len(hybrid_results)} hybrid results:")
        for i, result in enumerate(hybrid_results):
            print(f"\nHybrid Result {i+1} (hybrid_score: {result.get('hybrid_score', result.get('similarity', 0.0)):.4f}):")
            print(f"  Collection: {result['collection']}")
            print(f"  Metadata: {result.get('metadata', {})}")
            highlighted = retriever_instance.highlight_keywords(result['content'], test_query.split(), context_size=100)
            print(f"  Content: {highlighted}")
            print("-" * 40)

    else:
        print("Usage: python retriever.py <query> [top_k] [collection_name] [use_parents_true_false]")
        print("Example: python retriever.py \"제1보험기간이란\" 3 CI_20060401 True")