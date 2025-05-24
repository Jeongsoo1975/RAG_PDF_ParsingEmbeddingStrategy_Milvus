#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Amnesty QA 데이터 임베딩 및 Milvus 저장 스크립트
- 생성된 amnesty_qa 문서들을 임베딩하여 Milvus에 저장
- 기존 DocumentEmbedder와 MilvusClient 활용
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# 프로젝트 모듈 import
try:
    from src.rag.embedder import DocumentEmbedder
    from src.rag.parser import Document, TextChunk
    from src.vectordb.milvus_client import MilvusClient
    from src.utils.config import Config
    from src.utils.logger import get_logger
except ImportError as e:
    print(f"프로젝트 모듈 import 실패: {e}")
    import sys
    sys.exit(1)

# 로깅 설정
logger = get_logger("amnesty_embedder")

class AmnestyEmbeddingProcessor:
    """
    Amnesty QA 데이터를 임베딩하여 Milvus에 저장하는 클래스
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Amnesty 임베딩 프로세서 초기화
        
        Args:
            config: 설정 객체
        """
        self.config = config or Config()
        
        # DocumentEmbedder 초기화
        logger.info("DocumentEmbedder 초기화 중...")
        self.embedder = DocumentEmbedder(self.config)
        
        # Milvus 클라이언트 확인
        if not self.embedder.milvus_client:
            logger.error("Milvus 클라이언트가 초기화되지 않았습니다.")
            raise RuntimeError("Milvus 연결 실패")
        
        self.milvus_client = self.embedder.milvus_client
        
        # 컬렉션 이름 설정
        self.collection_name = "amnesty_qa_embeddings"
        
        logger.info("Amnesty 임베딩 프로세서 초기화 완료")
    
    def load_amnesty_documents(self, documents_file: str) -> List[Document]:
        """
        Amnesty 문서 JSON 파일에서 Document 객체들 로드
        
        Args:
            documents_file: 문서 JSON 파일 경로
            
        Returns:
            Document 객체 리스트
        """
        logger.info(f"Amnesty 문서 로드 중: {documents_file}")
        
        try:
            with open(documents_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            documents = []
            documents_data = data.get('documents', [])
            
            for doc_data in documents_data:
                # TextChunk 객체들 생성
                chunks = []
                for chunk_data in doc_data.get('chunks', []):
                    chunk = TextChunk(
                        chunk_id=chunk_data['chunk_id'],
                        doc_id=chunk_data['doc_id'],
                        text=chunk_data['text'],
                        page_num=chunk_data.get('page_num', 1),
                        chunk_type=chunk_data.get('chunk_type', 'context'),
                        metadata=chunk_data.get('metadata', {})
                    )
                    chunks.append(chunk)
                
                # Document 객체 생성
                document = Document(
                    doc_id=doc_data['doc_id'],
                    source=doc_data['source'],
                    title=doc_data['title'],
                    chunks=chunks,
                    parent_chunks=None,  # amnesty 데이터는 parent_chunks 없음
                    metadata=doc_data.get('metadata', {})
                )
                
                documents.append(document)
            
            logger.info(f"문서 로드 완료: {len(documents)}개 문서, {sum(len(doc.chunks) for doc in documents)}개 청크")
            return documents
            
        except Exception as e:
            logger.error(f"문서 로드 실패: {e}")
            raise
    
    def create_milvus_collection(self) -> bool:
        """
        Milvus에 amnesty_qa_embeddings 컬렉션 생성
        
        Returns:
            성공 여부
        """
        logger.info(f"Milvus 컬렉션 '{self.collection_name}' 생성 중...")
        
        try:
            # 기존 컬렉션이 있으면 삭제
            if self.milvus_client.has_collection(self.collection_name):
                logger.info(f"기존 컬렉션 '{self.collection_name}' 삭제 중...")
                self.milvus_client.drop_collection(self.collection_name)
            
            # 새 컬렉션 생성
            dimension = self.embedder.embedding_dimension
            success = self.milvus_client.create_collection(
                collection_name=self.collection_name,
                dimension=dimension,
                metric_type="COSINE"
            )
            
            if success:
                logger.info(f"컬렉션 '{self.collection_name}' 생성 성공 (차원: {dimension})")
                return True
            else:
                logger.error(f"컬렉션 '{self.collection_name}' 생성 실패")
                return False
                
        except Exception as e:
            logger.error(f"컬렉션 생성 중 오류: {e}")
            return False
    
    def embed_and_store_documents(self, documents: List[Document], batch_size: int = 5) -> bool:
        """
        문서들을 임베딩하고 Milvus에 저장
        
        Args:
            documents: 저장할 Document 리스트
            batch_size: 배치 크기
            
        Returns:
            성공 여부
        """
        logger.info(f"{len(documents)}개 문서 임베딩 및 저장 시작 (배치 크기: {batch_size})")
        
        total_chunks_stored = 0
        failed_documents = []
        
        # 배치 단위로 처리
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(documents) + batch_size - 1) // batch_size
            
            logger.info(f"배치 {batch_num}/{total_batches} 처리 중 ({len(batch)}개 문서)...")
            
            for doc in batch:
                try:
                    # 문서 임베딩 생성
                    logger.info(f"문서 '{doc.title}' 임베딩 중...")
                    embedded_doc = self.embedder.embed_document(doc)
                    
                    # Milvus에 저장
                    success = self.embedder.store_document_in_db(
                        embedded_doc, 
                        self.collection_name
                    )
                    
                    if success:
                        chunk_count = len(embedded_doc.chunks)
                        total_chunks_stored += chunk_count
                        logger.info(f"문서 '{doc.title}' 저장 성공 ({chunk_count}개 청크)")
                    else:
                        logger.error(f"문서 '{doc.title}' 저장 실패")
                        failed_documents.append(doc.title)
                        
                except Exception as e:
                    logger.error(f"문서 '{doc.title}' 처리 중 오류: {e}")
                    failed_documents.append(doc.title)
                    continue
            
            # 배치 간 잠시 대기 (메모리 정리)
            time.sleep(1)
        
        # 결과 요약
        success_count = len(documents) - len(failed_documents)
        logger.info(f"임베딩 및 저장 완료: 성공 {success_count}/{len(documents)}개 문서")
        logger.info(f"총 저장된 청크 수: {total_chunks_stored}")
        
        if failed_documents:
            logger.warning(f"실패한 문서들: {failed_documents}")
        
        return len(failed_documents) == 0
    
    def test_search(self, test_query: str = "What are human rights?", top_k: int = 3) -> List[Dict[str, Any]]:
        """
        저장된 데이터로 검색 테스트 수행
        
        Args:
            test_query: 테스트 쿼리
            top_k: 반환할 결과 수
            
        Returns:
            검색 결과 리스트
        """
        logger.info(f"검색 테스트 수행: '{test_query}'")
        
        try:
            # 쿼리 임베딩 생성
            query_embedding = self.embedder.embed_text(test_query)
            
            # Milvus에서 검색
            results = self.milvus_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                top_k=top_k,
                output_fields=["text", "doc_id", "chunk_type", "article_title"]
            )
            
            logger.info(f"검색 완료: {len(results)}개 결과 반환")
            
            # 결과 출력
            for i, result in enumerate(results, 1):
                score = result.get('score', 0)
                text = result.get('text', '')[:200] + '...' if len(result.get('text', '')) > 200 else result.get('text', '')
                logger.info(f"결과 {i}: 점수={score:.4f}, 텍스트='{text}'")
            
            return results
            
        except Exception as e:
            logger.error(f"검색 테스트 실패: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        컬렉션 통계 정보 반환
        
        Returns:
            통계 정보 딕셔너리
        """
        try:
            stats = self.milvus_client.get_collection_stats(self.collection_name)
            count = self.milvus_client.count(self.collection_name)
            
            stats['total_entities'] = count
            
            logger.info(f"컬렉션 '{self.collection_name}' 통계:")
            logger.info(f"- 총 엔티티 수: {count}")
            logger.info(f"- 상세 정보: {stats}")
            
            return stats
            
        except Exception as e:
            logger.error(f"통계 정보 조회 실패: {e}")
            return {}
    
    def run_complete_pipeline(self, documents_file: str) -> bool:
        """
        전체 임베딩 파이프라인 실행
        
        Args:
            documents_file: 문서 JSON 파일 경로
            
        Returns:
            성공 여부
        """
        logger.info("=== Amnesty QA 임베딩 파이프라인 시작 ===")
        start_time = time.time()
        
        try:
            # 1. 문서 로드
            documents = self.load_amnesty_documents(documents_file)
            
            # 2. Milvus 컬렉션 생성
            if not self.create_milvus_collection():
                return False
            
            # 3. 문서 임베딩 및 저장
            if not self.embed_and_store_documents(documents):
                logger.warning("일부 문서 저장에 실패했지만 계속 진행합니다.")
            
            # 4. 컬렉션 통계 확인
            self.get_collection_stats()
            
            # 5. 검색 테스트
            self.test_search("What are the basic principles of human rights?")
            self.test_search("How does Amnesty International work?")
            
            # 완료 시간 계산
            total_time = time.time() - start_time
            logger.info(f"=== 임베딩 파이프라인 완료 (소요시간: {total_time:.2f}초) ===")
            
            return True
            
        except Exception as e:
            logger.error(f"임베딩 파이프라인 실패: {e}")
            return False

def main():
    """메인 실행 함수"""
    try:
        # 설정 로드
        config = Config()
        
        # 프로세서 초기화
        processor = AmnestyEmbeddingProcessor(config)
        
        # 문서 파일 경로
        documents_file = "data/amnesty_qa/amnesty_qa_documents.json"
        
        # 파일 존재 확인
        if not Path(documents_file).exists():
            logger.error(f"문서 파일을 찾을 수 없습니다: {documents_file}")
            print("먼저 create_amnesty_data.py를 실행하여 데이터를 생성하세요.")
            return False
        
        # 파이프라인 실행
        success = processor.run_complete_pipeline(documents_file)
        
        if success:
            print("\n=== Amnesty QA 임베딩 처리 완료 ===")
            print(f"컬렉션: {processor.collection_name}")
            print(f"임베딩 차원: {processor.embedder.embedding_dimension}")
            print("검색 테스트가 성공적으로 수행되었습니다.")
        else:
            print("\n=== 임베딩 처리 실패 ===")
            print("로그를 확인하여 오류 원인을 파악하세요.")
        
        return success
        
    except Exception as e:
        logger.error(f"메인 실행 실패: {e}")
        print(f"실행 중 오류 발생: {e}")
        return False

if __name__ == "__main__":
    main()
