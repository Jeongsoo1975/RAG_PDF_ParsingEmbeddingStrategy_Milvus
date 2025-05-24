#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
간소화된 Amnesty QA 임베딩 스크립트
- torch 의존성 없이 간단한 TF-IDF 기반 임베딩 사용
- Milvus에 데이터 저장 테스트
"""

import os
import json
import logging
import time
import numpy as np
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import Counter
import re
import math

# 기본 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/amnesty_simple_embedder.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("amnesty_simple_embedder")

class SimpleTFIDFEmbedder:
    """
    간단한 TF-IDF 기반 임베딩 생성기
    torch 의존성 없이 기본적인 벡터 임베딩 제공
    """
    
    def __init__(self, dimension: int = 300):
        """
        TF-IDF 임베딩 생성기 초기화
        
        Args:
            dimension: 임베딩 차원
        """
        self.dimension = dimension
        self.vocabulary = {}
        self.idf_scores = {}
        self.documents = []
        
        logger.info(f"SimpleTFIDFEmbedder 초기화 (차원: {dimension})")
    
    def _preprocess_text(self, text: str) -> List[str]:
        """텍스트 전처리 및 토큰화"""
        # 소문자 변환 및 특수문자 제거
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        # 단어 분할
        words = text.split()
        # 길이가 2 이상인 단어만 사용
        words = [word for word in words if len(word) >= 2]
        return words
    
    def fit(self, documents: List[str]):
        """문서들로부터 TF-IDF 모델 학습"""
        logger.info(f"TF-IDF 모델 학습 시작 ({len(documents)}개 문서)")
        
        self.documents = documents
        
        # 어휘 구축
        word_doc_count = Counter()
        all_words = set()
        
        processed_docs = []
        for doc in documents:
            words = self._preprocess_text(doc)
            processed_docs.append(words)
            
            unique_words = set(words)
            for word in unique_words:
                word_doc_count[word] += 1
                all_words.add(word)
        
        # 상위 빈도 단어들을 어휘로 선택 (dimension 제한)
        common_words = word_doc_count.most_common(self.dimension)
        self.vocabulary = {word: idx for idx, (word, _) in enumerate(common_words)}
        
        # IDF 점수 계산
        total_docs = len(documents)
        for word, idx in self.vocabulary.items():
            doc_freq = word_doc_count[word]
            self.idf_scores[word] = math.log(total_docs / (doc_freq + 1))
        
        logger.info(f"어휘 크기: {len(self.vocabulary)}")
    
    def transform(self, text: str) -> np.ndarray:
        """텍스트를 TF-IDF 벡터로 변환"""
        words = self._preprocess_text(text)
        
        # TF 계산
        word_count = Counter(words)
        total_words = len(words)
        
        # 벡터 초기화
        vector = np.zeros(len(self.vocabulary))
        
        for word, count in word_count.items():
            if word in self.vocabulary:
                idx = self.vocabulary[word]
                tf = count / total_words if total_words > 0 else 0
                idf = self.idf_scores.get(word, 0)
                vector[idx] = tf * idf
        
        # 정규화
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector

class SimpleAmnestyEmbedder:
    """
    간소화된 Amnesty 임베딩 처리기
    """
    
    def __init__(self):
        """초기화"""
        self.embedding_dimension = 300
        self.embedder = SimpleTFIDFEmbedder(self.embedding_dimension)
        logger.info("SimpleAmnestyEmbedder 초기화 완료")
    
    def load_documents(self, documents_file: str) -> List[Dict[str, Any]]:
        """문서 JSON 파일 로드"""
        logger.info(f"문서 로드: {documents_file}")
        
        with open(documents_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = data.get('documents', [])
        logger.info(f"로드된 문서 수: {len(documents)}")
        
        return documents
    
    def prepare_embeddings(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """문서들에 대한 임베딩 생성"""
        logger.info("임베딩 준비 시작...")
        
        # 모든 텍스트 수집 (TF-IDF 학습용)
        all_texts = []
        for doc in documents:
            for chunk in doc.get('chunks', []):
                text = chunk.get('text', '')
                if text.strip():
                    all_texts.append(text)
        
        # TF-IDF 모델 학습
        self.embedder.fit(all_texts)
        
        # 각 청크에 임베딩 추가
        embedded_documents = []
        for doc in documents:
            new_doc = doc.copy()
            embedded_chunks = []
            
            for chunk in doc.get('chunks', []):
                new_chunk = chunk.copy()
                text = chunk.get('text', '')
                
                if text.strip():
                    # 임베딩 생성
                    embedding = self.embedder.transform(text)
                    
                    # 메타데이터에 임베딩 추가
                    if 'metadata' not in new_chunk:
                        new_chunk['metadata'] = {}
                    new_chunk['metadata']['embedding'] = embedding.tolist()
                
                embedded_chunks.append(new_chunk)
            
            new_doc['chunks'] = embedded_chunks
            embedded_documents.append(new_doc)
        
        logger.info(f"임베딩 생성 완료: {len(embedded_documents)}개 문서")
        return embedded_documents
    
    def save_embedded_documents(self, documents: List[Dict[str, Any]], output_file: str):
        """임베딩된 문서들을 파일로 저장"""
        output_data = {
            "dataset_info": {
                "name": "amnesty_qa_embedded",
                "embedding_dimension": self.embedding_dimension,
                "embedding_method": "tf_idf",
                "total_documents": len(documents),
                "creation_date": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "documents": documents
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"임베딩된 문서 저장: {output_file}")
    
    def test_similarity_search(self, documents: List[Dict[str, Any]], query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """간단한 유사도 검색 테스트"""
        logger.info(f"유사도 검색 테스트: '{query}'")
        
        # 쿼리 임베딩 생성
        query_embedding = self.embedder.transform(query)
        
        # 모든 청크와 유사도 계산
        similarities = []
        
        for doc in documents:
            for chunk in doc.get('chunks', []):
                if 'metadata' in chunk and 'embedding' in chunk['metadata']:
                    chunk_embedding = np.array(chunk['metadata']['embedding'])
                    
                    # 코사인 유사도 계산
                    similarity = np.dot(query_embedding, chunk_embedding)
                    
                    similarities.append({
                        'similarity': similarity,
                        'text': chunk.get('text', ''),
                        'doc_title': doc.get('title', ''),
                        'chunk_id': chunk.get('chunk_id', '')
                    })
        
        # 유사도 순으로 정렬
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # 상위 k개 반환
        top_results = similarities[:top_k]
        
        logger.info(f"검색 결과 {len(top_results)}개:")
        for i, result in enumerate(top_results, 1):
            logger.info(f"{i}. 유사도: {result['similarity']:.4f}")
            logger.info(f"   제목: {result['doc_title']}")
            logger.info(f"   텍스트: {result['text'][:100]}...")
        
        return top_results
    
    def create_milvus_data(self, documents: List[Dict[str, Any]]) -> Dict[str, List]:
        """Milvus 삽입용 데이터 준비"""
        logger.info("Milvus 삽입용 데이터 준비...")
        
        ids = []
        vectors = []
        metadata_list = []
        
        for doc in documents:
            for chunk in doc.get('chunks', []):
                if 'metadata' in chunk and 'embedding' in chunk['metadata']:
                    # ID 생성
                    chunk_id = chunk.get('chunk_id', str(uuid.uuid4()))
                    ids.append(chunk_id)
                    
                    # 벡터 추가
                    embedding = chunk['metadata']['embedding']
                    vectors.append(embedding)
                    
                    # 메타데이터 준비
                    metadata = {
                        "text": chunk.get('text', ''),
                        "doc_id": doc.get('doc_id', ''),
                        "doc_title": doc.get('title', ''),
                        "chunk_type": chunk.get('chunk_type', 'context'),
                        "page_num": chunk.get('page_num', 1),
                        "source": doc.get('source', ''),
                        "dataset": "amnesty_qa_simple"
                    }
                    metadata_list.append(metadata)
        
        logger.info(f"Milvus 데이터 준비 완료: {len(ids)}개 벡터")
        
        return {
            "ids": ids,
            "vectors": vectors,
            "metadata": metadata_list
        }
    
    def run_complete_pipeline(self, documents_file: str) -> bool:
        """전체 파이프라인 실행"""
        logger.info("=== 간소화된 Amnesty 임베딩 파이프라인 시작 ===")
        start_time = time.time()
        
        try:
            # 1. 문서 로드
            documents = self.load_documents(documents_file)
            
            # 2. 임베딩 생성
            embedded_documents = self.prepare_embeddings(documents)
            
            # 3. 임베딩된 문서 저장
            output_file = "data/amnesty_qa/amnesty_qa_embedded.json"
            self.save_embedded_documents(embedded_documents, output_file)
            
            # 4. 유사도 검색 테스트
            self.test_similarity_search(embedded_documents, "What are human rights?")
            self.test_similarity_search(embedded_documents, "How does Amnesty International work?")
            
            # 5. Milvus 데이터 준비
            milvus_data = self.create_milvus_data(embedded_documents)
            
            # Milvus 데이터 저장
            milvus_file = "data/amnesty_qa/amnesty_qa_milvus_data.json"
            with open(milvus_file, 'w', encoding='utf-8') as f:
                json.dump(milvus_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Milvus 데이터 저장: {milvus_file}")
            
            # 완료 시간
            total_time = time.time() - start_time
            logger.info(f"=== 파이프라인 완료 (소요시간: {total_time:.2f}초) ===")
            
            return True
            
        except Exception as e:
            logger.error(f"파이프라인 실패: {e}")
            return False

def main():
    """메인 실행 함수"""
    try:
        # 로그 디렉토리 생성
        Path("logs").mkdir(exist_ok=True)
        
        # 임베더 초기화
        embedder = SimpleAmnestyEmbedder()
        
        # 문서 파일 경로
        documents_file = "data/amnesty_qa/amnesty_qa_documents.json"
        
        # 파일 존재 확인
        if not Path(documents_file).exists():
            logger.error(f"문서 파일을 찾을 수 없습니다: {documents_file}")
            print("먼저 create_amnesty_data.py를 실행하여 데이터를 생성하세요.")
            return False
        
        # 파이프라인 실행
        success = embedder.run_complete_pipeline(documents_file)
        
        if success:
            print("\n=== Amnesty QA 간소화된 임베딩 처리 완료 ===")
            print(f"임베딩 차원: {embedder.embedding_dimension}")
            print("임베딩 방법: TF-IDF")
            print("유사도 검색 테스트가 성공적으로 수행되었습니다.")
            print("출력 파일:")
            print("- data/amnesty_qa/amnesty_qa_embedded.json")
            print("- data/amnesty_qa/amnesty_qa_milvus_data.json")
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
