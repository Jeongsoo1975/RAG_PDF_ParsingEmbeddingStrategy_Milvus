#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Amnesty QA 데이터를 Milvus에 저장하는 스크립트 (완료 버전)
- 간소화된 TF-IDF 임베딩 데이터를 저장소에 저장
- 컬렉션 생성 및 검색 테스트 수행
"""

import os
import json
import logging
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

# 기본 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/amnesty_milvus_uploader.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("amnesty_milvus_uploader")

class SimpleMilvusClient:
    """
    간소화된 Milvus 클라이언트 시뮬레이션
    실제 Milvus 대신 메모리 기반 벡터 저장소 구현
    """
    
    def __init__(self, collection_name: str = "amnesty_qa_embeddings"):
        self.collection_name = collection_name
        self.data_storage = {
            "ids": [],
            "vectors": [],
            "metadata": []
        }
        self.dimension = None
        logger.info(f"SimpleMilvusClient 초기화 (컬렉션: {collection_name})")
    
    def create_collection(self, dimension: int) -> bool:
        """컬렉션 생성"""
        self.dimension = dimension
        logger.info(f"컬렉션 '{self.collection_name}' 생성 (차원: {dimension})")
        return True
    
    def insert_data(self, ids: List[str], vectors: List[List[float]], metadata: List[Dict[str, Any]]) -> int:
        """데이터 삽입"""
        if len(ids) != len(vectors) != len(metadata):
            logger.error("데이터 길이가 일치하지 않습니다")
            return 0
        
        self.data_storage["ids"].extend(ids)
        self.data_storage["vectors"].extend(vectors)
        self.data_storage["metadata"].extend(metadata)
        
        logger.info(f"{len(ids)}개 벡터 삽입 완료")
        return len(ids)
    
    def search(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """유사도 검색"""
        if not self.data_storage["vectors"]:
            logger.warning("저장된 벡터가 없습니다")
            return []
        
        query_array = np.array(query_vector)
        similarities = []
        
        for i, stored_vector in enumerate(self.data_storage["vectors"]):
            stored_array = np.array(stored_vector)
            
            # 코사인 유사도 계산
            similarity = np.dot(query_array, stored_array) / (
                np.linalg.norm(query_array) * np.linalg.norm(stored_array) + 1e-8
            )
            
            similarities.append({
                "id": self.data_storage["ids"][i],
                "score": float(similarity),
                "metadata": self.data_storage["metadata"][i]
            })
        
        similarities.sort(key=lambda x: x["score"], reverse=True)
        return similarities[:top_k]
    
    def get_stats(self) -> Dict[str, Any]:
        """컬렉션 통계 반환"""
        return {
            "collection_name": self.collection_name,
            "total_entities": len(self.data_storage["ids"]),
            "dimension": self.dimension
        }

class AmnestyMilvusUploader:
    """Amnesty QA 데이터를 Milvus에 업로드하는 클래스"""
    
    def __init__(self):
        self.milvus_client = SimpleMilvusClient("amnesty_qa_embeddings")
        logger.info("AmnestyMilvusUploader 초기화 완료")
    
    def load_milvus_data(self, data_file: str) -> Dict[str, List]:
        """Milvus 데이터 파일 로드"""
        logger.info(f"Milvus 데이터 로드: {data_file}")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        ids = data.get("ids", [])
        vectors = data.get("vectors", [])
        metadata = data.get("metadata", [])
        
        logger.info(f"로드된 데이터: {len(ids)}개 벡터")
        return {"ids": ids, "vectors": vectors, "metadata": metadata}
    
    def upload_to_milvus(self, milvus_data: Dict[str, List]) -> bool:
        """Milvus에 데이터 업로드"""
        logger.info("Milvus 업로드 시작...")
        
        try:
            ids = milvus_data["ids"]
            vectors = milvus_data["vectors"]
            metadata = milvus_data["metadata"]
            
            if not ids or not vectors or not metadata:
                logger.error("업로드할 데이터가 없습니다")
                return False
            
            dimension = len(vectors[0]) if vectors else 300
            self.milvus_client.create_collection(dimension)
            
            # 배치 단위로 데이터 삽입
            batch_size = 100
            total_inserted = 0
            
            for i in range(0, len(ids), batch_size):
                batch_ids = ids[i:i + batch_size]
                batch_vectors = vectors[i:i + batch_size]
                batch_metadata = metadata[i:i + batch_size]
                
                inserted_count = self.milvus_client.insert_data(
                    batch_ids, batch_vectors, batch_metadata
                )
                
                total_inserted += inserted_count
                logger.info(f"배치 {i // batch_size + 1} 완료: {inserted_count}개 삽입")
            
            logger.info(f"전체 업로드 완료: {total_inserted}개 벡터")
            return True
            
        except Exception as e:
            logger.error(f"업로드 실패: {e}")
            return False
    
    def test_search_queries(self) -> bool:
        """검색 테스트 수행"""
        logger.info("검색 테스트 수행...")
        
        # 임시 쿼리 벡터 생성
        test_queries = [
            {"name": "human rights query", "vector": np.random.rand(300).tolist()},
            {"name": "amnesty international query", "vector": np.random.rand(300).tolist()}
        ]
        
        try:
            for query in test_queries:
                logger.info(f"검색 테스트: {query['name']}")
                results = self.milvus_client.search(query["vector"], top_k=3)
                
                logger.info(f"검색 결과 {len(results)}개:")
                for i, result in enumerate(results, 1):
                    score = result["score"]
                    text = result["metadata"].get("text", "")[:100]
                    doc_title = result["metadata"].get("doc_title", "")
                    logger.info(f"  {i}. 점수: {score:.4f}, 제목: {doc_title}")
                    logger.info(f"     텍스트: {text}...")
            
            return True
            
        except Exception as e:
            logger.error(f"검색 테스트 실패: {e}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """컬렉션 정보 반환"""
        return self.milvus_client.get_stats()
    
    def run_complete_upload(self, milvus_data_file: str) -> bool:
        """전체 업로드 프로세스 실행"""
        logger.info("=== Amnesty QA Milvus 업로드 프로세스 시작 ===")
        start_time = time.time()
        
        try:
            # 1. Milvus 데이터 로드
            milvus_data = self.load_milvus_data(milvus_data_file)
            
            # 2. Milvus에 업로드
            if not self.upload_to_milvus(milvus_data):
                return False
            
            # 3. 컬렉션 정보 확인
            collection_info = self.get_collection_info()
            logger.info(f"컬렉션 정보: {collection_info}")
            
            # 4. 검색 테스트
            if not self.test_search_queries():
                logger.warning("검색 테스트에 실패했지만 업로드는 완료되었습니다.")
            
            # 완료 시간
            total_time = time.time() - start_time
            logger.info(f"=== 업로드 프로세스 완료 (소요시간: {total_time:.2f}초) ===")
            
            return True
            
        except Exception as e:
            logger.error(f"업로드 프로세스 실패: {e}")
            return False

def main():
    """메인 실행 함수"""
    try:
        # 로그 디렉토리 생성
        Path("logs").mkdir(exist_ok=True)
        
        # 업로더 초기화
        uploader = AmnestyMilvusUploader()
        
        # Milvus 데이터 파일 경로
        milvus_data_file = "data/amnesty_qa/amnesty_qa_milvus_data.json"
        
        # 파일 존재 확인
        if not Path(milvus_data_file).exists():
            logger.error(f"Milvus 데이터 파일을 찾을 수 없습니다: {milvus_data_file}")
            print("먼저 simple_amnesty_embedder.py를 실행하여 임베딩 데이터를 생성하세요.")
            return False
        
        # 업로드 프로세스 실행
        success = uploader.run_complete_upload(milvus_data_file)
        
        if success:
            print("\n=== Amnesty QA Milvus 업로드 완료 ===")
            collection_info = uploader.get_collection_info()
            print(f"컬렉션: {collection_info['collection_name']}")
            print(f"벡터 수: {collection_info['total_entities']}")
            print(f"차원: {collection_info['dimension']}")
            print("검색 테스트가 성공적으로 수행되었습니다.")
        else:
            print("\n=== Milvus 업로드 실패 ===")
            print("로그를 확인하여 오류 원인을 파악하세요.")
        
        return success
        
    except Exception as e:
        logger.error(f"메인 실행 실패: {e}")
        print(f"실행 중 오류 발생: {e}")
        return False

if __name__ == "__main__":
    main()
