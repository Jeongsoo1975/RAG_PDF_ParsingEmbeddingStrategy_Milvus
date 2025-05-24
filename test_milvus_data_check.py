#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Milvus 데이터 검증 및 테스트
실제 사용 시나리오 테스트 전에 Milvus에 데이터가 제대로 저장되어 있는지 확인합니다.
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

try:
    from src.utils.config import Config
    from src.rag.retriever import DocumentRetriever
    from src.utils.logger import setup_logger
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# 로그 설정
logger = setup_logger("milvus_data_check")

def main():
    """Milvus 데이터 상태 확인"""
    try:
        logger.info("=== Milvus 데이터 검증 시작 ===")
        
        # Config 로드
        config = Config()
        logger.info("Config 로드 완료")
        
        # Retriever 초기화
        retriever = DocumentRetriever(config)
        logger.info("DocumentRetriever 초기화 완료")
        
        # 1. 컬렉션 목록 확인
        try:
            collections = retriever.milvus_client.list_collections()
            logger.info(f"사용 가능한 컬렉션 수: {len(collections)}")
            for i, collection in enumerate(collections):
                logger.info(f"  {i+1}. {collection}")
        except Exception as e:
            logger.error(f"컬렉션 목록 조회 실패: {str(e)}")
            return False
        
        # 2. 첫 번째 컬렉션의 데이터 개수 확인
        if collections:
            test_collection = collections[0]
            try:
                # Milvus에서 직접 쿼리를 통해 개수 확인
                from pymilvus import connections, utility
                
                # 연결이 이미 되어 있다고 가정하고 utility 사용
                count = utility.get_entity_num(test_collection)
                logger.info(f"컬렉션 '{test_collection}'의 데이터 개수: {count}")
                
                if count == 0:
                    logger.warning(f"컬렉션 '{test_collection}'에 데이터가 없습니다!")
                    # 하지만 다른 컬렉션도 확인해보자
                    for collection_name in collections[:3]:  # 처음 3개만 확인
                        try:
                            count = utility.get_entity_num(collection_name)
                            logger.info(f"컬렉션 '{collection_name}': {count}개 데이터")
                            if count > 0:
                                break
                        except Exception as e:
                            logger.warning(f"컬렉션 '{collection_name}' 조회 오류: {e}")
                    
            except Exception as e:
                logger.error(f"데이터 개수 조회 실패: {str(e)}")
                logger.info("다른 방법으로 데이터 존재 여부를 확인합니다.")
        
        # 3. 간단한 검색 테스트 (임계값을 낮춰서)
        test_queries = [
            "human rights",
            "rights",
            "what",
            "the"
        ]
        
        logger.info("=== 낮은 임계값(0.3)으로 검색 테스트 ===")
        
        for query in test_queries:
            try:
                # 임계값을 0.3으로 낮춰서 테스트
                results = retriever.retrieve(query=query, top_k=3, threshold=0.3)
                logger.info(f"쿼리 '{query}': {len(results) if results else 0}개 결과")
                
                if results and len(results) > 0:
                    logger.info(f"  첫 번째 결과 미리보기: {str(results[0])[:100]}...")
                    return True  # 하나라도 결과가 나오면 성공
                    
            except Exception as e:
                logger.error(f"검색 테스트 실패 '{query}': {str(e)}")
        
        logger.warning("모든 검색 테스트에서 결과를 찾지 못했습니다.")
        
        # 4. 임계값 없이 테스트 (threshold=0.0)
        logger.info("=== 임계값 없이(0.0) 검색 테스트 ===")
        
        try:
            results = retriever.retrieve(query="human", top_k=5, threshold=0.0)
            logger.info(f"임계값 0.0으로 'human' 검색: {len(results) if results else 0}개 결과")
            
            if results and len(results) > 0:
                for i, result in enumerate(results[:3]):
                    logger.info(f"  결과 {i+1}: {str(result)[:150]}...")
                return True
                
        except Exception as e:
            logger.error(f"임계값 0.0 검색 실패: {str(e)}")
        
        return False
        
    except Exception as e:
        logger.error(f"검증 중 오류 발생: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Milvus 데이터 검증 성공!")
        sys.exit(0)
    else:
        print("\n❌ Milvus 데이터 검증 실패!")
        sys.exit(1)
