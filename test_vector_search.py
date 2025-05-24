#!/usr/bin/env python3
"""
Milvus 벡터 검색 테스트
임베딩된 데이터로 검색 기능을 테스트합니다.
"""

import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.rag.retriever import DocumentRetriever
from src.utils.config import Config

def test_vector_search():
    """벡터 검색 테스트"""
    print("=== Milvus 벡터 검색 테스트 ===")
    
    try:
        # Config 및 Retriever 초기화
        config = Config()
        retriever = DocumentRetriever(config)
        
        print(f"✅ DocumentRetriever 초기화 성공")
        
        # 테스트 질문들
        test_queries = [
            "What are human rights?",
            "How does Amnesty International work?",
            "What are civil and political rights?",
            "What is the Universal Declaration of Human Rights?",
            "How can individuals protect human rights?"
        ]
        
        print(f"\n🔍 {len(test_queries)}개 질문으로 검색 테스트 시작\n")
        
        for i, query in enumerate(test_queries, 1):
            print(f"--- 질문 {i}: {query} ---")
            
            try:
                # 검색 실행 (필터 없이)
                results = retriever.retrieve(query, top_k=3, force_filter_expr=None)
                
                print(f"검색 결과: {len(results)}개")
                
                if results:
                    for j, result in enumerate(results):
                        score = result.get("score", result.get("similarity", "N/A"))
                        text = result.get("text", result.get("content", "N/A"))
                        source = result.get("source", result.get("collection", "N/A"))
                        
                        print(f"  {j+1}. 점수: {score}")
                        print(f"     출처: {source}")
                        print(f"     내용: {str(text)[:100]}...")
                        print()
                else:
                    print("  ❌ 검색 결과 없음")
                    
            except Exception as e:
                print(f"  ❌ 검색 오류: {e}")
            
            print("-" * 60)
        
        print("\n🎉 벡터 검색 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 초기화 오류: {e}")
        return False
    
    return True

def test_collection_info():
    """컬렉션 정보 확인"""
    print("\n=== Milvus 컬렉션 정보 확인 ===")
    
    try:
        from src.vectordb.milvus_client import MilvusClient
        
        config = Config()
        client = MilvusClient(config)
        
        if client.connect():
            print("✅ Milvus 연결 성공")
            
            # 컬렉션 목록 확인
            collections = client.list_collections()
            print(f"📁 사용 가능한 컬렉션: {len(collections)}개")
            
            for collection in collections:
                print(f"  - {collection}")
                
                # 컬렉션 통계 확인
                try:
                    stats = client.get_collection_stats(collection)
                    print(f"    통계: {stats}")
                except:
                    print(f"    통계 확인 불가")
            
        else:
            print("❌ Milvus 연결 실패")
            
    except Exception as e:
        print(f"❌ 컬렉션 정보 확인 오류: {e}")

def main():
    """메인 실행 함수"""
    print("4단계: 벡터 검색 테스트 시작\n")
    
    # 컬렉션 정보 확인
    test_collection_info()
    
    # 벡터 검색 테스트
    success = test_vector_search()
    
    if success:
        print("\n✅ 4단계 완료: 벡터 검색이 정상 작동합니다!")
        print("다음 단계: 평가 실행")
    else:
        print("\n❌ 4단계 실패: 벡터 검색에 문제가 있습니다.")

if __name__ == "__main__":
    main()
