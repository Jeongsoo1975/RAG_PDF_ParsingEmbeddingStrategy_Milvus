#!/usr/bin/env python3
"""
단계별 RAG 평가: 1단계 - Milvus 검색 성능 평가
PyTorch 의존성 없이 현재 작동하는 Milvus 검색 기능으로 평가 진행
"""

import sys
import os
import json
import time
import random
from datetime import datetime
from typing import List, Dict, Any, Tuple

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def print_section_header(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")

def load_evaluation_dataset():
    """평가 데이터셋 로드"""
    try:
        with open('src/evaluation/data/insurance_eval_dataset.json', 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        questions = dataset.get('questions', [])
        print(f"[OK] 평가 데이터셋 로드 성공: {len(questions)}개 질문")
        return questions
        
    except Exception as e:
        print(f"[FAIL] 평가 데이터셋 로드 실패: {e}")
        return []

def create_simple_query_embeddings(questions: List[Dict]) -> List[Tuple[str, List[float], str]]:
    """간단한 방법으로 쿼리 임베딩 생성"""
    print("[INFO] 쿼리 임베딩 생성 중...")
    
    query_embeddings = []
    
    for i, q in enumerate(questions[:10]):  # 처음 10개만 테스트
        question_text = q.get('text', '')
        question_id = q.get('id', f'q_{i}')
        
        # 임시 768차원 벡터 생성
        embedding = [random.random() for _ in range(768)]
        
        # 키워드 기반 가중치 부여
        if any(keyword in question_text for keyword in ['보험', '계약', '보장']):
            for j in range(0, 100):
                embedding[j] += 0.3
        
        if any(keyword in question_text for keyword in ['지급', '보험금', '청구']):
            for j in range(100, 200):
                embedding[j] += 0.3
        
        # 정규화
        norm = sum(x*x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x/norm for x in embedding]
        
        query_embeddings.append((question_id, embedding, question_text))
    
    print(f"[OK] {len(query_embeddings)}개 쿼리 임베딩 생성 완료")
    return query_embeddings

def evaluate_milvus_retrieval(query_embeddings: List[Tuple[str, List[float], str]]) -> Dict[str, Any]:
    """Milvus 검색 성능 평가"""
    print_section_header("Milvus 검색 성능 평가")
    
    try:
        from src.vectordb.milvus_client import MilvusClient
        from src.utils.config import Config
        
        config = Config()
        client = MilvusClient(config)
        
        if not client.is_connected():
            print("[FAIL] Milvus 연결 실패")
            return {}
        
        print("[OK] Milvus 연결 성공")
        
        # 사용 가능한 컬렉션 확인
        collections = client.list_collections()
        target_collection = "insurance_ko_sroberta"
        
        if target_collection not in collections:
            print(f"[FAIL] 대상 컬렉션 '{target_collection}' 없음")
            return {}
        
        print(f"[OK] 대상 컬렉션: {target_collection}")
        
        evaluation_results = {
            "collection_name": target_collection,
            "total_queries": len(query_embeddings),
            "search_results": [],
            "performance_metrics": {},
            "timestamp": datetime.now().isoformat()
        }
        
        total_search_time = 0
        successful_searches = 0
        
        for i, (q_id, embedding, question_text) in enumerate(query_embeddings):
            print(f"\n[SEARCH] 검색 {i+1}/{len(query_embeddings)}: {q_id}")
            print(f"   질문: {question_text[:100]}...")
            
            try:
                start_time = time.time()
                
                # 검색 실행
                search_results = client.search(
                    collection_name=target_collection,
                    query_vector=embedding,
                    top_k=5,
                    output_fields=["id", "text", "doc_id", "source", "page_num", "chunk_type"]
                )
                
                search_time = time.time() - start_time
                total_search_time += search_time
                
                if search_results:
                    successful_searches += 1
                    print(f"   [OK] {len(search_results)}개 결과 반환 (소요시간: {search_time:.3f}초)")
                    
                    # 상위 결과 미리보기
                    for j, result in enumerate(search_results[:3]):
                        score = result.get("score", 0)
                        text = result.get("text", "")[:100]
                        chunk_type = result.get("chunk_type", "")
                        print(f"     [{j+1}] Score: {score:.4f}, Type: {chunk_type}")
                        print(f"         Text: {text}...")
                
                else:
                    print(f"   [FAIL] 검색 결과 없음")
                    
            except Exception as search_error:
                print(f"   [ERROR] 검색 오류: {search_error}")
        
        # 성능 메트릭 계산
        evaluation_results["performance_metrics"] = {
            "success_rate": successful_searches / len(query_embeddings),
            "average_search_time": total_search_time / len(query_embeddings),
            "total_search_time": total_search_time,
            "successful_searches": successful_searches,
            "failed_searches": len(query_embeddings) - successful_searches
        }
        
        client.close()
        print(f"\n[OK] 검색 성능 평가 완료")
        return evaluation_results
        
    except Exception as e:
        print(f"[FAIL] Milvus 검색 평가 실패: {e}")
        import traceback
        traceback.print_exc()
        return {}

def save_evaluation_results(results: Dict[str, Any], filename: str = None):
    """평가 결과 저장"""
    if not results:
        print("[FAIL] 저장할 결과가 없습니다.")
        return
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_results/step1_milvus_retrieval_{timestamp}.json"
    
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"[OK] 평가 결과 저장: {filename}")
        
    except Exception as e:
        print(f"[FAIL] 결과 저장 실패: {e}")

def print_evaluation_summary(results: Dict[str, Any]):
    """평가 결과 요약 출력"""
    if not results:
        return
    
    print_section_header("평가 결과 요약")
    
    metrics = results.get("performance_metrics", {})
    
    print(f"[METRICS] 검색 성능 통계:")
    print(f"   - 총 쿼리 수: {results.get('total_queries', 0)}개")
    print(f"   - 성공률: {metrics.get('success_rate', 0):.1%}")
    print(f"   - 평균 검색 시간: {metrics.get('average_search_time', 0):.3f}초")
    print(f"   - 성공한 검색: {metrics.get('successful_searches', 0)}개")
    print(f"   - 실패한 검색: {metrics.get('failed_searches', 0)}개")
    
    print(f"\n[OK] 1단계 평가 완료!")
    print(f"다음 단계에서는 수동 답변 데이터를 이용한 RAGAS 구조 검증을 진행합니다.")

def main():
    """메인 실행 함수"""
    print_section_header("RAG 평가 1단계: Milvus 검색 성능 평가")
    print("현재 작동하는 Milvus 검색 기능을 이용한 성능 평가를 시작합니다.")
    
    # 1. 평가 데이터셋 로드
    questions = load_evaluation_dataset()
    if not questions:
        print("[FAIL] 평가를 진행할 수 없습니다.")
        return
    
    # 2. 쿼리 임베딩 생성 (임시)
    query_embeddings = create_simple_query_embeddings(questions)
    if not query_embeddings:
        print("[FAIL] 쿼리 임베딩 생성 실패")
        return
    
    # 3. Milvus 검색 성능 평가
    evaluation_results = evaluate_milvus_retrieval(query_embeddings)
    if not evaluation_results:
        print("[FAIL] 검색 성능 평가 실패")
        return
    
    # 4. 결과 저장 및 요약
    save_evaluation_results(evaluation_results)
    print_evaluation_summary(evaluation_results)
    
    print(f"\n[NEXT] 다음 단계 안내:")
    print(f"   - 2단계: 수동 답변 데이터로 RAGAS 구조 검증")
    print(f"   - 3단계: 검색 기반 평가 메트릭 개발")

if __name__ == "__main__":
    main()
