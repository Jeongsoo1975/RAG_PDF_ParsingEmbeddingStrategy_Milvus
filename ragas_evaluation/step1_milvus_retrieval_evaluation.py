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
    """
    간단한 방법으로 쿼리 임베딩 생성 (실제 임베딩 대신 임시 벡터 사용)
    실제로는 임베딩 모델을 사용해야 하지만, 현재는 테스트용으로 랜덤 벡터 사용
    """
    print("[INFO] 쿼리 임베딩 생성 중...")
    
    query_embeddings = []
    
    for i, q in enumerate(questions[:10]):  # 처음 10개만 테스트
        question_text = q.get('text', '')
        question_id = q.get('id', f'q_{i}')
        
        # 임시 768차원 벡터 생성 (실제로는 임베딩 모델 사용해야 함)
        # 키워드 기반으로 약간의 패턴 부여
        embedding = [random.random() for _ in range(768)]
        
        # 보험, 계약 등 키워드가 있으면 특정 차원에 가중치 부여
        if any(keyword in question_text for keyword in ['보험', '계약', '보장']):
            for j in range(0, 100):  # 처음 100개 차원에 가중치
                embedding[j] += 0.3
        
        if any(keyword in question_text for keyword in ['지급', '보험금', '청구']):
            for j in range(100, 200):  # 다음 100개 차원에 가중치
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
            print("❌ Milvus 연결 실패")
            return {}
        
        print("✅ Milvus 연결 성공")
        
        # 사용 가능한 컬렉션 확인
        collections = client.list_collections()
        target_collection = "insurance_ko_sroberta"  # 가장 최신 모델 컬렉션
        
        if target_collection not in collections:
            print(f"❌ 대상 컬렉션 '{target_collection}' 없음")
            return {}
        
        print(f"✅ 대상 컬렉션: {target_collection}")
        
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
            print(f"\n🔍 검색 {i+1}/{len(query_embeddings)}: {q_id}")
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
                    print(f"   ✅ {len(search_results)}개 결과 반환 (소요시간: {search_time:.3f}초)")
                    
                    # 결과 분석
                    result_analysis = analyze_search_results(search_results, question_text)
                    
                    evaluation_results["search_results"].append({
                        "question_id": q_id,
                        "question": question_text,
                        "search_time": search_time,
                        "num_results": len(search_results),
                        "top_result": {
                            "score": search_results[0].get("score", 0),
                            "text": search_results[0].get("text", "")[:200],
                            "chunk_type": search_results[0].get("chunk_type", ""),
                            "page_num": search_results[0].get("page_num", -1)
                        },
                        "analysis": result_analysis
                    })
                    
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

def analyze_search_results(results: List[Dict], question: str) -> Dict[str, Any]:
    """검색 결과 분석"""
    if not results:
        return {"relevance_indicators": [], "chunk_type_distribution": {}, "score_statistics": {}}
    
    # 관련성 지표 분석 (키워드 기반)
    question_keywords = extract_keywords(question)
    relevance_indicators = []
    
    for result in results:
        text = result.get("text", "").lower()
        relevance_score = sum(1 for keyword in question_keywords if keyword.lower() in text)
        relevance_indicators.append(relevance_score)
    
    # 청크 타입 분포
    chunk_types = {}
    for result in results:
        chunk_type = result.get("chunk_type", "unknown")
        chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
    
    # 점수 통계
    scores = [result.get("score", 0) for result in results]
    score_stats = {
        "min_score": min(scores) if scores else 0,
        "max_score": max(scores) if scores else 0,
        "avg_score": sum(scores) / len(scores) if scores else 0,
        "score_range": max(scores) - min(scores) if scores else 0
    }
    
    return {
        "relevance_indicators": relevance_indicators,
        "chunk_type_distribution": chunk_types,
        "score_statistics": score_stats,
        "avg_relevance": sum(relevance_indicators) / len(relevance_indicators) if relevance_indicators else 0
    }

def extract_keywords(text: str) -> List[str]:
    """텍스트에서 키워드 추출"""
    insurance_keywords = [
        '보험', '계약', '보장', '지급', '청구', '보험금', '보험료', '피보험자', '계약자',
        '특약', '해지', '해약', '납입', '급여금', '수익자', '진단', '질병', '상해', 
        '재해', '장해', '사망', '만기', '갱신', '부활', '면제'
    ]
    
    found_keywords = []
    text_lower = text.lower()
    
    for keyword in insurance_keywords:
        if keyword in text_lower:
            found_keywords.append(keyword)
    
    return found_keywords

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
    
    # 청크 타입 분석
    chunk_type_summary = {}
    relevance_scores = []
    
    for result in results.get("search_results", []):
        analysis = result.get("analysis", {})
        
        # 청크 타입 집계
        for chunk_type, count in analysis.get("chunk_type_distribution", {}).items():
            chunk_type_summary[chunk_type] = chunk_type_summary.get(chunk_type, 0) + count
        
        # 관련성 점수 수집
        avg_relevance = analysis.get("avg_relevance", 0)
        if avg_relevance > 0:
            relevance_scores.append(avg_relevance)
    
    print(f"\n[ANALYSIS] 검색 결과 분석:")
    print(f"   - 청크 타입 분포:")
    for chunk_type, count in chunk_type_summary.items():
        print(f"     * {chunk_type}: {count}개")
    
    if relevance_scores:
        avg_relevance = sum(relevance_scores) / len(relevance_scores)
        print(f"   - 평균 관련성 점수: {avg_relevance:.2f}")
    
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
    main()f"     [{j+1}] Score: {score:.4f}, Type: {chunk_type}")
                        print(f"         Text: {text}...")
                
                else:
                    print(f"   ❌ 검색 결과 없음")
                    
            except Exception as search_error:
                print(f"   ❌ 검색 오류: {search_error}")
        
        # 성능 메트릭 계산
        evaluation_results["performance_metrics"] = {
            "success_rate": successful_searches / len(query_embeddings),
            "average_search_time": total_search_time / len(query_embeddings),
            "total_search_time": total_search_time,
            "successful_searches": successful_searches,
            "failed_searches": len(query_embeddings) - successful_searches
        }
        
        client.close()
        print(f"\n✅ 검색 성능 평가 완료")
        return evaluation_results
        
    except Exception as e:
        print(f"❌ Milvus 검색 평가 실패: {e}")
        import traceback
        traceback.print_exc()
        return {}

def analyze_search_results(results: List[Dict], question: str) -> Dict[str, Any]:
    """검색 결과 분석"""
    if not results:
        return {"relevance_indicators": [], "chunk_type_distribution": {}, "score_statistics": {}}
    
    # 관련성 지표 분석 (키워드 기반)
    question_keywords = extract_keywords(question)
    relevance_indicators = []
    
    for result in results:
        text = result.get("text", "").lower()
        relevance_score = sum(1 for keyword in question_keywords if keyword.lower() in text)
        relevance_indicators.append(relevance_score)
    
    # 청크 타입 분포
    chunk_types = {}
    for result in results:
        chunk_type = result.get("chunk_type", "unknown")
        chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
    
    # 점수 통계
    scores = [result.get("score", 0) for result in results]
    score_stats = {
        "min_score": min(scores) if scores else 0,
        "max_score": max(scores) if scores else 0,
        "avg_score": sum(scores) / len(scores) if scores else 0,
        "score_range": max(scores) - min(scores) if scores else 0
    }
    
    return {
        "relevance_indicators": relevance_indicators,
        "chunk_type_distribution": chunk_types,
        "score_statistics": score_stats,
        "avg_relevance": sum(relevance_indicators) / len(relevance_indicators) if relevance_indicators else 0
    }

def extract_keywords(text: str) -> List[str]:
    """텍스트에서 키워드 추출 (간단한 방법)"""
    # 보험 관련 주요 키워드
    insurance_keywords = [
        '보험', '계약', '보장', '지급', '청구', '보험금', '보험료', '피보험자', '계약자',
        '특약', '해지', '해약', '납입', '급여금', '수익자', '진단', '질병', '상해', 
        '재해', '장해', '사망', '만기', '갱신', '부활', '면제'
    ]
    
    found_keywords = []
    text_lower = text.lower()
    
    for keyword in insurance_keywords:
        if keyword in text_lower:
            found_keywords.append(keyword)
    
    return found_keywords

def save_evaluation_results(results: Dict[str, Any], filename: str = None):
    """평가 결과 저장"""
    if not results:
        print("❌ 저장할 결과가 없습니다.")
        return
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_results/step1_milvus_retrieval_{timestamp}.json"
    
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 평가 결과 저장: {filename}")
        
    except Exception as e:
        print(f"❌ 결과 저장 실패: {e}")

def print_evaluation_summary(results: Dict[str, Any]):
    """평가 결과 요약 출력"""
    if not results:
        return
    
    print_section_header("평가 결과 요약")
    
    metrics = results.get("performance_metrics", {})
    
    print(f"📊 검색 성능 통계:")
    print(f"   - 총 쿼리 수: {results.get('total_queries', 0)}개")
    print(f"   - 성공률: {metrics.get('success_rate', 0):.1%}")
    print(f"   - 평균 검색 시간: {metrics.get('average_search_time', 0):.3f}초")
    print(f"   - 성공한 검색: {metrics.get('successful_searches', 0)}개")
    print(f"   - 실패한 검색: {metrics.get('failed_searches', 0)}개")
    
    # 청크 타입 분석
    chunk_type_summary = {}
    relevance_scores = []
    
    for result in results.get("search_results", []):
        analysis = result.get("analysis", {})
        
        # 청크 타입 집계
        for chunk_type, count in analysis.get("chunk_type_distribution", {}).items():
            chunk_type_summary[chunk_type] = chunk_type_summary.get(chunk_type, 0) + count
        
        # 관련성 점수 수집
        avg_relevance = analysis.get("avg_relevance", 0)
        if avg_relevance > 0:
            relevance_scores.append(avg_relevance)
    
    print(f"\n📈 검색 결과 분석:")
    print(f"   - 청크 타입 분포:")
    for chunk_type, count in chunk_type_summary.items():
        print(f"     * {chunk_type}: {count}개")
    
    if relevance_scores:
        avg_relevance = sum(relevance_scores) / len(relevance_scores)
        print(f"   - 평균 관련성 점수: {avg_relevance:.2f}")
    
    print(f"\n✅ 1단계 평가 완료!")
    print(f"다음 단계에서는 수동 답변 데이터를 이용한 RAGAS 구조 검증을 진행합니다.")

def main():
    """메인 실행 함수"""
    print_section_header("RAG 평가 1단계: Milvus 검색 성능 평가")
    print("현재 작동하는 Milvus 검색 기능을 이용한 성능 평가를 시작합니다.")
    
    # 1. 평가 데이터셋 로드
    questions = load_evaluation_dataset()
    if not questions:
        print("❌ 평가를 진행할 수 없습니다.")
        return
    
    # 2. 쿼리 임베딩 생성 (임시)
    query_embeddings = create_simple_query_embeddings(questions)
    if not query_embeddings:
        print("❌ 쿼리 임베딩 생성 실패")
        return
    
    # 3. Milvus 검색 성능 평가
    evaluation_results = evaluate_milvus_retrieval(query_embeddings)
    if not evaluation_results:
        print("❌ 검색 성능 평가 실패")
        return
    
    # 4. 결과 저장 및 요약
    save_evaluation_results(evaluation_results)
    print_evaluation_summary(evaluation_results)
    
    print(f"\n🎯 다음 단계 안내:")
    print(f"   - 2단계: 수동 답변 데이터로 RAGAS 구조 검증")
    print(f"   - 3단계: 검색 기반 평가 메트릭 개발")

if __name__ == "__main__":
    main()
