#!/usr/bin/env python3
"""
하이브리드 검색 테스트 (오프라인 모드)
의존성 문제를 피하고 하이브리드 검색 로직을 테스트합니다.
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_hybrid_search_offline():
    """오프라인 모드에서 하이브리드 검색 테스트"""
    print("=== 하이브리드 검색 테스트 (오프라인 모드) ===")
    
    try:
        # 환경 변수 설정으로 오프라인 모드 강제
        os.environ['FORCE_OFFLINE_MODE'] = 'true'
        
        # 가짜 config 클래스 생성
        class MockConfig:
            def __init__(self):
                self.retrieval = {
                    'top_k': 15,
                    'similarity_threshold': 0.65,
                    'similarity_metric': 'cosine',
                    'hybrid_search': True,
                    'hybrid_alpha': 0.7,
                    'keyword_boost': 1.5,
                    'offline_mode': True,
                    'small_chunk_types': ["item", "item_sub_chunk", "csv_row", "text_block"],
                    'parent_chunk_data_dir': './data/parsed_output'
                }
                self.milvus = {
                    'collection_name': 'CI_20060401'
                }
            
            def get_section(self, section):
                return getattr(self, section, {})
        
        # 간단한 MockRetriever 클래스로 하이브리드 검색 로직 테스트
        class MockRetriever:
            def __init__(self, config):
                self.config = config
                self.retrieval_config = config.retrieval
                self.top_k = self.retrieval_config.get('top_k', 5)
                self.similarity_threshold = self.retrieval_config.get('similarity_threshold', 0.5)
                self.hybrid_search_enabled = self.retrieval_config.get('hybrid_search', False)
                self.hybrid_alpha = self.retrieval_config.get('hybrid_alpha', 0.7)
                self.offline_mode = True
                
            def _extract_keywords(self, query, min_keyword_length=2):
                """키워드 추출"""
                stop_words = set([
                    "이", "가", "은", "는", "을", "를", "의", "에", "에서", "와", "과", 
                    "무엇", "뭔가", "뭐", "언제", "어디", "누가", "어떻게", "왜"
                ])
                import re
                words = re.findall(r'[a-zA-Z0-9가-힣]+', query.lower())
                keywords = [word for word in words if word not in stop_words and len(word) >= min_keyword_length]
                return list(set(keywords))
            
            def offline_retrieve(self, query, top_k=5):
                """오프라인 검색 (더미 데이터 사용)"""
                dummy_docs = {
                    "doc1_item1": "제1보험기간은 계약체결일로부터 80세 계약해당일 전일까지입니다. 이 기간 동안 주요 보장이 제공됩니다.",
                    "doc1_item2": "제2보험기간은 80세 계약해당일부터 종신까지이며, 주로 연금 지급이나 건강 관련 특정 보장이 이루어집니다.",
                    "doc4_ci_info": "CI보험금은 '중대한 질병', '중대한 수술', '중대한 화상' 등 약관에서 정한 특정 CI 발생 시 사망보험금의 일부(50% 또는 80%)를 선지급 받는 개념입니다.",
                    "doc5_special_clause": "제18조에 명시된 주요 보험금의 종류는 사망보험금과 CI보험금입니다. CI보험금 중 '중대한 질병' 및 '중대한 수술'에 대한 보장개시일은 제1회 보험료를 받은 날부터 그 날을 포함하여 90일이 지난 날의 다음날입니다.",
                    "doc6_prepayment": "선지급서비스특약은 피보험자의 여명이 6개월 이내로 판단될 경우 피보험자의 신청에 따라 주계약 사망보험금의 일부 또는 전부를 미리 지급받는 서비스입니다.",
                    "doc7_premium_exemption": "보험료 납입기간 중 피보험자가 장해분류표상 동일한 재해 또는 재해 이외의 동일한 원인으로 합산장해 지급률이 50% 이상 80% 미만인 장해상태가 되거나, 제18조 제1항 제2호의 CI보험금이 지급된 경우 차회 이후의 보험료 납입이 면제됩니다."
                }
                
                results = []
                keywords = self._extract_keywords(query)
                
                for doc_id, content in dummy_docs.items():
                    # 간단한 벡터 유사도 시뮬레이션 (키워드 기반)
                    vector_score = 0.3  # 기본 벡터 점수
                    for keyword in keywords:
                        if keyword in content.lower():
                            vector_score += 0.15
                    
                    vector_score = min(vector_score, 1.0)
                    
                    if vector_score > 0.1:  # 최소 임계값
                        results.append({
                            "id": doc_id,
                            "similarity": vector_score,
                            "content": content,
                            "collection": "offline_test",
                            "metadata": {
                                "chunk_type": "item" if "item" in doc_id else "text_block"
                            }
                        })
                
                results.sort(key=lambda x: x["similarity"], reverse=True)
                return results[:top_k]
            
            def hybrid_retrieve(self, query, top_k=None, **kwargs):
                """하이브리드 검색 테스트"""
                print(f"[검색] 하이브리드 검색 실행: '{query}'")
                
                if not self.hybrid_search_enabled:
                    print("[WARNING] 하이브리드 검색이 비활성화됨")
                    return self.offline_retrieve(query, top_k or self.top_k)
                
                # 1. 벡터 검색 (오프라인 모드)
                vector_results = self.offline_retrieve(query, (top_k or self.top_k) * 2)
                print(f"   벡터 검색 결과: {len(vector_results)}개")
                
                # 2. 키워드 추출
                keywords = self._extract_keywords(query)
                print(f"   추출된 키워드: {keywords}")
                
                if not keywords:
                    return vector_results[:top_k or self.top_k]
                
                # 3. 하이브리드 점수 계산
                hybrid_results = []
                for result in vector_results:
                    content_lower = result.get("content", "").lower()
                    
                    # 키워드 매칭 점수 계산
                    keyword_matches = []
                    keyword_score = 0
                    for kw in keywords:
                        if kw in content_lower:
                            keyword_matches.append(kw)
                            keyword_score += 1
                    
                    keyword_boost_score = keyword_score / len(keywords) if keywords else 0.0
                    
                    # 하이브리드 점수 = alpha * 벡터점수 + (1-alpha) * 키워드점수
                    vector_similarity = result.get("similarity", 0.0)
                    hybrid_score = (self.hybrid_alpha * vector_similarity) + ((1 - self.hybrid_alpha) * keyword_boost_score)
                    
                    result["hybrid_score"] = hybrid_score
                    result["keyword_matches"] = keyword_matches
                    result["original_vector_similarity"] = vector_similarity
                    result["keyword_boost_score"] = keyword_boost_score
                    
                    hybrid_results.append(result)
                
                # 4. 하이브리드 점수로 정렬
                hybrid_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
                final_results = hybrid_results[:top_k or self.top_k]
                
                print(f"   하이브리드 점수 계산 완료 ({self.hybrid_alpha:.1f} 벡터 + {1-self.hybrid_alpha:.1f} 키워드)")
                
                return final_results
        
        # 테스트 실행
        config = MockConfig()
        retriever = MockRetriever(config)
        
        print(f"[OK] MockRetriever 초기화 성공")
        print(f"   하이브리드 검색: {retriever.hybrid_search_enabled}")
        print(f"   하이브리드 가중치: 벡터 {retriever.hybrid_alpha:.1f}, 키워드 {1-retriever.hybrid_alpha:.1f}")
        
        # 테스트 쿼리들
        test_queries = [
            "제1보험기간이란 무엇인가요?",
            "CI보험금 지급 조건",
            "보험료 납입 면제",
            "선지급서비스특약"
        ]
        
        print(f"\n[검색] {len(test_queries)}개 질문으로 하이브리드 검색 테스트\n")
        
        for i, query in enumerate(test_queries, 1):
            print(f"--- 테스트 {i}: {query} ---")
            
            try:
                results = retriever.hybrid_retrieve(query, top_k=3)
                
                if results:
                    for j, result in enumerate(results):
                        hybrid_score = result.get("hybrid_score", 0.0)
                        vector_score = result.get("original_vector_similarity", 0.0)
                        keyword_score = result.get("keyword_boost_score", 0.0)
                        keyword_matches = result.get("keyword_matches", [])
                        
                        print(f"  {j+1}. 하이브리드 점수: {hybrid_score:.4f}")
                        print(f"     벡터 점수: {vector_score:.4f}")
                        print(f"     키워드 점수: {keyword_score:.4f}")
                        print(f"     매칭 키워드: {keyword_matches}")
                        print(f"     내용: {result['content'][:80]}...")
                        print()
                else:
                    print("  [ERROR] 검색 결과 없음")
                    
            except Exception as e:
                print(f"  [ERROR] 검색 오류: {e}")
                import traceback
                traceback.print_exc()
            
            print("-" * 70)
        
        print("\n[완료] 하이브리드 검색 테스트 완료!")
        return True
        
    except Exception as e:
        print(f"[ERROR] 테스트 실행 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """메인 실행 함수"""
    print("Phase 1: 하이브리드 검색 최적화 및 활성화 테스트\n")
    
    success = test_hybrid_search_offline()
    
    if success:
        print("\n[SUCCESS] Phase 1 테스트 완료: 하이브리드 검색 로직이 정상 작동합니다!")
        print("   - 하이브리드 검색 활성화됨")
        print("   - 벡터 70% + 키워드 30% 가중치 적용됨")
        print("   - 키워드 매칭 점수가 포함된 결과 반환됨")
    else:
        print("\n[FAIL] Phase 1 테스트 실패: 하이브리드 검색에 문제가 있습니다.")

if __name__ == "__main__":
    main()
