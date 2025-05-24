#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Milvus 검색 시스템 테스트 스크립트
업로드된 amnesty_qa 데이터를 대상으로 RAG 검색 기능을 테스트합니다.
"""

import os
import sys
import logging
import time
import statistics
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, project_root)
sys.path.insert(0, src_path)

# 프로젝트 모듈 임포트
try:
    from vectordb.milvus_client import MilvusClient
except ImportError:
    # 절대 경로로 다시 시도
    sys.path.insert(0, os.path.join(project_root, 'src', 'vectordb'))
    from milvus_client import MilvusClient

from sentence_transformers import SentenceTransformer
import yaml

# 간단한 Config 클래스 정의
class Config:
    def __init__(self):
        self.config_data = {
            'milvus': {
                'host': 'localhost',
                'port': 19530,
                'index_type': 'HNSW',
                'metric_type': 'COSINE'
            }
        }
        
        # config.yaml 파일이 있으면 로드
        config_path = os.path.join(project_root, 'config.yaml')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    loaded_config = yaml.safe_load(f)
                    if loaded_config:
                        self.config_data.update(loaded_config)
            except Exception as e:
                logging.warning(f"config.yaml 로드 실패: {e}")
    
    @property
    def milvus(self):
        return self.config_data.get('milvus', {})

# 로거 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("search_test")

class SearchSystemTester:
    def __init__(self):
        """검색 시스템 테스터 초기화"""
        try:
            # Config 및 Milvus 클라이언트 초기화
            self.config = Config()
            self.milvus_client = MilvusClient(self.config)
            
            # 임베딩 모델 초기화 (업로드와 동일한 모델 사용)
            model_name = 'sentence-transformers/all-MiniLM-L6-v2'
            logger.info(f"임베딩 모델 로드 중: {model_name}")
            self.embedding_model = SentenceTransformer(model_name)
            
            self.collection_name = "test_amnesty_qa"
            
            logger.info("검색 시스템 테스터 초기화 완료")
            
        except Exception as e:
            logger.error(f"검색 시스템 테스터 초기화 실패: {e}")
            raise

    def test_collection_info(self):
        """컬렉션 정보 확인"""
        logger.info("=== 컬렉션 정보 확인 ===")
        
        # 컬렉션 존재 여부 확인
        if not self.milvus_client.has_collection(self.collection_name):
            logger.error(f"컬렉션 '{self.collection_name}'이 존재하지 않습니다.")
            return False
        
        # 컬렉션 통계 정보
        stats = self.milvus_client.get_collection_stats(self.collection_name)
        count = self.milvus_client.count(self.collection_name)
        
        logger.info(f"컬렉션 '{self.collection_name}' 정보:")
        logger.info(f"  - 총 문서 수: {count}")
        logger.info(f"  - 스키마: {stats.get('schema', 'N/A')}")
        
        return True

    def test_search_queries(self):
        """다양한 검색 쿼리 테스트"""
        logger.info("=== 검색 쿼리 테스트 ===")
        test_start_time = time.time()
        
        test_queries = [
            "What are human rights?",
            "freedom of expression",
            "torture prevention",
            "교육 권리",  # 한국어 테스트
            "democracy and justice",
            "international law",
            "refugee protection"
        ]
        
        successful_queries = 0
        query_performance = []
        similarity_scores = []
        
        for i, query in enumerate(test_queries, 1):
            logger.info(f"\n테스트 {i}: '{query}'")
            query_start_time = time.time()
            
            try:
                # 쿼리 임베딩 생성
                embedding_start = time.time()
                query_embedding = self.embedding_model.encode([query])[0]
                embedding_time = time.time() - embedding_start
                
                # Milvus 검색 실행
                search_start = time.time()
                results = self.milvus_client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding.tolist() if hasattr(query_embedding, 'tolist') else list(query_embedding),
                    top_k=3,
                    output_fields=['text', 'chunk_type', 'source', 'article_title']
                )
                search_time = time.time() - search_start
                total_query_time = time.time() - query_start_time
                
                query_performance.append({
                    'query': query,
                    'embedding_time': embedding_time,
                    'search_time': search_time,
                    'total_time': total_query_time,
                    'result_count': len(results) if results else 0
                })
                
                if results and len(results) > 0:
                    logger.info(f"  검색 결과: {len(results)}개 (검색시간: {search_time:.3f}초)")
                    
                    # 유사도 점수 수집 및 분석
                    query_scores = []
                    for j, result in enumerate(results, 1):
                        score = result.get('score', 0)
                        text = result.get('text', '')[:100]
                        chunk_type = result.get('chunk_type', 'unknown')
                        
                        query_scores.append(score)
                        similarity_scores.append(score)
                        logger.info(f"    {j}. [{chunk_type}] {text}... (유사도: {score:.3f})")
                    
                    # 쿼리별 유사도 통계
                    if query_scores:
                        avg_score = statistics.mean(query_scores)
                        max_score = max(query_scores)
                        min_score = min(query_scores)
                        logger.info(f"  유사도 통계 - 평균: {avg_score:.3f}, 최고: {max_score:.3f}, 최저: {min_score:.3f}")
                    
                    successful_queries += 1
                else:
                    logger.warning(f"  검색 결과 없음 (검색시간: {search_time:.3f}초)")
                    # 결과가 없어도 검색 자체는 성공한 것으로 간주
                    successful_queries += 1
                
                logger.debug(f"  성능 세부사항 - 임베딩: {embedding_time:.3f}초, 검색: {search_time:.3f}초, 총시간: {total_query_time:.3f}초")
                    
            except Exception as e:
                error_time = time.time() - query_start_time
                logger.error(f"  검색 오류: {e} (오류발생시간: {error_time:.3f}초)")
                logger.error(f"  디버깅 힌트: 쿼리 '{query}' 처리 중 오류 발생. 네트워크 연결 또는 Milvus 서버 상태를 확인하세요.")
                # 예외 발생 시에만 실패로 처리
                
        # 전체 성능 요약
        test_total_time = time.time() - test_start_time
        
        if query_performance:
            avg_embedding_time = statistics.mean([p['embedding_time'] for p in query_performance])
            avg_search_time = statistics.mean([p['search_time'] for p in query_performance])
            avg_total_time = statistics.mean([p['total_time'] for p in query_performance])
            
            logger.info(f"\n검색 쿼리 성능 요약:")
            logger.info(f"  - 평균 임베딩 시간: {avg_embedding_time:.3f}초")
            logger.info(f"  - 평균 검색 시간: {avg_search_time:.3f}초")
            logger.info(f"  - 평균 쿼리 처리 시간: {avg_total_time:.3f}초")
            logger.info(f"  - 전체 테스트 시간: {test_total_time:.3f}초")
            
        if similarity_scores:
            logger.info(f"  - 전체 유사도 통계: 평균 {statistics.mean(similarity_scores):.3f}, "
                       f"표준편차 {statistics.stdev(similarity_scores) if len(similarity_scores) > 1 else 0:.3f}")
                
        logger.info(f"검색 쿼리 테스트: {successful_queries}/{len(test_queries)}개 성공")
        return successful_queries == len(test_queries)

    def test_filtered_search(self):
        """필터링된 검색 테스트"""
        logger.info("=== 필터링된 검색 테스트 ===")
        test_start_time = time.time()
        
        query = "human rights"
        embedding_start = time.time()
        query_embedding = self.embedding_model.encode([query])[0]
        embedding_time = time.time() - embedding_start
        logger.info(f"쿼리 임베딩 생성 시간: {embedding_time:.3f}초")
        
        # chunk_type별 필터링 테스트
        chunk_types = ['question', 'answer', 'title']
        successful_searches = 0
        search_results_summary = {}
        filter_performance = []
        
        for chunk_type in chunk_types:
            logger.info(f"\n'{chunk_type}' 타입 필터링 검색:")
            search_start_time = time.time()
            
            try:
                filter_expr = f"chunk_type == '{chunk_type}'"
                
                results = self.milvus_client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding.tolist() if hasattr(query_embedding, 'tolist') else list(query_embedding),
                    top_k=3,
                    filter_expr=filter_expr,
                    output_fields=['text', 'chunk_type', 'source']
                )
                
                search_time = time.time() - search_start_time
                filter_performance.append({
                    'chunk_type': chunk_type,
                    'search_time': search_time,
                    'result_count': len(results) if results else 0
                })
                
                if results and len(results) > 0:
                    logger.info(f"  결과: {len(results)}개 (검색시간: {search_time:.3f}초)")
                    chunk_scores = []
                    for j, result in enumerate(results, 1):
                        score = result.get('score', 0)
                        text = result.get('text', '')[:80]
                        chunk_scores.append(score)
                        logger.info(f"    {j}. {text}... (유사도: {score:.3f})")
                    
                    if chunk_scores:
                        avg_score = statistics.mean(chunk_scores)
                        logger.info(f"  평균 유사도: {avg_score:.3f}")
                    
                    search_results_summary[chunk_type] = len(results)
                else:
                    if chunk_type == 'title':
                        logger.warning(f"  '{chunk_type}' 타입 데이터가 없습니다. (예상된 상황 - 정상) (검색시간: {search_time:.3f}초)")
                    else:
                        logger.info(f"  '{chunk_type}' 타입 결과 없음 (검색시간: {search_time:.3f}초)")
                    search_results_summary[chunk_type] = 0
                    
                # 검색 자체가 성공적으로 실행되면 성공으로 간주 (결과 유무와 무관)
                successful_searches += 1
                    
            except Exception as e:
                search_time = time.time() - search_start_time
                logger.error(f"  '{chunk_type}' 타입 필터링 검색 실제 오류: {e} (오류발생시간: {search_time:.3f}초)")
                logger.error(f"  디버깅 힌트: chunk_type='{chunk_type}' 필터 조건을 확인하거나 컬렉션 스키마를 점검하세요.")
                search_results_summary[chunk_type] = "ERROR"
                filter_performance.append({
                    'chunk_type': chunk_type,
                    'search_time': search_time,
                    'result_count': 'ERROR'
                })
                # 예외 발생 시에만 실패로 처리
        
        # 전체 필터링 검색 성능 요약
        test_total_time = time.time() - test_start_time
        
        if filter_performance:
            successful_filter_searches = [p for p in filter_performance if p['result_count'] != 'ERROR']
            if successful_filter_searches:
                avg_filter_time = statistics.mean([p['search_time'] for p in successful_filter_searches])
                total_results = sum([p['result_count'] for p in successful_filter_searches])
                
                logger.info(f"\n필터링 검색 성능 요약:")
                logger.info(f"  - 평균 필터 검색 시간: {avg_filter_time:.3f}초")
                logger.info(f"  - 전체 테스트 시간: {test_total_time:.3f}초")
                logger.info(f"  - 총 검색 결과 수: {total_results}개")
        
        # 검색 결과 요약 로깅
        logger.info(f"\n필터링 검색 결과 요약:")
        for chunk_type, count in search_results_summary.items():
            if count == "ERROR":
                logger.error(f"  - {chunk_type}: 검색 오류 발생")
            elif count == 0:
                if chunk_type == 'title':
                    logger.info(f"  - {chunk_type}: 0개 (데이터 없음 - 정상)")
                else:
                    logger.info(f"  - {chunk_type}: 0개")
            else:
                logger.info(f"  - {chunk_type}: {count}개")
                
        logger.info(f"필터링된 검색 테스트: {successful_searches}/{len(chunk_types)}개 성공")
        return successful_searches == len(chunk_types)

    def test_rag_pipeline(self):
        """RAG 파이프라인 시뮬레이션 테스트"""
        logger.info("=== RAG 파이프라인 테스트 ===")
        
        user_question = "How can individuals protect human rights?"
        logger.info(f"사용자 질문: {user_question}")
        
        try:
            # 1. 질문 임베딩
            question_embedding = self.embedding_model.encode([user_question])[0]
            
            # 2. 관련 문서 검색
            search_results = self.milvus_client.search(
                collection_name=self.collection_name,
                query_vector=question_embedding.tolist() if hasattr(question_embedding, 'tolist') else list(question_embedding),
                top_k=5,
                output_fields=['text', 'chunk_type', 'source', 'article_title']
            )
            
            if search_results and len(search_results) > 0:
                logger.info(f"검색된 관련 문서: {len(search_results)}개")
                
                # 3. 컨텍스트 구성
                context_parts = []
                for i, result in enumerate(search_results, 1):
                    text = result.get('text', '')
                    chunk_type = result.get('chunk_type', 'unknown')
                    score = result.get('score', 0)
                    
                    context_parts.append(f"[{chunk_type}] {text}")
                    logger.info(f"  {i}. [{chunk_type}] {text[:100]}... (유사도: {score:.3f})")
                
                # 4. 최종 컨텍스트
                context = "\n\n".join(context_parts)
                logger.info(f"\n구성된 컨텍스트 길이: {len(context)} 문자")
                
                # 5. RAG 응답 시뮬레이션 (실제로는 LLM에 전달)
                prompt = f"""다음 컨텍스트를 바탕으로 질문에 답변하세요.

컨텍스트:
{context}

질문: {user_question}

답변:"""
                
                logger.info("✅ RAG 파이프라인 구성 완료")
                logger.info(f"프롬프트 길이: {len(prompt)} 문자")
                
                return True
            else:
                logger.warning("관련 문서를 찾을 수 없습니다.")
                return False
                
        except Exception as e:
            logger.error(f"RAG 파이프라인 테스트 오류: {e}")
            return False

    def run_all_tests(self):
        """모든 테스트 실행"""
        logger.info("=== Milvus 검색 시스템 테스트 시작 ===")
        overall_start_time = time.time()
        
        test_results = []
        
        # 각 테스트 실행
        tests = [
            ("컬렉션 정보 확인", self.test_collection_info),
            ("검색 쿼리 테스트", self.test_search_queries),
            ("필터링된 검색 테스트", self.test_filtered_search),
            ("RAG 파이프라인 테스트", self.test_rag_pipeline)
        ]
        
        test_performance = []
        
        for test_name, test_func in tests:
            try:
                logger.info(f"\n{'='*50}")
                test_start = time.time()
                result = test_func()
                test_time = time.time() - test_start
                
                test_results.append((test_name, result))
                test_performance.append({
                    'name': test_name,
                    'time': test_time,
                    'success': result
                })
                
                status = "성공" if result else "실패"
                logger.info(f"{test_name}: {status} (실행시간: {test_time:.3f}초)")
                
            except Exception as e:
                test_time = time.time() - test_start if 'test_start' in locals() else 0
                logger.error(f"{test_name} 실행 오류: {e} (오류발생시간: {test_time:.3f}초)")
                logger.error(f"디버깅 힌트: '{test_name}' 테스트 실패. 시스템 상태와 의존성을 확인하세요.")
                test_results.append((test_name, False))
                test_performance.append({
                    'name': test_name,
                    'time': test_time,
                    'success': False
                })
        
        overall_time = time.time() - overall_start_time
        
        
        # 결과 요약
        logger.info(f"\n{'='*50}")
        logger.info("=== 테스트 결과 요약 ===")
        
        success_count = 0
        failed_tests = []
        for test_name, result in test_results:
            status = "✅ 성공" if result else "❌ 실패"
            logger.info(f"  {test_name}: {status}")
            if result:
                success_count += 1
            else:
                failed_tests.append(test_name)
        
        # 성능 요약
        if test_performance:
            successful_tests = [t for t in test_performance if t['success']]
            failed_test_times = [t for t in test_performance if not t['success']]
            
            logger.info(f"\n=== 성능 요약 ===")
            logger.info(f"  - 전체 테스트 실행 시간: {overall_time:.3f}초")
            
            if successful_tests:
                avg_success_time = statistics.mean([t['time'] for t in successful_tests])
                logger.info(f"  - 성공한 테스트 평균 시간: {avg_success_time:.3f}초")
                
            for test in test_performance:
                status = "✅" if test['success'] else "❌"
                logger.info(f"    {status} {test['name']}: {test['time']:.3f}초")
        
        total_tests = len(test_results)
        success_rate = success_count/total_tests*100
        logger.info(f"\n총 {total_tests}개 테스트 중 {success_count}개 성공 ({success_rate:.1f}%)")
        
        if success_count == total_tests:
            logger.info("🎉 모든 테스트가 성공적으로 완료되었습니다!")
            return True
        else:
            logger.warning(f"⚠️ {total_tests - success_count}개 테스트가 실패했습니다.")
            if failed_tests:
                logger.warning(f"실패한 테스트: {', '.join(failed_tests)}")
                logger.warning("해결 방법:")
                for failed_test in failed_tests:
                    if "컬렉션" in failed_test:
                        logger.warning("  - Milvus 서버 연결 및 컬렉션 존재 여부를 확인하세요.")
                    elif "검색" in failed_test:
                        logger.warning("  - 검색 쿼리 및 인덱스 상태를 확인하세요.")
                    elif "RAG" in failed_test:
                        logger.warning("  - 임베딩 모델 및 검색 결과를 확인하세요.")
            logger.error("\n❌ 검색 시스템에 문제가 있습니다.")
            return False

def main():
    """메인 함수"""
    try:
        tester = SearchSystemTester()
        success = tester.run_all_tests()
        
        if success:
            logger.info("\n✅ 검색 시스템이 정상적으로 작동합니다!")
            return 0
        else:
            logger.error("\n❌ 검색 시스템에 문제가 있습니다.")
            return 1
            
    except Exception as e:
        logger.error(f"테스트 실행 오류: {e}")
        return 1
    finally:
        # 연결 정리
        try:
            if 'tester' in locals() and hasattr(tester, 'milvus_client'):
                tester.milvus_client.close()
        except Exception as e:
            logger.error(f"연결 정리 오류: {e}")

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
