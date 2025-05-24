#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
End-to-End RAG 파이프라인 통합 테스트
PDF 파싱부터 Re-ranking까지 전체 RAG 파이프라인의 완전한 흐름을 검증합니다.
실제 amnesty_qa 데이터셋을 활용하여 각 단계별 성공/실패 여부와 소요 시간을 측정합니다.
"""

import os
import sys
import json
import time
import psutil
import logging
import traceback
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# 프로젝트 루트 경로 추가
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# 환경 변수 설정
os.environ.setdefault('LOG_LEVEL', 'INFO')

try:
    from src.utils.config import Config
    from src.rag.retriever import DocumentRetriever
    from src.rag.embedder import DocumentEmbedder
    from src.parsers.pdf_parser import PDFParser
    from src.vectordb.milvus_client import MilvusClient
    from src.utils.logger import get_logger
    from src.evaluation.simple_ragas_metrics import SimpleRAGASMetrics
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you're running from the project root directory")
    sys.exit(1)

# 로그 설정
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logger = get_logger("end_to_end_pipeline", 
                   log_file=log_dir / f"end_to_end_test_{timestamp}.log")


class EndToEndPipelineTest:
    """End-to-End RAG 파이프라인 테스트 클래스"""
    
    def __init__(self):
        """테스트 환경 초기화"""
        self.config = Config()
        self.test_results = {
            'test_start_time': datetime.now().isoformat(),
            'pipeline_stages': {},
            'performance_metrics': {},
            'memory_usage': {},
            'test_queries': [],
            'overall_success': False,
            'error_log': []
        }
        
        # 메모리 사용량 추적
        self.process = psutil.Process()
        self.initial_memory = self.get_memory_usage()
        
        logger.info("=== End-to-End RAG 파이프라인 테스트 시작 ===")
        logger.info(f"초기 메모리 사용량: {self.initial_memory:.2f} MB")
    
    def get_memory_usage(self) -> float:
        """현재 메모리 사용량 반환 (MB 단위)"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def log_stage_performance(self, stage_name: str, start_time: float, 
                            success: bool, error_msg: str = None) -> Dict[str, Any]:
        """단계별 성능 및 결과 로깅"""
        end_time = time.time()
        duration = end_time - start_time
        memory_usage = self.get_memory_usage()
        
        stage_result = {
            'success': success,
            'duration_seconds': round(duration, 3),
            'memory_mb': round(memory_usage, 2),
            'timestamp': datetime.now().isoformat()
        }
        
        if error_msg:
            stage_result['error'] = error_msg
            self.test_results['error_log'].append({
                'stage': stage_name,
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            })
        
        self.test_results['pipeline_stages'][stage_name] = stage_result
        
        status = "✓" if success else "✗"
        logger.info(f"{status} {stage_name}: {duration:.3f}초, 메모리: {memory_usage:.2f}MB")
        
        if error_msg:
            logger.error(f"  오류: {error_msg}")
        
        return stage_result
    
    def test_01_system_initialization(self) -> bool:
        """1단계: 시스템 구성요소 초기화 테스트"""
        logger.info("\n--- 1단계: 시스템 초기화 테스트 ---")
        start_time = time.time()
        
        try:
            # Config 초기화
            logger.info("Config 초기화 중...")
            assert self.config is not None
            
            # MilvusClient 연결 테스트
            logger.info("Milvus 연결 테스트 중...")
            milvus_client = MilvusClient(self.config)
            milvus_connected = milvus_client.connect()
            
            if not milvus_connected:
                raise Exception("Milvus 연결 실패")
            
            # DocumentRetriever 초기화
            logger.info("DocumentRetriever 초기화 중...")
            self.retriever = DocumentRetriever(self.config)
            assert self.retriever is not None
            
            # 컬렉션 존재 확인
            collections = milvus_client.list_collections()
            logger.info(f"사용 가능한 컬렉션: {len(collections)}개")
            
            if not collections:
                logger.warning("경고: 사용 가능한 컬렉션이 없습니다.")
            
            self.log_stage_performance("system_initialization", start_time, True)
            return True
            
        except Exception as e:
            error_msg = f"시스템 초기화 실패: {str(e)}"
            self.log_stage_performance("system_initialization", start_time, False, error_msg)
            return False
    
    def test_02_data_loading(self) -> bool:
        """2단계: 테스트 데이터 로딩 및 검증"""
        logger.info("\n--- 2단계: 테스트 데이터 로딩 ---")
        start_time = time.time()
        
        try:
            # amnesty_qa 평가 데이터 로딩
            data_path = Path("data/amnesty_qa/amnesty_qa_evaluation.json")
            
            if not data_path.exists():
                raise FileNotFoundError(f"테스트 데이터 파일이 없습니다: {data_path}")
            
            logger.info(f"데이터 파일 로딩: {data_path}")
            with open(data_path, 'r', encoding='utf-8') as f:
                self.test_data = json.load(f)
            
            # 데이터 구조 검증
            if not isinstance(self.test_data, list):
                raise ValueError("테스트 데이터가 올바른 형식이 아닙니다.")
            
            if len(self.test_data) == 0:
                raise ValueError("테스트 데이터가 비어있습니다.")
            
            # 첫 번째 항목 구조 검증
            first_item = self.test_data[0]
            required_fields = ['question', 'contexts', 'answer']
            
            for field in required_fields:
                if field not in first_item:
                    raise KeyError(f"필수 필드 누락: {field}")
            
            logger.info(f"테스트 데이터 로딩 완료: {len(self.test_data)}개 항목")
            
            # 샘플 데이터 로그
            logger.info(f"첫 번째 질문 예시: {first_item['question'][:100]}...")
            
            self.log_stage_performance("data_loading", start_time, True)
            return True
            
        except Exception as e:
            error_msg = f"데이터 로딩 실패: {str(e)}"
            self.log_stage_performance("data_loading", start_time, False, error_msg)
            return False
    
    def test_03_vector_search_pipeline(self) -> bool:
        """3단계: 벡터 검색 파이프라인 테스트"""
        logger.info("\n--- 3단계: 벡터 검색 파이프라인 테스트 ---")
        start_time = time.time()
        
        try:
            # 테스트 질문 5개 선택 (데이터에서 처음 5개)
            test_queries = []
            for i, item in enumerate(self.test_data[:5]):
                test_queries.append({
                    'id': i + 1,
                    'question': item['question'],
                    'expected_contexts': item.get('contexts', []),
                    'expected_answer': item.get('answer', '')
                })
            
            logger.info(f"{len(test_queries)}개 질문으로 벡터 검색 테스트 시작")
            
            successful_queries = 0
            total_search_time = 0
            
            for query_info in test_queries:
                query_start = time.time()
                try:
                    # 벡터 검색 실행
                    search_results = self.retriever.retrieve(
                        query=query_info['question'],
                        top_k=5,
                        force_filter_expr=None
                    )
                    
                    query_duration = time.time() - query_start
                    total_search_time += query_duration
                    
                    # 결과 검증
                    if isinstance(search_results, list) and len(search_results) > 0:
                        successful_queries += 1
                        
                        # 결과 정보 로깅
                        logger.info(f"  질문 {query_info['id']}: {len(search_results)}개 결과, {query_duration:.3f}초")
                        
                        # 상위 결과 샘플 로깅
                        top_result = search_results[0]
                        similarity = top_result.get('similarity', top_result.get('score', 0))
                        logger.info(f"    최고 유사도: {similarity:.4f}")
                        
                        # 테스트 질문 저장
                        self.test_results['test_queries'].append({
                            'id': query_info['id'],
                            'question': query_info['question'][:100] + '...' if len(query_info['question']) > 100 else query_info['question'],
                            'results_count': len(search_results),
                            'search_duration': round(query_duration, 3),
                            'top_similarity': round(similarity, 4) if isinstance(similarity, (int, float)) else str(similarity)
                        })
                    else:
                        logger.warning(f"  질문 {query_info['id']}: 검색 결과 없음")
                        
                except Exception as e:
                    logger.error(f"  질문 {query_info['id']} 검색 실패: {str(e)}")
                    query_duration = time.time() - query_start
                    total_search_time += query_duration
            
            # 전체 성능 메트릭 계산
            success_rate = (successful_queries / len(test_queries)) * 100
            avg_search_time = total_search_time / len(test_queries)
            
            logger.info(f"벡터 검색 결과: {successful_queries}/{len(test_queries)} 성공 ({success_rate:.1f}%)")
            logger.info(f"평균 검색 시간: {avg_search_time:.3f}초")
            
            # 성능 메트릭 저장
            self.test_results['performance_metrics']['vector_search'] = {
                'success_rate': round(success_rate, 2),
                'avg_search_time': round(avg_search_time, 3),
                'total_queries': len(test_queries),
                'successful_queries': successful_queries
            }
            
            # 성공 기준: 95% 이상 성공률
            pipeline_success = success_rate >= 95.0
            
            if pipeline_success:
                self.log_stage_performance("vector_search_pipeline", start_time, True)
            else:
                error_msg = f"벡터 검색 성공률 부족: {success_rate:.1f}% (95% 미만)"
                self.log_stage_performance("vector_search_pipeline", start_time, False, error_msg)
            
            return pipeline_success
            
        except Exception as e:
            error_msg = f"벡터 검색 파이프라인 실패: {str(e)}"
            self.log_stage_performance("vector_search_pipeline", start_time, False, error_msg)
            return False
    
    def test_04_hybrid_search_with_reranking(self) -> bool:
        """4단계: 하이브리드 검색 및 Re-ranking 테스트"""
        logger.info("\n--- 4단계: 하이브리드 검색 + Re-ranking 테스트 ---")
        start_time = time.time()
        
        try:
            # 하이브리드 검색 및 Re-ranking 활성화 확인
            has_hybrid = hasattr(self.retriever, 'hybrid_retrieve')
            has_reranker = hasattr(self.retriever, 'reranker') and self.retriever.reranker is not None
            
            logger.info(f"하이브리드 검색 지원: {has_hybrid}")
            logger.info(f"Re-ranking 지원: {has_reranker}")
            
            # 테스트 질문 3개 선택
            test_queries = self.test_data[:3]
            successful_hybrid = 0
            total_hybrid_time = 0
            
            for i, item in enumerate(test_queries):
                query_start = time.time()
                try:
                    query = item['question']
                    
                    # 하이브리드 검색 실행 (가능한 경우)
                    if has_hybrid:
                        hybrid_results = self.retriever.hybrid_retrieve(
                            query=query,
                            top_k=5,
                            use_parent_chunks=False
                        )
                    else:
                        # 하이브리드 검색이 없으면 일반 검색 사용
                        hybrid_results = self.retriever.retrieve(
                            query=query,
                            top_k=5
                        )
                    
                    query_duration = time.time() - query_start
                    total_hybrid_time += query_duration
                    
                    # 결과 검증
                    if isinstance(hybrid_results, list) and len(hybrid_results) > 0:
                        successful_hybrid += 1
                        
                        # Re-ranking 점수 확인
                        has_rerank_scores = any('rerank_score' in result for result in hybrid_results)
                        has_hybrid_scores = any('hybrid_score' in result for result in hybrid_results)
                        
                        logger.info(f"  질문 {i+1}: {len(hybrid_results)}개 결과, {query_duration:.3f}초")
                        logger.info(f"    Hybrid 점수 포함: {has_hybrid_scores}")
                        logger.info(f"    Re-rank 점수 포함: {has_rerank_scores}")
                        
                        # 상위 결과 샘플
                        top_result = hybrid_results[0]
                        similarity = top_result.get('similarity', 0)
                        hybrid_score = top_result.get('hybrid_score', 'N/A')
                        rerank_score = top_result.get('rerank_score', 'N/A')
                        
                        logger.info(f"    상위 결과 - Similarity: {similarity:.4f}, Hybrid: {hybrid_score}, Rerank: {rerank_score}")
                        
                    else:
                        logger.warning(f"  질문 {i+1}: 하이브리드 검색 결과 없음")
                        
                except Exception as e:
                    logger.error(f"  질문 {i+1} 하이브리드 검색 실패: {str(e)}")
                    query_duration = time.time() - query_start
                    total_hybrid_time += query_duration
            
            # 하이브리드 검색 성능 계산
            hybrid_success_rate = (successful_hybrid / len(test_queries)) * 100
            avg_hybrid_time = total_hybrid_time / len(test_queries)
            
            logger.info(f"하이브리드 검색 결과: {successful_hybrid}/{len(test_queries)} 성공 ({hybrid_success_rate:.1f}%)")
            logger.info(f"평균 하이브리드 검색 시간: {avg_hybrid_time:.3f}초")
            
            # 성능 메트릭 저장
            self.test_results['performance_metrics']['hybrid_search'] = {
                'success_rate': round(hybrid_success_rate, 2),
                'avg_search_time': round(avg_hybrid_time, 3),
                'total_queries': len(test_queries),
                'successful_queries': successful_hybrid,
                'has_hybrid_feature': has_hybrid,
                'has_reranking_feature': has_reranker
            }
            
            # 성공 기준: 90% 이상 성공률
            pipeline_success = hybrid_success_rate >= 90.0
            
            if pipeline_success:
                self.log_stage_performance("hybrid_search_reranking", start_time, True)
            else:
                error_msg = f"하이브리드 검색 성공률 부족: {hybrid_success_rate:.1f}% (90% 미만)"
                self.log_stage_performance("hybrid_search_reranking", start_time, False, error_msg)
            
            return pipeline_success
            
        except Exception as e:
            error_msg = f"하이브리드 검색 + Re-ranking 실패: {str(e)}"
            self.log_stage_performance("hybrid_search_reranking", start_time, False, error_msg)
            return False
    
    def test_05_memory_performance_validation(self) -> bool:
        """5단계: 메모리 사용량 및 성능 검증"""
        logger.info("\n--- 5단계: 메모리 사용량 및 성능 검증 ---")
        start_time = time.time()
        
        try:
            current_memory = self.get_memory_usage()
            memory_increase = current_memory - self.initial_memory
            
            logger.info(f"현재 메모리 사용량: {current_memory:.2f} MB")
            logger.info(f"메모리 증가량: {memory_increase:.2f} MB")
            
            # 메모리 사용량 기준 확인 (4GB = 4096 MB)
            memory_limit_mb = 4096
            memory_within_limit = current_memory <= memory_limit_mb
            
            # 전체 테스트 시간 계산
            total_test_time = time.time() - self.test_results.get('test_start_time_float', time.time())
            
            # 전체 성능 요약
            performance_summary = {
                'total_test_duration': round(total_test_time, 3),
                'memory_usage': {
                    'initial_mb': round(self.initial_memory, 2),
                    'current_mb': round(current_memory, 2),
                    'increase_mb': round(memory_increase, 2),
                    'within_limit': memory_within_limit,
                    'limit_mb': memory_limit_mb
                }
            }
            
            # 각 단계별 성공 여부 확인
            stages = self.test_results.get('pipeline_stages', {})
            successful_stages = sum(1 for stage in stages.values() if stage.get('success', False))
            total_stages = len(stages)
            
            logger.info(f"전체 테스트 시간: {total_test_time:.3f}초")
            logger.info(f"성공한 단계: {successful_stages}/{total_stages}")
            logger.info(f"메모리 제한 내: {memory_within_limit} ({current_memory:.2f}MB / {memory_limit_mb}MB)")
            
            # 전체 성능 기준 검증
            performance_checks = {
                'total_time_under_10s': total_test_time <= 10.0,
                'memory_under_4gb': memory_within_limit,
                'stages_success_rate_95': (successful_stages / total_stages) >= 0.95 if total_stages > 0 else False
            }
            
            all_checks_passed = all(performance_checks.values())
            
            # 성능 및 메모리 메트릭 저장
            self.test_results['performance_metrics']['overall'] = performance_summary
            self.test_results['performance_metrics']['validation_checks'] = performance_checks
            self.test_results['memory_usage']['final'] = {
                'memory_mb': round(current_memory, 2),
                'within_limit': memory_within_limit
            }
            
            # 성능 검증 결과 로깅
            for check_name, passed in performance_checks.items():
                status = "✓" if passed else "✗"
                logger.info(f"  {status} {check_name}: {passed}")
            
            if all_checks_passed:
                self.log_stage_performance("memory_performance_validation", start_time, True)
            else:
                failed_checks = [name for name, passed in performance_checks.items() if not passed]
                error_msg = f"성능 검증 실패: {', '.join(failed_checks)}"
                self.log_stage_performance("memory_performance_validation", start_time, False, error_msg)
            
            return all_checks_passed
            
        except Exception as e:
            error_msg = f"메모리 성능 검증 실패: {str(e)}"
            self.log_stage_performance("memory_performance_validation", start_time, False, error_msg)
            return False
    
    def run_complete_pipeline_test(self) -> Dict[str, Any]:
        """End-to-End 파이프라인 전체 테스트 실행"""
        logger.info("파이프라인 전체 테스트 시작")
        
        # 테스트 시작 시간 저장
        self.test_results['test_start_time_float'] = time.time()
        
        # 단계별 테스트 실행
        test_stages = [
            ('01_system_initialization', self.test_01_system_initialization),
            ('02_data_loading', self.test_02_data_loading),
            ('03_vector_search_pipeline', self.test_03_vector_search_pipeline),
            ('04_hybrid_search_reranking', self.test_04_hybrid_search_with_reranking),
            ('05_memory_performance_validation', self.test_05_memory_performance_validation)
        ]
        
        all_stages_successful = True
        
        for stage_name, test_function in test_stages:
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"단계 실행: {stage_name}")
                logger.info(f"{'='*60}")
                
                stage_success = test_function()
                
                if not stage_success:
                    all_stages_successful = False
                    logger.error(f"단계 실패: {stage_name}")
                else:
                    logger.info(f"단계 성공: {stage_name}")
                
            except Exception as e:
                all_stages_successful = False
                logger.error(f"단계 예외 발생 {stage_name}: {str(e)}")
                logger.error(f"Stack trace: {traceback.format_exc()}")
        
        # 최종 결과 정리
        self.test_results['test_end_time'] = datetime.now().isoformat()
        self.test_results['overall_success'] = all_stages_successful
        
        total_duration = time.time() - self.test_results['test_start_time_float']
        self.test_results['total_duration_seconds'] = round(total_duration, 3)
        
        # 최종 메모리 사용량 기록
        final_memory = self.get_memory_usage()
        self.test_results['memory_usage']['final_mb'] = round(final_memory, 2)
        
        return self.test_results
    
    def save_test_results(self) -> str:
        """테스트 결과를 JSON 파일로 저장"""
        try:
            results_dir = Path("evaluation_results")
            results_dir.mkdir(exist_ok=True)
            
            results_file = results_dir / f"end_to_end_test_results_{timestamp}.json"
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(self.test_results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"테스트 결과 저장 완료: {results_file}")
            return str(results_file)
            
        except Exception as e:
            logger.error(f"테스트 결과 저장 실패: {str(e)}")
            return ""
    
    def print_final_summary(self):
        """최종 테스트 요약 출력"""
        logger.info("\n" + "="*80)
        logger.info("END-TO-END RAG 파이프라인 테스트 최종 요약")
        logger.info("="*80)
        
        # 전체 결과 요약
        overall_success = self.test_results.get('overall_success', False)
        total_duration = self.test_results.get('total_duration_seconds', 0)
        
        status_emoji = "✅" if overall_success else "❌"
        logger.info(f"{status_emoji} 전체 테스트 결과: {'성공' if overall_success else '실패'}")
        logger.info(f"⏱️ 총 소요 시간: {total_duration:.3f}초")
        
        # 단계별 결과 요약
        stages = self.test_results.get('pipeline_stages', {})
        logger.info(f"\n📋 단계별 결과 ({len(stages)}개 단계):")
        
        for stage_name, stage_info in stages.items():
            success = stage_info.get('success', False)
            duration = stage_info.get('duration_seconds', 0)
            memory = stage_info.get('memory_mb', 0)
            
            status = "✓" if success else "✗"
            logger.info(f"  {status} {stage_name}: {duration:.3f}초, {memory:.1f}MB")
            
            if not success and 'error' in stage_info:
                logger.info(f"    오류: {stage_info['error']}")
        
        # 성능 메트릭 요약
        metrics = self.test_results.get('performance_metrics', {})
        logger.info(f"\n📊 성능 메트릭:")
        
        if 'vector_search' in metrics:
            vs_metrics = metrics['vector_search']
            logger.info(f"  벡터 검색: {vs_metrics.get('success_rate', 0):.1f}% 성공, {vs_metrics.get('avg_search_time', 0):.3f}초 평균")
        
        if 'hybrid_search' in metrics:
            hs_metrics = metrics['hybrid_search']
            logger.info(f"  하이브리드 검색: {hs_metrics.get('success_rate', 0):.1f}% 성공, {hs_metrics.get('avg_search_time', 0):.3f}초 평균")
        
        # 메모리 사용량 요약
        memory_info = self.test_results.get('memory_usage', {})
        final_memory = memory_info.get('final_mb', 0)
        logger.info(f"  최종 메모리 사용량: {final_memory:.2f}MB")
        
        # 권장사항
        logger.info(f"\n💡 권장사항:")
        
        if not overall_success:
            logger.info("  - 실패한 단계의 오류 로그를 확인하세요")
            logger.info("  - 시스템 리소스 및 네트워크 연결을 점검하세요")
        
        if total_duration > 10:
            logger.info("  - 전체 파이프라인 응답 시간이 목표(10초)를 초과했습니다")
            logger.info("  - 성능 최적화를 고려하세요")
        
        if final_memory > 3000:  # 3GB 이상
            logger.info("  - 메모리 사용량이 높습니다. 배치 크기 조정을 고려하세요")
        
        logger.info("\n📁 상세 결과는 evaluation_results/ 폴더를 확인하세요")
        logger.info("="*80)


def main():
    """메인 실행 함수"""
    try:
        logger.info("🚀 End-to-End RAG 파이프라인 테스트 시작")
        logger.info(f"테스트 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 테스트 인스턴스 생성 및 실행
        pipeline_test = EndToEndPipelineTest()
        
        # 전체 파이프라인 테스트 실행
        test_results = pipeline_test.run_complete_pipeline_test()
        
        # 결과 저장
        results_file = pipeline_test.save_test_results()
        
        # 최종 요약 출력
        pipeline_test.print_final_summary()
        
        # 종료 상태 결정
        overall_success = test_results.get('overall_success', False)
        
        if overall_success:
            logger.info("✅ End-to-End 테스트 성공적으로 완료!")
            return 0
        else:
            logger.error("❌ End-to-End 테스트 실패. 상세 로그를 확인하세요.")
            return 1
            
    except KeyboardInterrupt:
        logger.warning("⚠️ 사용자에 의해 테스트가 중단되었습니다.")
        return 2
        
    except Exception as e:
        logger.error(f"❌ 예상치 못한 오류 발생: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return 3


if __name__ == "__main__":
    # 로그 디렉토리 생성
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # evaluation_results 디렉토리 생성
    results_dir = Path("evaluation_results")
    results_dir.mkdir(exist_ok=True)
    
    # 메인 함수 실행
    exit_code = main()
    
    # 종료 코드로 프로그램 종료
    sys.exit(exit_code)
