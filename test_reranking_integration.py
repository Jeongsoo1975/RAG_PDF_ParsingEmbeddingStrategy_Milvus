#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Re-ranking 기능 통합 검증 테스트
retriever.py와 hybrid_retrieve 메서드에 통합된 Re-ranking 기능이 정상적으로 작동하는지 확인하는 기본 테스트
"""

import os
import sys
import logging
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# 프로젝트 루트 경로 추가
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# 환경 변수 설정 (테스트용)
os.environ.setdefault('LOG_LEVEL', 'INFO')

try:
    from src.utils.config import Config
    from src.rag.retriever import DocumentRetriever
    from src.rag.reranker import DocumentReranker
    from src.utils.logger import get_logger
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you're running from the project root directory")
    sys.exit(1)

logger = get_logger("test_reranking_integration")

class TestReRankingIntegration(unittest.TestCase):
    """Re-ranking 기능 통합 테스트 클래스"""
    
    def setUp(self):
        """테스트 환경 설정"""
        self.config = Config()
        
        # 테스트용 설정 수정
        self.config.retrieval = {
            'top_k': 5,
            'similarity_threshold': 0.3,
            'reranking': True,
            'reranking_batch_size': 4,
            'reranking_models': [
                "jhgan/ko-sroberta-multitask",
                "cross-encoder/ms-marco-MiniLM-L-6-v2"
            ],
            'offline_mode': True,  # 오프라인 모드로 테스트
            'hybrid_search': True,
            'hybrid_alpha': 0.7
        }
        
        logger.info("=== Re-ranking 통합 테스트 시작 ===")
    
    def test_01_document_reranker_initialization(self):
        """1. DocumentReranker 초기화 테스트"""
        logger.info("테스트 1: DocumentReranker 초기화 테스트")
        
        try:
            reranker = DocumentReranker(self.config)
            
            # 기본 속성 확인
            self.assertIsNotNone(reranker)
            self.assertEqual(reranker.reranking_enabled, True)
            self.assertEqual(reranker.reranking_batch_size, 4)
            self.assertIsInstance(reranker.reranking_models, list)
            self.assertGreater(len(reranker.reranking_models), 0)
            
            logger.info("✓ DocumentReranker 초기화 성공")
            logger.info(f"  - Re-ranking 활성화: {reranker.reranking_enabled}")
            logger.info(f"  - 배치 크기: {reranker.reranking_batch_size}")
            logger.info(f"  - 사용 가능한 모델: {reranker.reranking_models}")
            
        except Exception as e:
            logger.error(f"✗ DocumentReranker 초기화 실패: {e}")
            self.fail(f"DocumentReranker 초기화 실패: {e}")
    
    def test_02_retriever_with_reranking_offline(self):
        """2. retrieve 메서드에서 Re-ranking 적용 테스트 (오프라인 모드)"""
        logger.info("테스트 2: retrieve 메서드 Re-ranking 테스트 (오프라인)")
        
        try:
            retriever = DocumentRetriever(self.config)
            
            # 오프라인 모드 확인
            self.assertTrue(retriever.offline_mode)
            self.assertIsNotNone(retriever.reranker)
            
            # 테스트 쿼리 실행
            test_query = "CI보험금 선지급서비스특약"
            results_with_reranking = retriever.retrieve(
                query=test_query,
                top_k=3,
                use_parent_chunks=False
            )
            
            # 결과 검증
            self.assertIsInstance(results_with_reranking, list)
            self.assertGreater(len(results_with_reranking), 0)
            
            # Re-ranking 점수 확인
            has_rerank_scores = any('rerank_score' in result for result in results_with_reranking)
            
            logger.info(f"✓ retrieve 메서드 실행 성공")
            logger.info(f"  - 쿼리: '{test_query}'")
            logger.info(f"  - 결과 수: {len(results_with_reranking)}")
            logger.info(f"  - Re-rank 점수 포함: {has_rerank_scores}")
            
            # 결과 상세 로그
            for i, result in enumerate(results_with_reranking[:2]):  # 상위 2개만 로그
                logger.info(f"  - 결과 {i+1}: similarity={result.get('similarity', 0):.4f}, "
                           f"rerank_score={result.get('rerank_score', 'N/A')}")
            
        except Exception as e:
            logger.error(f"✗ retrieve 메서드 Re-ranking 테스트 실패: {e}")
            self.fail(f"retrieve 메서드 Re-ranking 테스트 실패: {e}")
    
    def test_03_hybrid_retrieve_with_reranking(self):
        """3. hybrid_retrieve 메서드에서 Re-ranking 적용 테스트"""
        logger.info("테스트 3: hybrid_retrieve 메서드 Re-ranking 테스트")
        
        try:
            retriever = DocumentRetriever(self.config)
            
            # 하이브리드 검색 활성화 확인
            self.assertTrue(retriever.hybrid_search_enabled_by_default or 
                          os.environ.get("FORCE_HYBRID_SEARCH"))
            
            # 테스트 쿼리 실행
            test_query = "무배당 정기특약 보험료 납입"
            results_hybrid = retriever.hybrid_retrieve(
                query=test_query,
                top_k=3,
                use_parent_chunks=False
            )
            
            # 결과 검증
            self.assertIsInstance(results_hybrid, list)
            self.assertGreater(len(results_hybrid), 0)
            
            # 하이브리드 및 Re-ranking 점수 확인
            has_hybrid_scores = any('hybrid_score' in result for result in results_hybrid)
            has_rerank_scores = any('rerank_score' in result for result in results_hybrid)
            
            logger.info(f"✓ hybrid_retrieve 메서드 실행 성공")
            logger.info(f"  - 쿼리: '{test_query}'")
            logger.info(f"  - 결과 수: {len(results_hybrid)}")
            logger.info(f"  - Hybrid 점수 포함: {has_hybrid_scores}")
            logger.info(f"  - Re-rank 점수 포함: {has_rerank_scores}")
            
            # 결과 상세 로그
            for i, result in enumerate(results_hybrid[:2]):  # 상위 2개만 로그
                hybrid_score = result.get('hybrid_score', 'N/A')
                rerank_score = result.get('rerank_score', 'N/A')
                hybrid_str = f"{hybrid_score:.4f}" if isinstance(hybrid_score, (int, float)) else str(hybrid_score)
                rerank_str = f"{rerank_score:.4f}" if isinstance(rerank_score, (int, float)) else str(rerank_score)
                logger.info(f"  - 결과 {i+1}: hybrid_score={hybrid_str}, rerank_score={rerank_str}")
            
        except Exception as e:
            logger.error(f"✗ hybrid_retrieve 메서드 Re-ranking 테스트 실패: {e}")
            self.fail(f"hybrid_retrieve 메서드 Re-ranking 테스트 실패: {e}")
    
    def test_04_reranking_fallback_test(self):
        """4. Re-ranking 실패 시 fallback 테스트"""
        logger.info("테스트 4: Re-ranking 실패 시 fallback 테스트")
        
        try:
            # Re-ranking 비활성화 설정으로 테스트
            config_no_rerank = Config()
            config_no_rerank.retrieval = {
                'top_k': 3,
                'similarity_threshold': 0.3,
                'reranking': False,  # Re-ranking 비활성화
                'offline_mode': True
            }
            
            retriever = DocumentRetriever(config_no_rerank)
            
            # Re-ranking이 비활성화되었는지 확인
            self.assertIsNone(retriever.reranker)
            
            # 쿼리 실행 (Re-ranking 없이)
            test_query = "제1보험기간"
            results_no_rerank = retriever.retrieve(
                query=test_query,
                top_k=3
            )
            
            # 결과 검증
            self.assertIsInstance(results_no_rerank, list)
            self.assertGreater(len(results_no_rerank), 0)
            
            # Re-ranking 점수가 없는지 확인
            has_rerank_scores = any('rerank_score' in result for result in results_no_rerank)
            self.assertFalse(has_rerank_scores)
            
            logger.info(f"✓ Re-ranking 비활성화 상태 테스트 성공")
            logger.info(f"  - 쿼리: '{test_query}'")
            logger.info(f"  - 결과 수: {len(results_no_rerank)}")
            logger.info(f"  - Re-rank 점수 포함: {has_rerank_scores} (예상: False)")
            
        except Exception as e:
            logger.error(f"✗ Re-ranking fallback 테스트 실패: {e}")
            self.fail(f"Re-ranking fallback 테스트 실패: {e}")
    
    def test_05_offline_fallback_reranking(self):
        """5. 오프라인 모드에서 fallback Re-ranking 테스트"""
        logger.info("테스트 5: 오프라인 모드 fallback Re-ranking 테스트")
        
        try:
            retriever = DocumentRetriever(self.config)
            reranker = retriever.reranker
            
            # 오프라인 모드 확인
            self.assertTrue(retriever.offline_mode)
            self.assertIsNotNone(reranker)
            
            # 샘플 검색 결과 생성
            sample_results = [
                {
                    'id': 'test_1',
                    'content': 'CI보험금은 중대한 질병 발생 시 지급되는 보험금입니다.',
                    'similarity': 0.8,
                    'collection': 'test_collection'
                },
                {
                    'id': 'test_2', 
                    'content': '선지급서비스특약은 여명이 6개월 이내일 때 적용됩니다.',
                    'similarity': 0.7,
                    'collection': 'test_collection'
                },
                {
                    'id': 'test_3',
                    'content': '보험료 납입이 면제되는 경우가 있습니다.',
                    'similarity': 0.6,
                    'collection': 'test_collection'
                }
            ]
            
            # Re-ranking 적용
            test_query = "CI보험금 선지급"
            reranked_results = reranker.rerank_results(
                query=test_query,
                search_results=sample_results,
                top_k=3
            )
            
            # 결과 검증
            self.assertIsInstance(reranked_results, list)
            self.assertEqual(len(reranked_results), 3)
            
            # Re-ranking 점수나 원본 점수 확인
            for result in reranked_results:
                self.assertIn('similarity', result)
                # fallback 모드에서는 rerank_score가 있을 수도 없을 수도 있음
            
            logger.info(f"✓ 오프라인 fallback Re-ranking 테스트 성공")
            logger.info(f"  - 쿼리: '{test_query}'")
            logger.info(f"  - 원본 결과 수: {len(sample_results)}")
            logger.info(f"  - Re-ranking 결과 수: {len(reranked_results)}")
            
            # 순위 변화 로그
            for i, result in enumerate(reranked_results):
                original_rank = next((j for j, r in enumerate(sample_results) if r['id'] == result['id']), -1)
                rank_change = "순위 유지" if original_rank == i else f"{original_rank+1} → {i+1}"
                logger.info(f"  - {result['id']}: {rank_change}, "
                           f"similarity={result.get('similarity', 0):.3f}")
            
        except Exception as e:
            logger.error(f"✗ 오프라인 fallback Re-ranking 테스트 실패: {e}")
            self.fail(f"오프라인 fallback Re-ranking 테스트 실패: {e}")
    
    def test_06_reranking_comparison(self):
        """6. Re-ranking 적용 전후 결과 비교 테스트"""
        logger.info("테스트 6: Re-ranking 적용 전후 결과 비교")
        
        try:
            # Re-ranking 활성화된 retriever
            retriever_with_rerank = DocumentRetriever(self.config)
            
            # Re-ranking 비활성화된 retriever
            config_no_rerank = Config()
            config_no_rerank.retrieval = dict(self.config.retrieval)
            config_no_rerank.retrieval['reranking'] = False
            retriever_no_rerank = DocumentRetriever(config_no_rerank)
            
            test_query = "제18조 CI보험금"
            
            # 두 방식으로 검색 실행
            results_with_rerank = retriever_with_rerank.retrieve(query=test_query, top_k=3)
            results_no_rerank = retriever_no_rerank.retrieve(query=test_query, top_k=3)
            
            # 결과 비교
            self.assertIsInstance(results_with_rerank, list)
            self.assertIsInstance(results_no_rerank, list)
            self.assertGreater(len(results_with_rerank), 0)
            self.assertGreater(len(results_no_rerank), 0)
            
            logger.info(f"✓ Re-ranking 적용 전후 비교 완료")
            logger.info(f"  - 쿼리: '{test_query}'")
            logger.info(f"  - Re-ranking 적용: {len(results_with_rerank)}개 결과")
            logger.info(f"  - Re-ranking 미적용: {len(results_no_rerank)}개 결과")
            
            # 상위 결과 ID 비교
            if results_with_rerank and results_no_rerank:
                top_id_with = results_with_rerank[0].get('id', 'unknown')
                top_id_without = results_no_rerank[0].get('id', 'unknown')
                
                rank_changed = top_id_with != top_id_without
                logger.info(f"  - 1위 결과 변화: {rank_changed} (With: {top_id_with}, Without: {top_id_without})")
                
                # 전체 순위 변화 분석
                if len(results_with_rerank) >= 2 and len(results_no_rerank) >= 2:
                    ids_with = [r.get('id') for r in results_with_rerank[:3]]
                    ids_without = [r.get('id') for r in results_no_rerank[:3]]
                    
                    rank_changes = 0
                    for i, id_with in enumerate(ids_with):
                        if i < len(ids_without) and id_with != ids_without[i]:
                            rank_changes += 1
                    
                    logger.info(f"  - 상위 3개 중 순위 변화: {rank_changes}/3")
            
        except Exception as e:
            logger.error(f"✗ Re-ranking 비교 테스트 실패: {e}")
            self.fail(f"Re-ranking 비교 테스트 실패: {e}")
    
    def tearDown(self):
        """테스트 정리"""
        logger.info("=== Re-ranking 통합 테스트 완료 ===\n")


def main():
    """테스트 실행 함수"""
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/test_reranking_integration.log', encoding='utf-8')
        ]
    )
    
    # 로그 디렉토리 생성
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    logger.info("Re-ranking 통합 검증 테스트 시작")
    
    # 테스트 실행
    unittest.main(verbosity=2, exit=False)
    
    logger.info("Re-ranking 통합 검증 테스트 완료")


if __name__ == '__main__':
    main()
