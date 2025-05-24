#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Re-ranking 성능 비교 평가 테스트
Re-ranking 적용 전후의 검색 성능을 정량적으로 비교 분석하여 효과를 검증합니다.
"""

import os
import sys
import time
import json
import logging
import statistics
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from unittest.mock import patch, MagicMock

# 프로젝트 루트 경로 추가
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# 환경 변수 설정
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

logger = get_logger("test_reranking_performance")

class ReRankingPerformanceEvaluator:
    """Re-ranking 성능 비교 평가 클래스"""
    
    def __init__(self, config: Optional[Config] = None):
        """초기화"""
        self.config = config or Config()
        self.logger = logger
        
        # 결과 저장 경로
        self.results_dir = Path("evaluation_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # 테스트 쿼리 (한국어 보험 관련)
        self.test_queries = [
            "CI보험금이란 무엇인가요?",
            "선지급서비스특약에 대해 설명해주세요",
            "제1보험기간과 제2보험기간의 차이점은?",
            "무배당 정기특약 보험료 납입 면제 조건",
            "제18조에 명시된 보험금의 종류",
            "보험료 납입 면제 사유",
            "CI보험금과 선지급서비스특약의 차이",
            "중대한 질병에 해당하는 조건",
            "보험계약 해지 시 환급금",
            "보장개시일은 언제부터인가요?"
        ]
        
        # 성능 메트릭 저장소
        self.performance_metrics = {
            "with_reranking": {},
            "without_reranking": {},
            "comparison": {}
        }
        
    def setup_retrievers(self) -> Tuple[DocumentRetriever, DocumentRetriever]:
        """Re-ranking 적용/미적용 retriever 설정"""
        self.logger.info("Setting up retrievers for performance comparison...")
        
        # Re-ranking 활성화된 retriever
        config_with_rerank = Config()
        config_with_rerank.retrieval = {
            'top_k': 5,
            'similarity_threshold': 0.3,
            'reranking': True,
            'reranking_batch_size': 8,
            'offline_mode': True,  # 일관된 테스트를 위해 오프라인 모드 사용
            'hybrid_search': False  # 순수 vector + reranking 테스트
        }
        
        # Re-ranking 비활성화된 retriever
        config_without_rerank = Config()
        config_without_rerank.retrieval = {
            'top_k': 5,
            'similarity_threshold': 0.3,
            'reranking': False,
            'offline_mode': True,
            'hybrid_search': False
        }
        
        retriever_with_rerank = DocumentRetriever(config_with_rerank)
        retriever_without_rerank = DocumentRetriever(config_without_rerank)
        
        return retriever_with_rerank, retriever_without_rerank
    
    def measure_search_performance(self, retriever: DocumentRetriever, query: str, 
                                  test_name: str) -> Dict[str, Any]:
        """단일 쿼리에 대한 검색 성능 측정"""
        start_time = time.time()
        
        try:
            # 검색 실행
            results = retriever.retrieve(
                query=query,
                top_k=5,
                use_parent_chunks=False,
                enable_query_optimization=True
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # 결과 분석
            result_analysis = {
                "query": query,
                "test_name": test_name,
                "execution_time": execution_time,
                "result_count": len(results),
                "results": []
            }
            
            # 각 결과에 대한 상세 정보 추출
            for i, result in enumerate(results):
                result_info = {
                    "rank": i + 1,
                    "id": result.get("id", f"unknown_{i}"),
                    "similarity": result.get("similarity", 0.0),
                    "rerank_score": result.get("rerank_score", None),
                    "hybrid_score": result.get("hybrid_score", None),
                    "content_preview": result.get("content", "")[:100] + "...",
                    "chunk_type": result.get("metadata", {}).get("chunk_type", "unknown")
                }
                result_analysis["results"].append(result_info)
            
            self.logger.debug(f"{test_name} - Query: '{query}' completed in {execution_time:.4f}s")
            return result_analysis
            
        except Exception as e:
            self.logger.error(f"Error in {test_name} for query '{query}': {e}")
            return {
                "query": query,
                "test_name": test_name,
                "execution_time": -1,
                "error": str(e),
                "result_count": 0,
                "results": []
            }
    
    def compare_ranking_changes(self, results_with: List[Dict], results_without: List[Dict]) -> Dict[str, Any]:
        """랜킹 변화 분석"""
        if not results_with or not results_without:
            return {"error": "One or both result sets are empty"}
        
        # ID 기반 랜킹 비교
        ids_with = [r.get("id") for r in results_with]
        ids_without = [r.get("id") for r in results_without]
        
        # 순위 변화 계산
        rank_changes = []
        position_improvements = 0
        position_degradations = 0
        
        for i, id_with in enumerate(ids_with):
            if id_with in ids_without:
                old_rank = ids_without.index(id_with) + 1
                new_rank = i + 1
                rank_change = old_rank - new_rank  # 양수면 순위 상승
                
                rank_changes.append({
                    "id": id_with,
                    "old_rank": old_rank,
                    "new_rank": new_rank,
                    "rank_change": rank_change
                })
                
                if rank_change > 0:
                    position_improvements += 1
                elif rank_change < 0:
                    position_degradations += 1
        
        # 새로 나타난 결과 (Re-ranking으로 인해 대체된 결과)
        new_results = [id_with for id_with in ids_with if id_with not in ids_without]
        
        # 사라진 결과
        removed_results = [id_without for id_without in ids_without if id_without not in ids_with]
        
        return {
            "total_compared": len(rank_changes),
            "position_improvements": position_improvements,
            "position_degradations": position_degradations,
            "position_unchanged": len(rank_changes) - position_improvements - position_degradations,
            "new_results_count": len(new_results),
            "removed_results_count": len(removed_results),
            "rank_changes": rank_changes,
            "new_results": new_results,
            "removed_results": removed_results,
            "improvement_rate": position_improvements / len(rank_changes) if rank_changes else 0.0
        }
    
    def run_performance_comparison(self) -> Dict[str, Any]:
        """전체 성능 비교 테스트 실행"""
        self.logger.info("=== Re-ranking 성능 비교 평가 시작 ===\n")
        
        # Retriever 설정
        retriever_with_rerank, retriever_without_rerank = self.setup_retrievers()
        
        # 각 쿼리별 성능 비교
        query_results = []
        overall_stats = {
            "total_queries": len(self.test_queries),
            "successful_comparisons": 0,
            "failed_comparisons": 0,
            "execution_times": {"with_reranking": [], "without_reranking": []},
            "improvement_rates": [],
            "position_changes": {"improvements": 0, "degradations": 0, "unchanged": 0}
        }
        
        for i, query in enumerate(self.test_queries, 1):
            self.logger.info(f"\n[{i}/{len(self.test_queries)}] 테스트 쿼리: '{query}'")
            
            # Re-ranking 적용한 검색
            self.logger.info("  → Re-ranking 적용 검색 실행...")
            results_with = self.measure_search_performance(
                retriever_with_rerank, query, "WITH_RERANKING"
            )
            
            # Re-ranking 미적용 검색
            self.logger.info("  → Re-ranking 미적용 검색 실행...")
            results_without = self.measure_search_performance(
                retriever_without_rerank, query, "WITHOUT_RERANKING"
            )
            
            # 결과 비교 분석
            if (results_with.get("execution_time", -1) > 0 and 
                results_without.get("execution_time", -1) > 0 and
                results_with.get("result_count", 0) > 0 and
                results_without.get("result_count", 0) > 0):
                
                # 랭킹 변화 분석
                ranking_comparison = self.compare_ranking_changes(
                    results_with["results"], results_without["results"]
                )
                
                # 실행 시간 비교
                time_with = results_with["execution_time"]
                time_without = results_without["execution_time"]
                time_overhead = time_with - time_without
                time_ratio = time_with / time_without if time_without > 0 else float('inf')
                
                query_result = {
                    "query": query,
                    "query_index": i,
                    "results_with_reranking": results_with,
                    "results_without_reranking": results_without,
                    "ranking_comparison": ranking_comparison,
                    "performance_metrics": {
                        "execution_time_with": time_with,
                        "execution_time_without": time_without,
                        "time_overhead": time_overhead,
                        "time_ratio": time_ratio,
                        "improvement_rate": ranking_comparison.get("improvement_rate", 0.0)
                    }
                }
                
                # 통계 업데이트
                overall_stats["successful_comparisons"] += 1
                overall_stats["execution_times"]["with_reranking"].append(time_with)
                overall_stats["execution_times"]["without_reranking"].append(time_without)
                overall_stats["improvement_rates"].append(ranking_comparison.get("improvement_rate", 0.0))
                
                # 위치 변화 통계
                overall_stats["position_changes"]["improvements"] += ranking_comparison.get("position_improvements", 0)
                overall_stats["position_changes"]["degradations"] += ranking_comparison.get("position_degradations", 0)
                overall_stats["position_changes"]["unchanged"] += ranking_comparison.get("position_unchanged", 0)
                
                # 로그 출력
                self.logger.info(f"  ✓ 실행 시간: {time_with:.4f}s (Re-ranking) vs {time_without:.4f}s (기본)")
                self.logger.info(f"  ✓ 시간 오버헤드: +{time_overhead:.4f}s ({time_ratio:.2f}배)")
                self.logger.info(f"  ✓ 순위 개선율: {ranking_comparison.get('improvement_rate', 0.0):.2%}")
                self.logger.info(f"  ✓ 순위 변화: 개선 {ranking_comparison.get('position_improvements', 0)}, "
                                f"하락 {ranking_comparison.get('position_degradations', 0)}, "
                                f"유지 {ranking_comparison.get('position_unchanged', 0)}")
                
            else:
                query_result = {
                    "query": query,
                    "query_index": i,
                    "error": "One or both searches failed",
                    "results_with_reranking": results_with,
                    "results_without_reranking": results_without
                }
                overall_stats["failed_comparisons"] += 1
                self.logger.warning(f"  ✗ 쿼리 '{query}' 비교 실패")
            
            query_results.append(query_result)
        
        # 전체 통계 계산
        overall_summary = self.calculate_overall_statistics(overall_stats)
        
        return {
            "test_info": {
                "test_name": "Re-ranking Performance Comparison",
                "timestamp": datetime.now().isoformat(),
                "total_queries": len(self.test_queries),
                "test_queries": self.test_queries
            },
            "query_results": query_results,
            "overall_statistics": overall_stats,
            "overall_summary": overall_summary
        }
    
    def calculate_overall_statistics(self, overall_stats: Dict[str, Any]) -> Dict[str, Any]:
        """전체 통계 계산"""
        summary = {}
        
        # 성공률
        total_queries = overall_stats["total_queries"]
        successful = overall_stats["successful_comparisons"]
        failed = overall_stats["failed_comparisons"]
        
        summary["success_rate"] = successful / total_queries if total_queries > 0 else 0.0
        summary["failure_rate"] = failed / total_queries if total_queries > 0 else 0.0
        
        # 실행 시간 통계
        times_with = overall_stats["execution_times"]["with_reranking"]
        times_without = overall_stats["execution_times"]["without_reranking"]
        
        if times_with and times_without:
            summary["avg_time_with_reranking"] = statistics.mean(times_with)
            summary["avg_time_without_reranking"] = statistics.mean(times_without)
            summary["median_time_with_reranking"] = statistics.median(times_with)
            summary["median_time_without_reranking"] = statistics.median(times_without)
            
            summary["avg_time_overhead"] = summary["avg_time_with_reranking"] - summary["avg_time_without_reranking"]
            summary["avg_time_ratio"] = summary["avg_time_with_reranking"] / summary["avg_time_without_reranking"]
            
            # 시간 오버헤드 백분율
            summary["time_overhead_percentage"] = (summary["avg_time_overhead"] / summary["avg_time_without_reranking"]) * 100
        else:
            summary.update({
                "avg_time_with_reranking": 0.0,
                "avg_time_without_reranking": 0.0,
                "median_time_with_reranking": 0.0,
                "median_time_without_reranking": 0.0,
                "avg_time_overhead": 0.0,
                "avg_time_ratio": 0.0,
                "time_overhead_percentage": 0.0
            })
        
        # 순위 개선 통계
        improvement_rates = overall_stats["improvement_rates"]
        if improvement_rates:
            summary["avg_improvement_rate"] = statistics.mean(improvement_rates)
            summary["median_improvement_rate"] = statistics.median(improvement_rates)
            summary["max_improvement_rate"] = max(improvement_rates)
            summary["min_improvement_rate"] = min(improvement_rates)
        else:
            summary.update({
                "avg_improvement_rate": 0.0,
                "median_improvement_rate": 0.0,
                "max_improvement_rate": 0.0,
                "min_improvement_rate": 0.0
            })
        
        # 순위 변화 통계
        position_changes = overall_stats["position_changes"]
        total_positions = sum(position_changes.values())
        
        if total_positions > 0:
            summary["position_improvement_rate"] = position_changes["improvements"] / total_positions
            summary["position_degradation_rate"] = position_changes["degradations"] / total_positions
            summary["position_unchanged_rate"] = position_changes["unchanged"] / total_positions
        else:
            summary.update({
                "position_improvement_rate": 0.0,
                "position_degradation_rate": 0.0,
                "position_unchanged_rate": 0.0
            })
        
        # 성능 평가
        summary["performance_verdict"] = self.evaluate_performance(summary)
        
        return summary
    
    def evaluate_performance(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """성능 평가 및 결론 도출"""
        verdict = {
            "overall_rating": "Unknown",
            "improvement_assessment": "Unknown",
            "time_overhead_assessment": "Unknown",
            "recommendation": "More testing needed"
        }
        
        # 순위 개선 평가 (기준: 10% 이상)
        avg_improvement = summary.get("avg_improvement_rate", 0.0)
        if avg_improvement >= 0.15:  # 15% 이상
            verdict["improvement_assessment"] = "Excellent"
        elif avg_improvement >= 0.10:  # 10% 이상
            verdict["improvement_assessment"] = "Good"
        elif avg_improvement >= 0.05:  # 5% 이상
            verdict["improvement_assessment"] = "Moderate"
        else:
            verdict["improvement_assessment"] = "Poor"
        
        # 시간 오버헤드 평가 (기준: 2배 이내)
        time_ratio = summary.get("avg_time_ratio", 1.0)
        if time_ratio <= 1.5:  # 1.5배 이내
            verdict["time_overhead_assessment"] = "Excellent"
        elif time_ratio <= 2.0:  # 2배 이내
            verdict["time_overhead_assessment"] = "Good"
        elif time_ratio <= 3.0:  # 3배 이내
            verdict["time_overhead_assessment"] = "Moderate"
        else:
            verdict["time_overhead_assessment"] = "Poor"
        
        # 종합 평가
        improvement_score = {"Excellent": 4, "Good": 3, "Moderate": 2, "Poor": 1}[verdict["improvement_assessment"]]
        time_score = {"Excellent": 4, "Good": 3, "Moderate": 2, "Poor": 1}[verdict["time_overhead_assessment"]]
        
        overall_score = (improvement_score * 0.7) + (time_score * 0.3)  # 개선에 더 높은 가중치
        
        if overall_score >= 3.5:
            verdict["overall_rating"] = "Excellent"
            verdict["recommendation"] = "Re-ranking 기능을 활성화하는 것을 강력히 추천합니다."
        elif overall_score >= 2.5:
            verdict["overall_rating"] = "Good"
            verdict["recommendation"] = "Re-ranking 기능을 활성화하는 것을 추천합니다."
        elif overall_score >= 1.5:
            verdict["overall_rating"] = "Moderate"
            verdict["recommendation"] = "Re-ranking 기능의 설정을 최적화한 후 사용을 고려하세요."
        else:
            verdict["overall_rating"] = "Poor"
            verdict["recommendation"] = "Re-ranking 기능의 성능 개선이 필요합니다."
        
        return verdict
    
    def save_results(self, results: Dict[str, Any]) -> str:
        """결과를 JSON 파일로 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"reranking_performance_{timestamp}.json"
        filepath = self.results_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            
            self.logger.info(f"✓ 성능 비교 결과 저장 완료: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"✗ 결과 저장 실패: {e}")
            return ""
    
    def print_summary_report(self, results: Dict[str, Any]):
        """요약 보고서 출력"""
        summary = results.get("overall_summary", {})
        verdict = summary.get("performance_verdict", {})
        
        self.logger.info("\n" + "="*60)
        self.logger.info("         Re-ranking 성능 비교 평가 결과")
        self.logger.info("="*60)
        
        # 기본 정보
        test_info = results.get("test_info", {})
        self.logger.info(f"테스트 일시: {test_info.get('timestamp', 'Unknown')}")
        self.logger.info(f"총 쿼리 수: {test_info.get('total_queries', 0)}개")
        
        # 성공률
        self.logger.info(f"성공률: {summary.get('success_rate', 0.0):.2%}")
        
        # 실행 시간 비교
        self.logger.info("\n⏱️  실행 시간 비교:")
        self.logger.info(f"  - Re-ranking 적용: {summary.get('avg_time_with_reranking', 0.0):.4f}s (평균)")
        self.logger.info(f"  - Re-ranking 미적용: {summary.get('avg_time_without_reranking', 0.0):.4f}s (평균)")
        self.logger.info(f"  - 시간 오버헤드: +{summary.get('avg_time_overhead', 0.0):.4f}s ({summary.get('avg_time_ratio', 1.0):.2f}배)")
        self.logger.info(f"  - 오버헤드 비율: {summary.get('time_overhead_percentage', 0.0):.1f}%")
        
        # 순위 개선 비교
        self.logger.info("\n📈 순위 개선 효과:")
        self.logger.info(f"  - 평균 개선율: {summary.get('avg_improvement_rate', 0.0):.2%}")
        self.logger.info(f"  - 최대 개선율: {summary.get('max_improvement_rate', 0.0):.2%}")
        self.logger.info(f"  - 위치 변화 비율:")
        self.logger.info(f"    • 개선: {summary.get('position_improvement_rate', 0.0):.2%}")
        self.logger.info(f"    • 하락: {summary.get('position_degradation_rate', 0.0):.2%}")
        self.logger.info(f"    • 유지: {summary.get('position_unchanged_rate', 0.0):.2%}")
        
        # 성능 평가
        self.logger.info("\n🏆 성능 평가:")
        self.logger.info(f"  - 전체 평가: {verdict.get('overall_rating', 'Unknown')}")
        self.logger.info(f"  - 개선 효과: {verdict.get('improvement_assessment', 'Unknown')}")
        self.logger.info(f"  - 시간 오버헤드: {verdict.get('time_overhead_assessment', 'Unknown')}")
        
        # 추천 사항
        self.logger.info("\n📝 추천 사항:")
        self.logger.info(f"  {verdict.get('recommendation', 'More analysis needed')}")
        
        self.logger.info("\n" + "="*60)


def main():
    """메인 실행 함수"""
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/test_reranking_performance.log', encoding='utf-8')
        ]
    )
    
    # 로그 디렉토리 생성
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    logger.info("Re-ranking 성능 비교 평가 시작")
    
    try:
        # 평가자 인스턴스 생성
        evaluator = ReRankingPerformanceEvaluator()
        
        # 성능 비교 실행
        results = evaluator.run_performance_comparison()
        
        # 결과 저장
        saved_file = evaluator.save_results(results)
        
        # 요약 보고서 출력
        evaluator.print_summary_report(results)
        
        # 최종 성공 메시지
        overall_rating = results.get("overall_summary", {}).get("performance_verdict", {}).get("overall_rating", "Unknown")
        
        if overall_rating in ["Excellent", "Good"]:
            logger.info(f"\n✓ Re-ranking 성능 평가 완료 - 결과: {overall_rating}")
            logger.info("Re-ranking 기능의 성능이 우수합니다!")
        else:
            logger.info(f"\n⚠️ Re-ranking 성능 평가 완료 - 결과: {overall_rating}")
            logger.info("Re-ranking 기능의 추가 최적화가 필요할 수 있습니다.")
        
        if saved_file:
            logger.info(f"상세 결과는 {saved_file}에서 확인할 수 있습니다.")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Re-ranking 성능 평가 실패: {e}", exc_info=True)
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
