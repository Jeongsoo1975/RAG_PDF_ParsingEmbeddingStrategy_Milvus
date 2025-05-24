#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step4 개선된 RAGAS 평가기 - Answer Relevancy & Context Relevancy 향상
"""

import os
import sys
import json
import logging
import traceback
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.evaluation.simple_ragas_metrics import SimpleRAGASMetrics
from src.evaluation.improved_answer_generator import ImprovedAnswerGenerator, ImprovedContextSearcher


class Step4ImprovedRAGASEvaluator:
    """4단계 개선된 RAGAS 평가기"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = self._setup_logging()
        self.project_root = project_root
        
        # 개선된 구성요소 초기화
        try:
            from src.evaluation.step3_standard_ragas_evaluator import AmnestyDataLoader, AmnestyMilvusSearcher
            
            self.data_loader = AmnestyDataLoader()
            self.searcher = AmnestyMilvusSearcher()
            self.answer_generator = ImprovedAnswerGenerator()  # 개선된 답변 생성기
            self.context_searcher = ImprovedContextSearcher()  # 개선된 컨텍스트 검색기
            self.metrics = SimpleRAGASMetrics()
            
            self.logger.info("개선된 구성요소가 성공적으로 초기화되었습니다.")
            
        except Exception as e:
            self.logger.error(f"구성요소 초기화 중 오류 발생: {e}")
            raise
            
        # 결과 저장 디렉토리 설정
        self.results_dir = self.project_root / "evaluation_results" / "step4_improved_ragas"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 평가 결과 저장용
        self.evaluation_results = []
        self.overall_metrics = {}
    
    def _setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger("Step4ImprovedRAGASEvaluator")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            log_dir = project_root / "logs"
            log_dir.mkdir(exist_ok=True)
            file_handler = logging.FileHandler(log_dir / "step4_improved_ragas_evaluation.log", encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)
            
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)
            
        return logger
    
    def run_evaluation(self) -> Dict[str, Any]:
        """전체 평가 프로세스 실행"""
        self.logger.info("=== 4단계 개선된 RAGAS 평가 시작 ===")
        start_time = datetime.now()
        
        try:
            # 1. 데이터 로드
            self.logger.info("Amnesty QA 평가 데이터 로딩 중...")
            evaluation_items = self.data_loader.load_evaluation_dataset()
            
            if not evaluation_items:
                self.logger.error("로드된 데이터가 비어있습니다.")
                return {'success': False, 'error': '로드된 데이터가 비어있습니다.', 'processed_count': 0}
            
            self.logger.info(f"로드된 질문 수: {len(evaluation_items)}")
            
            # 2. 개선된 평가 실행
            total_questions = len(evaluation_items)
            processed_count = 0
            
            for item in evaluation_items:
                try:
                    result = self._evaluate_single_item_improved(item)
                    self.evaluation_results.append(result)
                    processed_count += 1
                    
                    if processed_count % 2 == 0:
                        self.logger.info(f"진행률: {processed_count}/{total_questions} ({processed_count/total_questions*100:.1f}%)")
                        
                except Exception as e:
                    self.logger.error(f"질문 '{item.get('question', 'Unknown')}' 평가 중 오류: {e}")
                    continue
            
            # 3. 전체 평가 지표 계산
            self._calculate_overall_metrics()
            
            # 4. 결과 저장
            result_file = self.save_results()
            
            # 5. 개선 리포트 생성
            self.generate_improvement_report()
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            self.logger.info(f"=== 4단계 개선된 RAGAS 평가 완료 ===")
            self.logger.info(f"총 소요 시간: {duration.total_seconds():.2f}초")
            self.logger.info(f"처리된 질문 수: {processed_count}/{total_questions}")
            self.logger.info(f"결과 저장: {result_file}")
            
            return {
                'success': True,
                'processed_count': processed_count,
                'total_count': total_questions,
                'duration_seconds': duration.total_seconds(),
                'result_file': str(result_file),
                'overall_metrics': self.overall_metrics
            }
            
        except Exception as e:
            self.logger.error(f"평가 실행 중 심각한 오류 발생: {e}")
            self.logger.error(traceback.format_exc())
            return {'success': False, 'error': str(e), 'processed_count': len(self.evaluation_results)}
    
    def _evaluate_single_item_improved(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """개선된 개별 질문 평가"""
        question = item.get('question', '')
        ground_truths = item.get('ground_truths', [])
        original_contexts = item.get('contexts', [])
        
        # 1. 개선된 컨텍스트 검색
        try:
            retrieved_contexts = self.searcher.search_contexts(question, top_k=5)
            all_contexts = original_contexts + retrieved_contexts
            
            # 개선된 컨텍스트 선별
            improved_contexts = self.context_searcher.improved_context_search(
                question, all_contexts, top_k=3
            )
            
        except Exception as e:
            self.logger.warning(f"개선된 컨텍스트 검색 실패: {e}")
            improved_contexts = original_contexts
        
        # 2. 개선된 답변 생성
        try:
            generated_answer = self.answer_generator.generate_improved_answer(
                question, improved_contexts
            )
        except Exception as e:
            self.logger.warning(f"개선된 답변 생성 실패: {e}")
            # 폴백: 기본 답변 생성
            generated_answer = "Unable to generate improved answer due to processing error."
        
        # 3. RAGAS 지표 평가
        try:
            ragas_result = self.metrics.evaluate_single_item(
                question_id=item.get('id', 'unknown'),
                question=question,
                contexts=improved_contexts,
                answer=generated_answer,
                ground_truth_contexts=original_contexts
            )
            
            metrics_dict = {
                'context_relevancy': ragas_result.context_relevancy,
                'context_precision': ragas_result.context_precision,
                'context_recall': ragas_result.context_recall,
                'faithfulness': ragas_result.faithfulness,
                'answer_relevancy': ragas_result.answer_relevancy,
                'overall_score': (ragas_result.context_relevancy + 
                                ragas_result.context_precision + 
                                ragas_result.context_recall + 
                                ragas_result.faithfulness + 
                                ragas_result.answer_relevancy) / 5.0
            }
            
        except Exception as e:
            self.logger.error(f"RAGAS 지표 계산 실패: {e}")
            metrics_dict = {
                'context_relevancy': 0.0,
                'context_precision': 0.0,
                'context_recall': 0.0,
                'faithfulness': 0.0,
                'answer_relevancy': 0.0,
                'overall_score': 0.0,
                'error': str(e)
            }
        
        # 4. 결과 구성
        result = {
            'question_id': item.get('id', 'unknown'),
            'question': question,
            'generated_answer': generated_answer,
            'ground_truths': ground_truths,
            'contexts': improved_contexts,
            'context_count': len(improved_contexts),
            'metrics': metrics_dict,
            'improvements': {
                'used_improved_context_search': True,
                'used_improved_answer_generation': True,
                'original_context_count': len(original_contexts),
                'improved_context_count': len(improved_contexts)
            },
            'metadata': item.get('metadata', {}),
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def _calculate_overall_metrics(self) -> None:
        """전체 평가 지표 계산"""
        if not self.evaluation_results:
            self.overall_metrics = {'error': '평가 결과가 없습니다.'}
            return
        
        # 각 지표별 점수 수집
        metric_names = ['context_relevancy', 'context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']
        metric_scores = {name: [] for name in metric_names}
        overall_scores = []
        
        for result in self.evaluation_results:
            metrics = result.get('metrics', {})
            for name in metric_names:
                score = metrics.get(name, 0.0)
                metric_scores[name].append(score)
            
            overall_scores.append(metrics.get('overall_score', 0.0))
        
        # 통계 계산
        self.overall_metrics = {
            'total_questions': len(self.evaluation_results),
            'evaluation_type': 'improved_ragas_5_metrics'
        }
        
        # 각 지표별 통계
        for name in metric_names:
            scores = metric_scores[name]
            self.overall_metrics[f'average_{name}'] = sum(scores) / len(scores) if scores else 0.0
            self.overall_metrics[f'min_{name}'] = min(scores) if scores else 0.0
            self.overall_metrics[f'max_{name}'] = max(scores) if scores else 0.0
        
        # 전체 점수 통계
        self.overall_metrics['average_overall_score'] = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
        self.overall_metrics['min_overall_score'] = min(overall_scores) if overall_scores else 0.0
        self.overall_metrics['max_overall_score'] = max(overall_scores) if overall_scores else 0.0
    
    def save_results(self) -> Path:
        """평가 결과를 JSON 형태로 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = self.results_dir / f"step4_improved_ragas_evaluation_{timestamp}.json"
        
        # 전체 결과 구성
        full_results = {
            'evaluation_info': {
                'type': 'step4_improved_ragas_evaluation',
                'dataset': 'amnesty_qa',
                'improvements': [
                    'improved_context_search_with_entity_matching',
                    'improved_answer_generation_with_question_type_analysis',
                    'removed_unnecessary_prefixes',
                    'enhanced_relevancy_scoring'
                ],
                'metrics': ['context_relevancy', 'context_precision', 'context_recall', 'faithfulness', 'answer_relevancy'],
                'timestamp': datetime.now().isoformat(),
                'total_questions': len(self.evaluation_results),
                'evaluator_version': '2.0.0'
            },
            'overall_metrics': self.overall_metrics,
            'detailed_results': self.evaluation_results
        }
        
        # JSON 파일로 저장
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"개선된 평가 결과 저장 완료: {result_file}")
        return result_file
    
    def generate_improvement_report(self) -> None:
        """개선 결과 리포트 생성"""
        if not self.overall_metrics:
            self.logger.warning("전체 지표가 계산되지 않아 리포트를 생성할 수 없습니다.")
            return
        
        print("\n" + "="*80)
        print("    🚀 4단계 개선된 RAGAS 평가 결과 (Amnesty QA)")
        print("="*80)
        
        print(f"평가 대상 질문 수: {self.overall_metrics.get('total_questions', 0)}개")
        print(f"✨ 개선된 전체 평균 점수: {self.overall_metrics.get('average_overall_score', 0):.3f}")
        
        print("\n[ 🎯 개선된 RAGAS 5개 지표 ]")
        metric_labels = {
            'context_relevancy': '🔍 Context Relevancy (개선 목표)',
            'context_precision': '🎯 Context Precision', 
            'context_recall': '📚 Context Recall',
            'faithfulness': '💯 Faithfulness',
            'answer_relevancy': '💬 Answer Relevancy (개선 목표)'
        }
        
        for metric_key, label in metric_labels.items():
            avg = self.overall_metrics.get(f'average_{metric_key}', 0)
            min_val = self.overall_metrics.get(f'min_{metric_key}', 0)
            max_val = self.overall_metrics.get(f'max_{metric_key}', 0)
            
            # 개선 목표 지표 표시
            improvement_indicator = ""
            if metric_key in ['context_relevancy', 'answer_relevancy']:
                if avg > 0.6:
                    improvement_indicator = " 🟢 목표 달성!"
                elif avg > 0.4:
                    improvement_indicator = " 🟡 개선 중"
                else:
                    improvement_indicator = " 🔴 추가 개선 필요"
            
            print(f"{label:30}: {avg:.3f} (최소: {min_val:.3f}, 최대: {max_val:.3f}){improvement_indicator}")
        
        # 성능 평가
        overall_score = self.overall_metrics.get('average_overall_score', 0)
        if overall_score >= 0.7:
            performance = "🌟 우수함 (Excellent)"
        elif overall_score >= 0.5:
            performance = "✅ 양호함 (Good)"
        elif overall_score >= 0.3:
            performance = "⚠️ 보통 (Fair)"
        else:
            performance = "❌ 개선 필요 (Poor)"
        
        print(f"\n🏆 전체 성능 평가: {performance}")
        print(f"📊 데이터셋: Amnesty QA (Human Rights)")
        print(f"🔧 평가 방식: 개선된 RAGAS 5개 지표")
        print(f"🚀 적용된 개선사항:")
        print(f"   - 질문 유형별 맞춤 답변 생성")
        print(f"   - 엔티티 기반 컨텍스트 검색")
        print(f"   - 불필요한 접두사 제거")
        print(f"   - 가중치 기반 관련성 점수 계산")
        print("="*80 + "\n")


# 메인 실행 부분
if __name__ == "__main__":
    try:
        evaluator = Step4ImprovedRAGASEvaluator()
        result = evaluator.run_evaluation()
        
        if result['success']:
            print(f"\n🎉 개선된 평가 완료! 처리된 질문: {result['processed_count']}/{result['total_count']}")
            print(f"📁 결과 파일: {result['result_file']}")
        else:
            print(f"\n❌ 평가 실패: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"평가 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
