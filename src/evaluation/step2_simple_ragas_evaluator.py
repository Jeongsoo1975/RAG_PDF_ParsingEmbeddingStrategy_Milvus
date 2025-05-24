"""
2단계 RAGAS 평가 실행기
수동 답변 데이터와 Milvus 검색 결과를 결합하여 간소화된 RAGAS 지표를 계산
"""

import os
import sys
import json
import logging
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.evaluation.manual_data_loader import ManualAnswerDataLoader
from src.evaluation.milvus_context_generator import MilvusContextGenerator
from src.evaluation.simple_ragas_metrics import SimpleRAGASMetrics


class Step2SimpleRAGASEvaluator:
    """2단계 간소화된 RAGAS 평가 실행기"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        평가기 초기화
        
        Args:
            config_path: 설정 파일 경로 (선택적)
        """
        self.logger = self._setup_logging()
        self.project_root = project_root
        
        # 구성요소 초기화
        try:
            self.data_loader = ManualAnswerDataLoader()
            self.context_generator = MilvusContextGenerator()
            self.metrics = SimpleRAGASMetrics()
            
            self.logger.info("모든 구성요소가 성공적으로 초기화되었습니다.")
            
        except Exception as e:
            self.logger.error(f"구성요소 초기화 중 오류 발생: {e}")
            raise
            
        # 결과 저장 디렉토리 설정
        self.results_dir = self.project_root / "evaluation_results" / "step2_ragas"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 평가 결과 저장용
        self.evaluation_results = []
        self.overall_metrics = {}
        
    def _setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger("Step2RAGASEvaluator")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # 콘솔 핸들러
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # 파일 핸들러
            log_dir = project_root / "logs"
            log_dir.mkdir(exist_ok=True)
            file_handler = logging.FileHandler(log_dir / "step2_ragas_evaluation.log")
            file_handler.setLevel(logging.DEBUG)
            
            # 포맷터
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)
            
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)
            
        return logger
    
    def run_evaluation(self) -> Dict[str, Any]:
        """
        전체 평가 프로세스 실행
        
        Returns:
            Dict[str, Any]: 평가 결과
        """
        self.logger.info("=== 2단계 RAGAS 평가 시작 ===")
        start_time = datetime.now()
        
        try:
            # 1. 수동 답변 데이터 로드
            self.logger.info("수동 답변 데이터 로딩 중...")
            
            # TODO: 실제 평가 데이터셋 로드 로직 구현 필요
            # 현재는 샘플 데이터로 대체
            manual_data = [
                {
                    'question': '실제 평가 질문이 필요합니다.',
                    'gold_standard': {
                        'answer': '실제 정답이 필요합니다.',
                        'essential_elements': ['실제', '키워드'],
                        'source_quote': '실제 참조 컨텍스트가 필요합니다.'
                    }
                }
            ]
            
            if not manual_data:
                self.logger.error("로드된 데이터가 비어있습니다.")
                return {
                    'success': False,
                    'error': '로드된 데이터가 비어있습니다.',
                    'processed_count': 0
                }
            self.logger.info(f"로드된 질문 수: {len(manual_data)}")
            
            # 2. 각 질문에 대한 평가 실행
            total_questions = len(manual_data)
            processed_count = 0
            
            for item in manual_data:
                try:
                    result = self._evaluate_single_item(item)
                    self.evaluation_results.append(result)
                    processed_count += 1
                    
                    if processed_count % 5 == 0:
                        self.logger.info(f"진행률: {processed_count}/{total_questions} ({processed_count/total_questions*100:.1f}%)")
                        
                except Exception as e:
                    self.logger.error(f"질문 '{item.get('question', 'Unknown')}' 평가 중 오류: {e}")
                    continue
            
            # 3. 전체 평가 지표 계산
            self._calculate_overall_metrics()
            
            # 4. 결과 저장
            result_file = self.save_results()
            
            # 5. 리포트 생성
            self.generate_report()
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            self.logger.info(f"=== 2단계 RAGAS 평가 완료 ====")
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
            return {
                'success': False,
                'error': str(e),
                'processed_count': len(self.evaluation_results)
            }
    
    def _evaluate_single_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        개별 질문에 대한 평가 수행
        
        Args:
            item: 수동 답변 데이터 항목
            
        Returns:
            Dict[str, Any]: 평가 결과
        """
        question = item.get('question', '')
        gold_standard = item.get('gold_standard', {})
        reference_answer = gold_standard.get('answer', '')
        
        # Milvus 검색으로 컨텍스트 생성
        try:
            search_contexts = self.context_generator.generate_contexts(question)
            contexts = [ctx['content'] for ctx in search_contexts]
        except Exception as e:
            self.logger.warning(f"Milvus 검색 실패, 참조 컨텍스트 사용: {e}")
            # 폴백: 참조 컨텍스트 사용
            source_quote = gold_standard.get('source_quote', '')
            contexts = [source_quote] if source_quote else ['컨텍스트를 찾을 수 없습니다.']
        
        # 간소화된 RAGAS 지표 계산
        metrics_result = self.metrics.evaluate(
            question=question,
            answer=reference_answer,
            contexts=contexts
        )
        
        # 결과 구성
        result = {
            'question': question,
            'reference_answer': reference_answer,
            'contexts': contexts,
            'context_count': len(contexts),
            'metrics': metrics_result,
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def _calculate_overall_metrics(self) -> None:
        """
        전체 평가 지표 계산
        """
        if not self.evaluation_results:
            self.overall_metrics = {
                'error': '평가 결과가 없습니다.'
            }
            return
        
        # 각 지표별 평균 계산
        context_relevancy_scores = []
        faithfulness_scores = []
        answer_relevancy_scores = []
        
        for result in self.evaluation_results:
            metrics = result.get('metrics', {})
            context_relevancy_scores.append(metrics.get('context_relevancy', 0.0))
            faithfulness_scores.append(metrics.get('faithfulness', 0.0))
            answer_relevancy_scores.append(metrics.get('answer_relevancy', 0.0))
        
        self.overall_metrics = {
            'total_questions': len(self.evaluation_results),
            'average_context_relevancy': sum(context_relevancy_scores) / len(context_relevancy_scores),
            'average_faithfulness': sum(faithfulness_scores) / len(faithfulness_scores),
            'average_answer_relevancy': sum(answer_relevancy_scores) / len(answer_relevancy_scores),
            'min_context_relevancy': min(context_relevancy_scores),
            'max_context_relevancy': max(context_relevancy_scores),
            'min_faithfulness': min(faithfulness_scores),
            'max_faithfulness': max(faithfulness_scores),
            'min_answer_relevancy': min(answer_relevancy_scores),
            'max_answer_relevancy': max(answer_relevancy_scores)
        }
        
        # 전체 점수 계산 (3개 지표의 평균)
        self.overall_metrics['overall_score'] = (
            self.overall_metrics['average_context_relevancy'] +
            self.overall_metrics['average_faithfulness'] +
            self.overall_metrics['average_answer_relevancy']
        ) / 3.0
    
    def save_results(self) -> Path:
        """
        평가 결과를 JSON 형태로 저장
        
        Returns:
            Path: 저장된 파일 경로
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = self.results_dir / f"step2_ragas_evaluation_{timestamp}.json"
        
        # 전체 결과 구성
        full_results = {
            'evaluation_info': {
                'type': 'step2_simple_ragas_evaluation',
                'timestamp': datetime.now().isoformat(),
                'total_questions': len(self.evaluation_results),
                'evaluator_version': '1.0.0'
            },
            'overall_metrics': self.overall_metrics,
            'detailed_results': self.evaluation_results
        }
        
        # JSON 파일로 저장
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"평가 결과 저장 완료: {result_file}")
        return result_file
    
    def generate_report(self) -> None:
        """
        평가 결과 요약 리포트 생성
        """
        if not self.overall_metrics:
            self.logger.warning("전체 지표가 계산되지 않아 리포트를 생성할 수 없습니다.")
            return
        
        print("\n" + "="*60)
        print("    2단계 RAGAS 평가 결과 요약")
        print("="*60)
        
        print(f"평가 대상 질문 수: {self.overall_metrics.get('total_questions', 0)}개")
        print(f"전체 평균 점수: {self.overall_metrics.get('overall_score', 0):.3f}")
        
        print("\n[ 세부 지표 ]")
        print(f"Context Relevancy  : {self.overall_metrics.get('average_context_relevancy', 0):.3f} "
              f"(최소: {self.overall_metrics.get('min_context_relevancy', 0):.3f}, "
              f"최대: {self.overall_metrics.get('max_context_relevancy', 0):.3f})")
        
        print(f"Faithfulness       : {self.overall_metrics.get('average_faithfulness', 0):.3f} "
              f"(최소: {self.overall_metrics.get('min_faithfulness', 0):.3f}, "
              f"최대: {self.overall_metrics.get('max_faithfulness', 0):.3f})")
        
        print(f"Answer Relevancy   : {self.overall_metrics.get('average_answer_relevancy', 0):.3f} "
              f"(최소: {self.overall_metrics.get('min_answer_relevancy', 0):.3f}, "
              f"최대: {self.overall_metrics.get('max_answer_relevancy', 0):.3f})")
        
        # 성능 평가
        overall_score = self.overall_metrics.get('overall_score', 0)
        if overall_score >= 0.7:
            performance = "우수함 (Excellent)"
        elif overall_score >= 0.5:
            performance = "양호함 (Good)"
        elif overall_score >= 0.3:
            performance = "보통 (Fair)"
        else:
            performance = "개선 필요 (Poor)"
        
        print(f"\n전체 성능 평가: {performance}")
        print("="*60 + "\n")


# 메인 실행 부분
if __name__ == "__main__":
    try:
        evaluator = Step2SimpleRAGASEvaluator()
        result = evaluator.run_evaluation()
        
        if result['success']:
            print(f"\n평가 완료! 처리된 질문: {result['processed_count']}/{result['total_count']}")
            print(f"결과 파일: {result['result_file']}")
        else:
            print(f"\n평가 실패: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"평가 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()