#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RAGAS 기반 RAG 시스템 평가 모듈
- RAG 시스템 평가를 위한 RAGAS 기반 평가 도구
- Context Relevancy, Faithfulness, Answer Relevancy 등 평가 지표 제공
"""

import os
import json
import logging
import time
import yaml
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field, asdict
import pandas as pd
import numpy as np
from tqdm import tqdm

# RAGAS 관련 임포트
try:
    from ragas.metrics import (
        context_relevancy, faithfulness, answer_relevancy,
        context_precision, context_recall
    )
    from ragas.metrics.base import MetricWithLLM
    from ragas import evaluate
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    print("RAGAS 라이브러리를 찾을 수 없습니다. 'pip install ragas'로 설치하세요.")

# HuggingFace datasets 임포트
try:
    import datasets
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("datasets 라이브러리를 찾을 수 없습니다. 'pip install datasets'로 설치하세요.")

# 로컬 모듈 임포트
from src.evaluation.data.ragas_dataset_converter import RAGASDatasetConverter, RAGASDatasetItem

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/ragas_evaluator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ragas_evaluator")

@dataclass
class RAGASEvaluationResult:
    """RAGAS 평가 결과 데이터 클래스"""
    dataset_name: str
    total_questions: int
    metrics: Dict[str, float]
    detailed_results: Dict[str, List[float]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    evaluation_time: float = 0.0

class RAGASEvaluator:
    """RAGAS 기반 RAG 시스템 평가 클래스"""
    
    def __init__(self, config_path: str = None):
        """
        RAGAS 평가기 초기화
        
        Args:
            config_path: RAGAS 설정 파일 경로
        """
        # RAGAS 사용 가능 여부 확인
        if not RAGAS_AVAILABLE:
            logger.error("RAGAS 라이브러리를 찾을 수 없습니다. 평가 기능을 사용할 수 없습니다.")
            return
        
        # 설정 로드
        self.config = self._load_config(config_path)
        
        # 결과 저장 경로 설정
        results_config = self.config.get("results", {})
        self.results_path = results_config.get("path", "evaluation_results/ragas_results")
        self.results_format = results_config.get("format", "json")
        
        # 로깅 설정
        logging_config = self.config.get("logging", {})
        log_level = logging_config.get("level", "info").upper()
        log_file = logging_config.get("file", "logs/ragas_evaluation.log")
        
        # 로그 레벨 설정
        numeric_level = getattr(logging, log_level, None)
        if isinstance(numeric_level, int):
            logger.setLevel(numeric_level)
        
        # 결과 저장 경로 생성
        os.makedirs(self.results_path, exist_ok=True)
        
        # 평가 지표 설정
        self.metrics = self._initialize_metrics()
        
        logger.info(f"RAGAS 평가기 초기화 완료 (결과 저장 경로: {self.results_path})")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        RAGAS 설정 파일 로드
        
        Args:
            config_path: 설정 파일 경로
            
        Returns:
            설정 딕셔너리
        """
        default_config_path = "configs/ragas_config.yaml"
        
        if config_path is None:
            if os.path.exists(default_config_path):
                config_path = default_config_path
            else:
                logger.warning(f"기본 설정 파일({default_config_path})을 찾을 수 없습니다. 기본값을 사용합니다.")
                return {}
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"설정 파일 로드 완료: {config_path}")
            return config
        except Exception as e:
            logger.error(f"설정 파일 로드 실패: {str(e)}")
            return {}
    
    def _initialize_metrics(self) -> Dict[str, Any]:
        """
        RAGAS 평가 지표 초기화
        
        Returns:
            평가 지표 딕셔너리
        """
        metrics = {}
        
        # 설정에서 사용할 지표 가져오기
        core_metrics = self.config.get("metrics", {}).get("core", [
            "context_relevancy", 
            "faithfulness", 
            "answer_relevancy"
        ])
        additional_metrics = self.config.get("metrics", {}).get("additional", [
            "context_precision", 
            "context_recall"
        ])
        
        # 모델 설정 가져오기
        models_config = self.config.get("models", {})
        
        # 평가 지표 생성
        try:
            # 컨텍스트 관련성
            if "context_relevancy" in core_metrics:
                metrics["context_relevancy"] = context_relevancy
                logger.info("컨텍스트 관련성 지표 초기화 완료")
            
            # 충실도
            if "faithfulness" in core_metrics:
                metrics["faithfulness"] = faithfulness
                logger.info("충실도 지표 초기화 완료")
            
            # 답변 관련성
            if "answer_relevancy" in core_metrics:
                metrics["answer_relevancy"] = answer_relevancy
                logger.info("답변 관련성 지표 초기화 완료")
            
            # 컨텍스트 정확도
            if "context_precision" in additional_metrics:
                metrics["context_precision"] = context_precision
                logger.info("컨텍스트 정확도 지표 초기화 완료")
            
            # 컨텍스트 재현율
            if "context_recall" in additional_metrics:
                metrics["context_recall"] = context_recall
                logger.info("컨텍스트 재현율 지표 초기화 완료")
            
            logger.info(f"총 {len(metrics)}개 평가 지표 초기화 완료")
        except Exception as e:
            logger.error(f"평가 지표 초기화 실패: {str(e)}")
        
        return metrics
    
    def evaluate(self, dataset_path: str, output_path: Optional[str] = None) -> RAGASEvaluationResult:
        """
        RAGAS 평가 실행
        
        Args:
            dataset_path: 평가 데이터셋 파일 경로
            output_path: 평가 결과 저장 경로 (None인 경우 기본 경로 사용)
            
        Returns:
            평가 결과
        """
        logger.info(f"RAGAS 평가 시작: {dataset_path}")
        
        # 평가 시작 시간
        start_time = time.time()
        
        try:
            # 데이터셋 로드
            dataset = self._load_dataset(dataset_path)
            
            if not dataset or not isinstance(dataset, dict):
                logger.error("데이터셋 로드 실패 또는 형식 오류")
                return RAGASEvaluationResult(
                    dataset_name=os.path.basename(dataset_path),
                    total_questions=0,
                    metrics={},
                    detailed_results={}
                )
            
            # 필수 필드 확인
            required_fields = ["questions", "contexts", "ground_truths"]
            for field in required_fields:
                if field not in dataset or not dataset[field]:
                    logger.error(f"필수 필드({field})가 데이터셋에 없습니다.")
                    return RAGASEvaluationResult(
                        dataset_name=os.path.basename(dataset_path),
                        total_questions=0,
                        metrics={},
                        detailed_results={},
                        metadata={"error": f"필수 필드({field})가 없습니다."}
                    )
            
            # 데이터셋 크기 확인
            total_questions = len(dataset["questions"])
            logger.info(f"평가 데이터셋 크기: {total_questions}개 질문")
            
            # HuggingFace Dataset 형식으로 변환
            hf_dataset = self._convert_to_hf_dataset(dataset)
            
            # 평가 실행
            logger.info(f"RAGAS 평가 실행 중... ({len(self.metrics)}개 지표)")
            
            eval_results = evaluate(
                hf_dataset,
                metrics=list(self.metrics.values())
            )
            
            # 평가 결과 처리
            results = self._process_evaluation_results(eval_results, dataset)
            
            # 결과 저장
            if output_path is None:
                dataset_name = os.path.basename(dataset_path).replace('.json', '')
                output_path = os.path.join(self.results_path, f"{dataset_name}_ragas_results.json")
            
            self.save_results(results, output_path)
            
            # 평가 소요 시간
            evaluation_time = time.time() - start_time
            results.evaluation_time = evaluation_time
            
            logger.info(f"RAGAS 평가 완료: {output_path} (소요 시간: {evaluation_time:.2f}초)")
            
            return results
        
        except Exception as e:
            logger.error(f"RAGAS 평가 실패: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            # 실패한 평가 결과 생성
            return RAGASEvaluationResult(
                dataset_name=os.path.basename(dataset_path),
                total_questions=0,
                metrics={},
                detailed_results={},
                metadata={"error": str(e)}
            )
    
    def _load_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """
        평가 데이터셋 로드
        
        Args:
            dataset_path: 데이터셋 파일 경로
            
        Returns:
            데이터셋 딕셔너리
        """
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            logger.info(f"데이터셋 로드 완료: {dataset_path}")
            return dataset
        except Exception as e:
            logger.error(f"데이터셋 로드 실패: {str(e)}")
            return {}
    
    def _convert_to_hf_dataset(self, dataset: Dict[str, Any]) -> datasets.Dataset:
        """
        데이터셋을 HuggingFace Dataset 형식으로 변환
        
        Args:
            dataset: 데이터셋 딕셔너리
            
        Returns:
            HuggingFace Dataset
        """
        # RAGAS 형식에 맞게 데이터셋 필드 매핑
        hf_dict = {
            "question": dataset["questions"],
            "contexts": dataset["contexts"],
            "ground_truth": dataset["ground_truths"]
        }
        
        # 답변이 있는 경우 추가
        if "answers" in dataset and any(a is not None for a in dataset["answers"]):
            hf_dict["answer"] = dataset["answers"]
        
        # HuggingFace Dataset 생성
        try:
            hf_dataset = datasets.Dataset.from_dict(hf_dict)
            logger.info(f"HuggingFace Dataset 생성 완료: {len(hf_dataset)}개 레코드")
            return hf_dataset
        except Exception as e:
            logger.error(f"HuggingFace Dataset 생성 실패: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return datasets.Dataset.from_dict({"question": [], "contexts": [], "ground_truth": []})
    
    def _process_evaluation_results(
        self, eval_results: Dict[str, Any], dataset: Dict[str, Any]
    ) -> RAGASEvaluationResult:
        """
        RAGAS 평가 결과 처리
        
        Args:
            eval_results: RAGAS 평가 결과
            dataset: 원본 데이터셋
            
        Returns:
            처리된 평가 결과
        """
        # 데이터셋 이름
        dataset_name = dataset.get("dataset_info", {}).get("name", "unknown")
        
        # 질문 수
        total_questions = len(dataset["questions"])
        
        # 지표별 평균 점수
        metrics = {}
        detailed_results = {}
        
        for metric_name in self.metrics.keys():
            if metric_name in eval_results:
                # 평균 점수
                avg_score = float(eval_results[metric_name])
                metrics[metric_name] = avg_score
                
                # 개별 점수 (있는 경우)
                if f"{metric_name}_scores" in eval_results:
                    scores = eval_results[f"{metric_name}_scores"]
                    detailed_results[metric_name] = [float(score) for score in scores]
        
        # 메타데이터 생성
        metadata = {
            "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_info": dataset.get("dataset_info", {}),
            "metrics_used": list(self.metrics.keys())
        }
        
        # 평가 결과 생성
        result = RAGASEvaluationResult(
            dataset_name=dataset_name,
            total_questions=total_questions,
            metrics=metrics,
            detailed_results=detailed_results,
            metadata=metadata
        )
        
        return result
    
    def save_results(self, results: RAGASEvaluationResult, output_path: str):
        """
        평가 결과 저장
        
        Args:
            results: 평가 결과
            output_path: 결과 저장 경로
        """
        try:
            # 출력 디렉토리 생성
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # 결과를 딕셔너리로 변환
            results_dict = asdict(results)
            
            # JSON 파일로 저장
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results_dict, f, ensure_ascii=False, indent=2)
            
            logger.info(f"평가 결과 저장 완료: {output_path}")
            
        except Exception as e:
            logger.error(f"평가 결과 저장 실패: {str(e)}")
    
    def analyze_results(self, results: RAGASEvaluationResult) -> Dict[str, Any]:
        """
        평가 결과 분석
        
        Args:
            results: 평가 결과
            
        Returns:
            분석 결과 딕셔너리
        """
        analysis = {
            "summary": {},
            "details": {}
        }
        
        # 기본 정보
        analysis["summary"]["dataset_name"] = results.dataset_name
        analysis["summary"]["total_questions"] = results.total_questions
        analysis["summary"]["evaluation_time"] = results.evaluation_time
        
        # 지표별 평균 점수
        for metric_name, score in results.metrics.items():
            analysis["summary"][metric_name] = score
        
        # 지표별 상세 분석 (점수 분포 등)
        for metric_name, scores in results.detailed_results.items():
            if scores:
                analysis["details"][metric_name] = {
                    "min": min(scores),
                    "max": max(scores),
                    "median": np.median(scores),
                    "std": np.std(scores),
                    "percentiles": {
                        "25": np.percentile(scores, 25),
                        "50": np.percentile(scores, 50),
                        "75": np.percentile(scores, 75),
                        "90": np.percentile(scores, 90)
                    },
                    "distribution": {
                        "0.0-0.2": len([s for s in scores if 0.0 <= s < 0.2]),
                        "0.2-0.4": len([s for s in scores if 0.2 <= s < 0.4]),
                        "0.4-0.6": len([s for s in scores if 0.4 <= s < 0.6]),
                        "0.6-0.8": len([s for s in scores if 0.6 <= s < 0.8]),
                        "0.8-1.0": len([s for s in scores if 0.8 <= s <= 1.0])
                    }
                }
        
        return analysis
    
    def compare_results(self, results_list: List[RAGASEvaluationResult]) -> Dict[str, Any]:
        """
        여러 평가 결과 비교
        
        Args:
            results_list: 평가 결과 리스트
            
        Returns:
            비교 결과 딕셔너리
        """
        if not results_list:
            return {}
        
        comparison = {
            "datasets": [r.dataset_name for r in results_list],
            "metrics": {},
            "rankings": {}
        }
        
        # 모든 결과에 공통으로 있는 지표 찾기
        common_metrics = set.intersection(*[set(r.metrics.keys()) for r in results_list])
        
        # 지표별 비교
        for metric_name in common_metrics:
            # 지표별 점수
            comparison["metrics"][metric_name] = {
                r.dataset_name: r.metrics[metric_name] for r in results_list
            }
            
            # 지표별 순위 (높은 점수가 1위)
            scores = [(r.dataset_name, r.metrics[metric_name]) for r in results_list]
            scores.sort(key=lambda x: x[1], reverse=True)
            comparison["rankings"][metric_name] = {name: rank+1 for rank, (name, _) in enumerate(scores)}
        
        # 종합 점수 계산 (평균 점수)
        comparison["overall"] = {}
        for result in results_list:
            avg_score = sum(result.metrics[m] for m in common_metrics) / len(common_metrics)
            comparison["overall"][result.dataset_name] = avg_score
        
        # 종합 순위
        overall_scores = [(name, score) for name, score in comparison["overall"].items()]
        overall_scores.sort(key=lambda x: x[1], reverse=True)
        comparison["overall_ranking"] = {name: rank+1 for rank, (name, _) in enumerate(overall_scores)}
        
        return comparison

def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='RAGAS 평가 도구')
    parser.add_argument('--dataset', type=str, required=True, help='평가 데이터셋 파일 경로')
    parser.add_argument('--output', type=str, help='결과 저장 파일 경로')
    parser.add_argument('--config', type=str, help='RAGAS 설정 파일 경로')
    
    args = parser.parse_args()
    
    # 평가기 초기화
    evaluator = RAGASEvaluator(args.config)
    
    # 평가 실행
    results = evaluator.evaluate(args.dataset, args.output)
    
    # 결과 분석
    analysis = evaluator.analyze_results(results)
    
    # 결과 요약 출력
    print("\n=== RAGAS 평가 결과 요약 ===")
    print(f"데이터셋: {results.dataset_name}")
    print(f"전체 질문 수: {results.total_questions}")
    print(f"평가 소요 시간: {results.evaluation_time:.2f}초")
    print("\n평가 지표별 점수:")
    for metric_name, score in results.metrics.items():
        print(f"- {metric_name}: {score:.4f}")

if __name__ == "__main__":
    main()
