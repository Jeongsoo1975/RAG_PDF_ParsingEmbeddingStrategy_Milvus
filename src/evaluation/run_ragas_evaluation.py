#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configure UTF-8 encoding for console output
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except AttributeError:
    # For older Python versions
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

"""
RAGAS 평가 실행 스크립트
- RAGAS 기반 RAG 시스템 평가 실행
- 평가 데이터셋 변환 및 평가 실행, 결과 저장 기능 제공
"""

import os
import sys
import json
import logging
import time
import argparse
from typing import Dict, List, Any, Optional, Tuple, Union

# 로컬 모듈 임포트
from src.evaluation.ragas_evaluator import RAGASEvaluator, RAGASEvaluationResult
from src.evaluation.data.ragas_dataset_converter import RAGASDatasetConverter

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/run_ragas_evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("run_ragas_evaluation")

def convert_and_evaluate(
    input_path: str,
    output_dir: str = "evaluation_results/ragas_results",
    config_path: str = None,
    verbose: bool = False
) -> RAGASEvaluationResult:
    """
    기존 평가 데이터셋을 RAGAS 형식으로 변환하고 평가 실행
    
    Args:
        input_path: 입력 데이터셋 파일 경로
        output_dir: 출력 디렉토리 경로
        config_path: RAGAS 설정 파일 경로
        verbose: 상세 출력 여부
        
    Returns:
        평가 결과
    """
    # 시작 시간
    start_time = time.time()
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 변환기 초기화
    converter = RAGASDatasetConverter(config_path)
    
    # 데이터셋 변환
    dataset_filename = os.path.basename(input_path).replace('.json', '_ragas.json')
    dataset_output_path = os.path.join(output_dir, dataset_filename)
    
    logger.info(f"데이터셋 변환 시작: {input_path} -> {dataset_output_path}")
    items = converter.convert_from_json(input_path, dataset_output_path)
    
    if not items:
        logger.error("데이터셋 변환 실패")
        return RAGASEvaluationResult(
            dataset_name=os.path.basename(input_path),
            total_questions=0,
            metrics={},
            detailed_results={}
        )
    
    # 변환 결과 출력
    logger.info(f"데이터셋 변환 완료: {len(items)}개 항목")
    
    if verbose:
        print("\n=== 데이터셋 변환 결과 ===")
        print(f"전체 항목 수: {len(items)}")
        print(f"첫 번째 항목:")
        print(f"  질문: {items[0].question}")
        print(f"  문맥 수: {len(items[0].contexts)}")
        print(f"  정답 수: {len(items[0].ground_truths)}")
    
    # 평가기 초기화
    evaluator = RAGASEvaluator(config_path)
    
    # 평가 실행
    results_filename = os.path.basename(input_path).replace('.json', '_ragas_results.json')
    results_output_path = os.path.join(output_dir, results_filename)
    
    logger.info(f"RAGAS 평가 시작: {dataset_output_path}")
    results = evaluator.evaluate(dataset_output_path, results_output_path)
    
    # 소요 시간
    elapsed_time = time.time() - start_time
    logger.info(f"전체 소요 시간: {elapsed_time:.2f}초")
    
    # 평가 결과 요약
    if verbose:
        _print_evaluation_summary(results)
    
    return results

def evaluate_multiple_datasets(
    input_dir: str,
    pattern: str = "*.json",
    output_dir: str = "evaluation_results/ragas_results",
    config_path: str = None,
    verbose: bool = False
) -> List[RAGASEvaluationResult]:
    """
    여러 데이터셋 평가 실행
    
    Args:
        input_dir: 입력 데이터셋 디렉토리 경로
        pattern: 파일 패턴
        output_dir: 출력 디렉토리 경로
        config_path: RAGAS 설정 파일 경로
        verbose: 상세 출력 여부
        
    Returns:
        평가 결과 목록
    """
    import glob
    
    # 입력 파일 목록 조회
    input_files = glob.glob(os.path.join(input_dir, pattern))
    
    if not input_files:
        logger.error(f"입력 디렉토리({input_dir})에서 패턴({pattern})과 일치하는 파일을 찾을 수 없습니다.")
        return []
    
    logger.info(f"총 {len(input_files)}개 데이터셋 발견")
    
    # 결과 목록
    results_list = []
    
    # 각 데이터셋 평가
    for input_path in input_files:
        try:
            # 파일명만 추출
            filename = os.path.basename(input_path)
            
            logger.info(f"데이터셋 평가 시작: {filename}")
            results = convert_and_evaluate(input_path, output_dir, config_path, verbose)
            
            if results.total_questions > 0:
                results_list.append(results)
                
        except Exception as e:
            logger.error(f"데이터셋({input_path}) 평가 중 오류 발생: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    # 결과가 있는 경우 비교 분석
    if len(results_list) > 1:
        # 평가기 초기화
        evaluator = RAGASEvaluator(config_path)
        
        # 결과 비교
        comparison = evaluator.compare_results(results_list)
        
        # 비교 결과 저장
        comparison_path = os.path.join(output_dir, "ragas_comparison_results.json")
        try:
            with open(comparison_path, 'w', encoding='utf-8') as f:
                json.dump(comparison, f, ensure_ascii=False, indent=2)
            logger.info(f"비교 결과 저장 완료: {comparison_path}")
            
            if verbose:
                _print_comparison_summary(comparison)
                
        except Exception as e:
            logger.error(f"비교 결과 저장 실패: {str(e)}")
    
    return results_list

def _print_evaluation_summary(results: RAGASEvaluationResult):
    """
    평가 결과 요약 출력
    
    Args:
        results: 평가 결과
    """
    print("\n=== RAGAS 평가 결과 요약 ===")
    print(f"데이터셋: {results.dataset_name}")
    print(f"전체 질문 수: {results.total_questions}")
    print(f"평가 소요 시간: {results.evaluation_time:.2f}초")
    
    print("\n평가 지표별 점수:")
    for metric_name, score in results.metrics.items():
        print(f"- {metric_name}: {score:.4f}")

def _print_comparison_summary(comparison: Dict[str, Any]):
    """
    비교 결과 요약 출력
    
    Args:
        comparison: 비교 결과
    """
    print("\n=== RAGAS 평가 비교 결과 요약 ===")
    print(f"비교 데이터셋: {', '.join(comparison['datasets'])}")
    
    print("\n지표별 점수:")
    for metric_name, scores in comparison['metrics'].items():
        print(f"- {metric_name}:")
        for dataset_name, score in scores.items():
            rank = comparison['rankings'][metric_name][dataset_name]
            print(f"  - {dataset_name}: {score:.4f} (순위: {rank})")
    
    print("\n종합 점수:")
    for dataset_name, score in comparison['overall'].items():
        rank = comparison['overall_ranking'][dataset_name]
        print(f"- {dataset_name}: {score:.4f} (순위: {rank})")

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='RAGAS 평가 실행 도구')
    
    # 필수 인자
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='입력 데이터셋 파일 경로 또는 디렉토리'
    )
    
    # 선택적 인자
    parser.add_argument(
        '--output-dir',
        type=str,
        default='evaluation_results/ragas_results',
        help='결과 저장 디렉토리 경로'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='RAGAS 설정 파일 경로'
    )
    parser.add_argument(
        '--multiple',
        action='store_true',
        help='여러 데이터셋 평가 모드 (입력이 디렉토리인 경우)'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.json',
        help='파일 패턴 (여러 데이터셋 평가 모드에서 사용)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='상세 출력 모드'
    )
    
    args = parser.parse_args()
    
    # 로그 레벨 설정 (상세 모드인 경우 DEBUG)
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # 평가 실행
    if args.multiple or os.path.isdir(args.input):
        # 여러 데이터셋 평가
        input_dir = args.input
        results_list = evaluate_multiple_datasets(
            input_dir=input_dir,
            pattern=args.pattern,
            output_dir=args.output_dir,
            config_path=args.config,
            verbose=args.verbose
        )
        
        # 결과 요약
        if results_list:
            print(f"\n총 {len(results_list)}개 데이터셋 평가 완료")
            
    else:
        # 단일 데이터셋 평가
        results = convert_and_evaluate(
            input_path=args.input,
            output_dir=args.output_dir,
            config_path=args.config,
            verbose=args.verbose
        )
        
        # 결과 요약 출력 (verbose 모드가 아닌 경우에도 출력)
        if not args.verbose:
            _print_evaluation_summary(results)

if __name__ == "__main__":
    main()
