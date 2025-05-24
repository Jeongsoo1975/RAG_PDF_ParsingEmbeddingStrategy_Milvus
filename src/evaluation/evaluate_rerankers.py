#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
재순위화 모델 평가 스크립트
세 가지 재순위화 모델을 평가하고 결과를 비교합니다:
1. cross-encoder/ms-marco-MiniLM-L-6-v2: 기본 영어 재순위화 모델
2. cross-encoder/mmarco-mMiniLMv2-L12-H384-v1: 다국어 지원 재순위화 모델
3. jhgan/ko-sroberta-multitask: 한국어 특화 재순위화 모델
"""

import os
import json
import argparse
import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from tqdm import tqdm
import sys
from pathlib import Path

# Ensure the src directory is in the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import from src
from src.utils.config import Config
from src.rag.retriever import DocumentRetriever
from src.rag.embedder import DocumentEmbedder

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("reranking_evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("reranking_evaluator")

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    logger.error("CrossEncoder를 찾을 수 없습니다. 'pip install sentence-transformers'로 설치하세요.")


def load_evaluation_dataset(dataset_path: str) -> Dict[str, Any]:
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
        logger.info(f"질문 수: {len(dataset.get('questions', []))}")
        return dataset
    except Exception as e:
        logger.error(f"데이터셋 로드 실패: {e}")
        return {"questions": []}


def evaluate_reranking(
    query: str,
    initial_results: List[Dict[str, Any]],
    reranker_model: str,
    ground_truth_ids: List[str] = None
) -> Dict[str, Any]:
    """
    재순위화 성능 평가
    
    Args:
        query: 검색 쿼리
        initial_results: 초기 검색 결과 리스트 (각 항목은 id, content, score 등을 포함)
        reranker_model: 재순위화 모델 이름
        ground_truth_ids: 정답 문서 ID 목록
        
    Returns:
        평가 메트릭
    """
    if not CROSS_ENCODER_AVAILABLE:
        logger.error("CrossEncoder를 사용할 수 없습니다.")
        return {"error": "CrossEncoder not available"}
    
    # 결과가 없으면 빈 결과 반환
    if not initial_results:
        logger.warning("평가할 초기 결과가 없습니다.")
        return {"error": "No initial results to evaluate"}
    
    # 재순위화 모델 로드
    try:
        logger.info(f"재순위화 모델 로드 중: {reranker_model}")
        reranker = CrossEncoder(reranker_model)
    except Exception as e:
        logger.error(f"재순위화 모델 로드 실패: {e}")
        return {"error": f"Failed to load reranker model: {e}"}
    
    # 재순위화 입력 준비
    logger.info(f"재순위화 입력 준비: {len(initial_results)} 결과")
    pairs = [(query, doc["content"]) for doc in initial_results]
    
    # 배치 크기 설정
    batch_size = min(32, len(pairs))
    
    # 재순위화 점수 계산
    try:
        logger.info(f"재순위화 점수 계산 중 (배치 크기: {batch_size})")
        start_time = time.time()
        rerank_scores = reranker.predict(pairs, batch_size=batch_size)
        rerank_time = time.time() - start_time
        logger.info(f"재순위화 완료 (소요 시간: {rerank_time:.2f}초)")
    except Exception as e:
        logger.error(f"재순위화 점수 계산 실패: {e}")
        return {"error": f"Failed to compute reranking scores: {e}"}
    
    # 초기 결과 복사 후 재순위화 점수 추가
    reranked_results = initial_results.copy()
    for i, score in enumerate(rerank_scores):
        reranked_results[i]["rerank_score"] = float(score)
    
    # 재순위화 점수로 정렬
    reranked_results = sorted(reranked_results, key=lambda x: x["rerank_score"], reverse=True)
    
    # 초기 검색 및 재순위화 결과 비교
    metrics = {}
    
    # 관련 문서가 알려진 경우 메트릭 계산
    if ground_truth_ids:
        try:
            metrics["initial"] = calculate_metrics(initial_results, ground_truth_ids)
            metrics["reranked"] = calculate_metrics(reranked_results, ground_truth_ids)
            logger.info(f"메트릭 계산 완료")
        except Exception as e:
            logger.error(f"메트릭 계산 실패: {e}")
    
    # 상위 순위 변화 계산
    try:
        metrics["rank_changes"] = calculate_rank_changes(initial_results, reranked_results)
        logger.info(f"순위 변화 계산 완료")
    except Exception as e:
        logger.error(f"순위 변화 계산 실패: {e}")
    
    return {
        "reranker_model": reranker_model,
        "metrics": metrics,
        "rerank_time": rerank_time if 'rerank_time' in locals() else None,
        "reranked_results": reranked_results
    }


def calculate_metrics(results: List[Dict[str, Any]], ground_truth_ids: List[str]) -> Dict[str, Any]:
    """
    검색 결과 메트릭 계산
    
    Args:
        results: 검색 결과
        ground_truth_ids: 정답 문서 ID 목록
        
    Returns:
        계산된 메트릭
    """
    if not ground_truth_ids:
        return {}
    
    # 결과의 관련성 여부 확인 (1: 관련, 0: 비관련)
    relevance = [1 if doc["id"] in ground_truth_ids else 0 for doc in results]
    
    # Precision@K 계산 (K=3, 5, 10)
    k_values = [3, 5, 10]
    precision_at_k = {}
    for k in k_values:
        if k <= len(results):
            precision_at_k[f"precision@{k}"] = sum(relevance[:k]) / k
    
    # MRR 계산
    for i, rel in enumerate(relevance):
        if rel == 1:
            mrr = 1.0 / (i + 1)
            break
    else:
        mrr = 0.0
    
    # nDCG 계산
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance))
    ideal_relevance = sorted(relevance, reverse=True)
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    
    return {
        "precision_at_k": precision_at_k,
        "mrr": mrr,
        "ndcg": ndcg,
        "relevant_docs": sum(relevance),
        "total_docs": len(relevance)
    }


def calculate_rank_changes(initial_results: List[Dict[str, Any]], 
                          reranked_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    초기 결과와 재순위화 결과 간의 순위 변화 계산
    
    Args:
        initial_results: 초기 검색 결과
        reranked_results: 재순위화 검색 결과
        
    Returns:
        순위 변화 정보
    """
    # ID로 초기 순위 매핑
    initial_ranks = {doc["id"]: i for i, doc in enumerate(initial_results)}
    
    # 재순위화 후 순위 변화 계산
    changes = []
    for i, doc in enumerate(reranked_results):
        if doc["id"] in initial_ranks:
            old_rank = initial_ranks[doc["id"]]
            changes.append({
                "id": doc["id"],
                "old_rank": old_rank + 1,  # 1-based 순위
                "new_rank": i + 1,  # 1-based 순위
                "change": old_rank - i,
                "content_preview": doc["content"][:100] + "..." if len(doc["content"]) > 100 else doc["content"]
            })
    
    # 변화 통계
    improved = sum(1 for c in changes if c["change"] > 0)
    worsened = sum(1 for c in changes if c["change"] < 0)
    unchanged = sum(1 for c in changes if c["change"] == 0)
    
    return {
        "detail": changes,
        "summary": {
            "improved": improved,
            "worsened": worsened,
            "unchanged": unchanged,
            "avg_absolute_change": np.mean([abs(c["change"]) for c in changes]) if changes else 0
        }
    }


def run_reranking_evaluation(
    dataset_path: str,
    config_path: str = None,
    output_path: str = None
) -> Dict[str, Any]:
    """
    재순위화 평가 실행
    
    Args:
        dataset_path: 평가 데이터셋 경로
        config_path: 설정 파일 경로 (기본값: configs/default_config.yaml)
        output_path: 결과 저장 경로
        
    Returns:
        평가 결과
    """
    # 설정 로드
    config = Config(config_path) if config_path else Config()
    
    # 재순위화 모델 목록
    reranking_models = config.retrieval.get('reranking_models', [
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        "jhgan/ko-sroberta-multitask"
    ])
    
    # 리트리버 초기화
    logger.info("DocumentRetriever 초기화")
    retriever = DocumentRetriever(config)
    
    # 평가 데이터셋 로드
    logger.info(f"평가 데이터셋 로드: {dataset_path}")
    dataset = load_evaluation_dataset(dataset_path)
    questions = dataset.get('questions', [])
    
    if not questions:
        logger.error("평가 데이터셋에 질문이 없습니다.")
        return {"error": "No questions in dataset"}
    
    # 평가 결과
    all_results = {}
    
    # 각 질문에 대해 평가 실행
    for question in tqdm(questions, desc="질문 평가"):
        query = question.get('text', '')
        question_id = question.get('id', f"q_{len(all_results)}")
        
        logger.info(f"\n질문 {question_id} 평가: {query}")
        
        # 관련 문서 ID 추출
        ground_truth_ids = []
        gold_standard = question.get('gold_standard', {})
        
        if 'document_ids' in gold_standard:
            ground_truth_ids = gold_standard['document_ids']
        
        # 초기 검색 결과 가져오기
        try:
            logger.info(f"초기 검색 실행")
            initial_results = retriever.retrieve(
                query=query, 
                top_k=10,  # configs/default_config.yaml에서 설정된 최적값
                collections=None  # 모든 컬렉션 검색
            )
            logger.info(f"초기 검색 결과: {len(initial_results)}개")
        except Exception as e:
            logger.error(f"초기 검색 실패: {e}")
            continue
        
        # 각 재순위화 모델 평가
        model_results = {}
        for model in reranking_models:
            logger.info(f"모델 평가: {model}")
            try:
                eval_result = evaluate_reranking(
                    query=query, 
                    initial_results=initial_results.copy(),  # 복사하여 원본 변경 방지
                    reranker_model=model, 
                    ground_truth_ids=ground_truth_ids
                )
                model_results[model] = eval_result
                logger.info(f"모델 {model} 평가 완료")
            except Exception as e:
                logger.error(f"모델 {model} 평가 실패: {e}")
                model_results[model] = {"error": str(e)}
        
        all_results[question_id] = {
            'query': query,
            'model_results': model_results
        }
    
    # 결과 분석
    logger.info("전체 결과 분석")
    analysis = analyze_reranking_results(all_results, reranking_models)
    
    # 결과 저장
    if output_path:
        try:
            results_with_analysis = {
                "results": all_results,
                "analysis": analysis
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results_with_analysis, f, ensure_ascii=False, indent=2)
            logger.info(f"결과 저장 완료: {output_path}")
        except Exception as e:
            logger.error(f"결과 저장 실패: {e}")
    
    return {
        "results": all_results,
        "analysis": analysis
    }


def analyze_reranking_results(evaluation_results, reranking_models):
    """
    재순위화 결과 분석
    
    Args:
        evaluation_results: 평가 결과
        reranking_models: 재순위화 모델 목록
        
    Returns:
        분석 결과
    """
    # 모델별 평균 성능
    model_metrics = {model: {
        'ndcg': [], 
        'mrr': [], 
        'precision@3': [], 
        'precision@5': [],
        'improved_ratio': [],
        'processing_time': []
    } for model in reranking_models}
    
    for question_id, result in evaluation_results.items():
        for model, model_result in result.get('model_results', {}).items():
            if 'error' in model_result:
                continue
                
            metrics = model_result.get('metrics', {})
            
            # 재순위화 메트릭
            if 'reranked' in metrics:
                reranked = metrics['reranked']
                model_metrics[model]['ndcg'].append(reranked.get('ndcg', 0))
                model_metrics[model]['mrr'].append(reranked.get('mrr', 0))
                precision_at_k = reranked.get('precision_at_k', {})
                model_metrics[model]['precision@3'].append(precision_at_k.get('precision@3', 0))
                model_metrics[model]['precision@5'].append(precision_at_k.get('precision@5', 0))
            
            # 처리 시간
            if 'rerank_time' in model_result and model_result['rerank_time'] is not None:
                model_metrics[model]['processing_time'].append(model_result['rerank_time'])
            
            # 순위 향상 비율
            if 'rank_changes' in metrics:
                summary = metrics['rank_changes'].get('summary', {})
                total = summary.get('improved', 0) + summary.get('worsened', 0) + summary.get('unchanged', 0)
                if total > 0:
                    improved_ratio = summary.get('improved', 0) / total
                    model_metrics[model]['improved_ratio'].append(improved_ratio)
    
    # 평균 계산
    results = {}
    for model, metrics in model_metrics.items():
        results[model] = {
            'avg_ndcg': np.mean(metrics['ndcg']) if metrics['ndcg'] else 0,
            'avg_mrr': np.mean(metrics['mrr']) if metrics['mrr'] else 0,
            'avg_precision@3': np.mean(metrics['precision@3']) if metrics['precision@3'] else 0,
            'avg_precision@5': np.mean(metrics['precision@5']) if metrics['precision@5'] else 0,
            'avg_improved_ratio': np.mean(metrics['improved_ratio']) if metrics['improved_ratio'] else 0,
            'avg_processing_time': np.mean(metrics['processing_time']) if metrics['processing_time'] else 0,
            'sample_count': len(metrics['ndcg'])
        }
    
    # 모델 간 비교
    model_comparison = {}
    if len(reranking_models) > 1:
        metrics = ['avg_ndcg', 'avg_mrr', 'avg_precision@3', 'avg_precision@5', 'avg_improved_ratio']
        for metric in metrics:
            model_scores = [(model, results[model][metric]) for model in reranking_models 
                           if model in results and metric in results[model]]
            if model_scores:
                model_scores.sort(key=lambda x: x[1], reverse=True)
                model_comparison[metric] = {
                    'best_model': model_scores[0][0],
                    'best_score': model_scores[0][1],
                    'all_scores': {model: score for model, score in model_scores}
                }
    
    return {
        'model_metrics': results,
        'model_comparison': model_comparison
    }


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="재순위화 모델 평가")
    parser.add_argument("--dataset", required=True, help="평가 데이터셋 경로 (JSON 파일)")
    parser.add_argument("--config", help="설정 파일 경로")
    parser.add_argument("--output", help="결과 저장 경로")
    args = parser.parse_args()
    
    # 평가 실행
    results = run_reranking_evaluation(
        dataset_path=args.dataset,
        config_path=args.config,
        output_path=args.output
    )
    
    # 결과 요약 출력
    print("\n=== 재순위화 모델 평가 결과 ===")
    
    try:
        analysis = results.get('analysis', {})
        model_metrics = analysis.get('model_metrics', {})
        model_comparison = analysis.get('model_comparison', {})
        
        print("\n모델별 평균 성능:")
        for model, metrics in model_metrics.items():
            model_name = model.split('/')[-1]  # 모델 이름 간소화
            print(f"\n{model_name}:")
            print(f"- NDCG: {metrics.get('avg_ndcg', 0):.4f}")
            print(f"- MRR: {metrics.get('avg_mrr', 0):.4f}")
            print(f"- Precision@3: {metrics.get('avg_precision@3', 0):.4f}")
            print(f"- Precision@5: {metrics.get('avg_precision@5', 0):.4f}")
            print(f"- 순위 향상 비율: {metrics.get('avg_improved_ratio', 0):.2f}")
            print(f"- 평균 처리 시간: {metrics.get('avg_processing_time', 0):.4f}초")
            print(f"- 샘플 수: {metrics.get('sample_count', 0)}")
        
        print("\n지표별 최고 성능 모델:")
        for metric, comparison in model_comparison.items():
            best_model = comparison.get('best_model', '').split('/')[-1]  # 모델 이름 간소화
            best_score = comparison.get('best_score', 0)
            print(f"- {metric}: {best_model} ({best_score:.4f})")
        
    except Exception as e:
        print(f"결과 출력 중 오류 발생: {e}")
    
    print("\n평가 완료")


if __name__ == "__main__":
    main() 