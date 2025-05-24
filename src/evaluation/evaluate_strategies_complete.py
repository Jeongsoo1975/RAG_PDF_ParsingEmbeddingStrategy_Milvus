#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RAG 시스템 평가 자동화 스크립트
- 청킹은 조항 단위로 고정
- 임베딩 모델 및 검색 전략 비교
"""

import os
import subprocess
import json
from datetime import datetime
import argparse
import logging
from typing import Dict, List, Any
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
from pathlib import Path

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluation_automation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("evaluation_automation")

# 임베딩 모델 목록
EMBEDDING_MODELS = [
    # 다국어 모델
    {
        "name": "multilingual_mpnet",
        "model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "description": "다국어 MPNet 모델 (기본값)"
    },
    {
        "name": "multilingual_minilm",
        "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "description": "다국어 MiniLM 모델 (경량화)"
    },
    {
        "name": "e5_base_v2",
        "model": "intfloat/multilingual-e5-base",
        "description": "다국어 E5 모델 (Base)"
    },
    
    # 한국어 특화 모델
    {
        "name": "ko_simcse_roberta",
        "model": "BM-K/KoSimCSE-RoBERTa",
        "description": "한국어 SimCSE 모델 (RoBERTa)"
    },
    {
        "name": "ko_roberta_base",
        "model": "klue/roberta-base",
        "description": "한국어 RoBERTa 모델 (KLUE)"
    },
    {
        "name": "ko_simcse",
        "model": "BM-K/KoSimCSE-roberta-multitask",
        "description": "한국어 SimCSE 모델 (RoBERTa Multitask)"
    },
    {
        "name": "ko_sroberta",
        "model": "jhgan/ko-sroberta-multitask",
        "description": "한국어 SRoBERTa 모델"
    },
    {
        "name": "ko_sbert",
        "model": "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
        "description": "한국어 SBERT 모델 (KLUE 데이터셋)"
    }
]

# 검색 전략 설정
SEARCH_STRATEGIES = [
    {
        "name": "vector_only",
        "description": "벡터 검색만 사용 (Hybrid 검색 비활성화)",
        "hybrid_search": False,
        "reranking": False,
        "top_k": 5
    },
    {
        "name": "hybrid_basic",
        "description": "벡터 + 키워드 검색 (기본)",
        "hybrid_search": True,
        "reranking": False,
        "top_k": 5
    },
    {
        "name": "hybrid_rerank",
        "description": "벡터 + 키워드 검색 + 재순위화",
        "hybrid_search": True,
        "reranking": True,
        "top_k": 5
    },
    {
        "name": "top_k_3",
        "description": "상위 3개 결과",
        "hybrid_search": True,
        "reranking": True,
        "top_k": 3
    },
    {
        "name": "top_k_7",
        "description": "상위 7개 결과",
        "hybrid_search": True,
        "reranking": True,
        "top_k": 7
    },
    {
        "name": "top_k_10",
        "description": "상위 10개 결과",
        "hybrid_search": True,
        "reranking": True,
        "top_k": 10
    }
]

def ensure_dirs():
    """필요한 디렉토리 생성"""
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/embedding_models", exist_ok=True)
    os.makedirs("results/search_strategies", exist_ok=True)
    os.makedirs("results/combined", exist_ok=True)
    os.makedirs("results/charts", exist_ok=True)

def run_evaluation(config: Dict[str, Any], output_path: str) -> bool:
    """
    지정된 설정으로 평가 실행
    
    Args:
        config: 평가 설정
        output_path: 결과 저장 경로
        
    Returns:
        성공 여부
    """
    logger.info(f"평가 시작: {config.get('name', 'unnamed')}")
    logger.info(f"설정: {config}")
    
    # 평가 명령어 구성
    cmd = [
        "python", "src/evaluation/rag_evaluator.py",
        "--dataset", "src/evaluation/data/eval_dataset.json",  # TODO: 실제 데이터셋 경로로 변경 필요
        "--output", output_path,
        "--model", config.get("model", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"),
        "--index", config.get("index", "ci-20060401"),
        "--top-k", str(config.get("top_k", 5))
    ]
    
    # 로컬 임베딩 사용 시
    if "embedding_path" in config and config["embedding_path"]:
        cmd.extend(["--embedding_path", config["embedding_path"]])
    
    # 실행
    try:
        logger.info(f"실행 명령어: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"평가 성공: {config.get('name', 'unnamed')}")
            
            # 결과 요약 로그
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                logger.info(f"총 질문: {data['total']['questions']}")
                logger.info(f"정밀도: {data['total']['precision']:.4f}")
                logger.info(f"재현율: {data['total']['recall']:.4f}")
                logger.info(f"F1 점수: {data['total']['f1_score']:.4f}")
                logger.info(f"NDCG: {data['total']['ndcg']:.4f}")
                logger.info(f"답변 포함율: {data['total']['has_answer_rate']:.4f}")
            except Exception as e:
                logger.error(f"결과 파일 분석 실패: {e}")
            
            return True
        else:
            logger.error(f"평가 실패: {config.get('name', 'unnamed')}")
            logger.error(f"오류: {result.stderr}")
            return False
    
    except Exception as e:
        logger.error(f"평가 실행 중 예외 발생: {e}")
        return False

def evaluate_embedding_models():
    """임베딩 모델 평가"""
    logger.info("임베딩 모델 평가 시작")
    
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for model_config in EMBEDDING_MODELS:
        model_name = model_config["name"]
        output_path = f"results/embedding_models/{model_name}_{timestamp}.json"
        
        # 기본 검색 전략 사용 (하이브리드 + 재순위화)
        config = {
            "name": model_name,
            "model": model_config["model"],
            "index": "ci-20060401", # 인덱스 이름 (모든 평가에 동일한 인덱스 사용)
            "hybrid_search": True,
            "reranking": True,
            "top_k": 5
        }
        
        success = run_evaluation(config, output_path)
        
        if success:
            with open(output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            result = {
                "name": model_name,
                "description": model_config["description"],
                "precision": data['total']['precision'],
                "recall": data['total']['recall'],
                "f1_score": data['total']['f1_score'],
                "ndcg": data['total']['ndcg'],
                "has_answer_rate": data['total']['has_answer_rate'],
                "retrieval_time": data['total']['avg_retrieval_time_ms']
            }
            results.append(result)
    
    # 결과 정렬 및 저장
    if results:
        results.sort(key=lambda x: x['f1_score'], reverse=True)
        result_path = f"results/embedding_models_comparison_{timestamp}.json"
        
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"임베딩 모델 비교 결과 저장: {result_path}")
        
        # 결과 테이블 출력
        print("\n===== 임베딩 모델 비교 결과 =====")
        headers = ['모델명', '설명', '정밀도', '재현율', 'F1 점수', 'NDCG', '답변포함', '시간(ms)']
        
        table_data = []
        for r in results:
            table_data.append([
                r['name'], 
                r['description'], 
                f"{r['precision']:.4f}", 
                f"{r['recall']:.4f}", 
                f"{r['f1_score']:.4f}", 
                f"{r['ndcg']:.4f}", 
                f"{r['has_answer_rate']:.4f}", 
                f"{r['retrieval_time']:.1f}"
            ])
        
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
        
        # 결과 시각화
        plot_comparison_chart(results, 'embedding_models', ['precision', 'recall', 'f1_score', 'ndcg', 'has_answer_rate'])
    
    return results

def evaluate_search_strategies():
    """검색 전략 평가"""
    logger.info("검색 전략 평가 시작")
    
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 기본 임베딩 모델 사용 (다국어 MPNet)
    base_model = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    
    for strategy in SEARCH_STRATEGIES:
        strategy_name = strategy["name"]
        output_path = f"results/search_strategies/{strategy_name}_{timestamp}.json"
        
        config = {
            "name": strategy_name,
            "model": base_model,
            "index": "ci-20060401",  # 인덱스 이름 수정
            "hybrid_search": strategy["hybrid_search"],
            "reranking": strategy["reranking"],
            "top_k": strategy["top_k"]
        }
        
        success = run_evaluation(config, output_path)
        
        if success:
            with open(output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            result = {
                "name": strategy_name,
                "description": strategy["description"],
                "precision": data['total']['precision'],
                "recall": data['total']['recall'],
                "f1_score": data['total']['f1_score'],
                "ndcg": data['total']['ndcg'],
                "has_answer_rate": data['total']['has_answer_rate'],
                "retrieval_time": data['total']['avg_retrieval_time_ms']
            }
            results.append(result)
    
    # 결과 정렬 및 저장
    if results:
        results.sort(key=lambda x: x['f1_score'], reverse=True)
        result_path = f"results/search_strategies_comparison_{timestamp}.json"
        
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"검색 전략 비교 결과 저장: {result_path}")
        
        # 결과 테이블 출력
        print("\n===== 검색 전략 비교 결과 =====")
        headers = ['전략명', '설명', '정밀도', '재현율', 'F1 점수', 'NDCG', '답변포함', '시간(ms)']
        
        table_data = []
        for r in results:
            table_data.append([
                r['name'], 
                r['description'], 
                f"{r['precision']:.4f}", 
                f"{r['recall']:.4f}", 
                f"{r['f1_score']:.4f}", 
                f"{r['ndcg']:.4f}", 
                f"{r['has_answer_rate']:.4f}", 
                f"{r['retrieval_time']:.1f}"
            ])
        
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
        
        # 결과 시각화
        plot_comparison_chart(results, 'search_strategies', ['precision', 'recall', 'f1_score', 'ndcg', 'has_answer_rate'])
    
    return results

def evaluate_combined_best(embedding_results, strategy_results):
    """최적 임베딩 모델과 검색 전략 조합 평가"""
    if not embedding_results or not strategy_results:
        logger.error("임베딩 모델 또는 검색 전략 평가 결과가 없습니다.")
        return []
    
    logger.info("최적 조합 평가 시작")
    
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 상위 2개 임베딩 모델
    top_embeddings = sorted(embedding_results, key=lambda x: x['f1_score'], reverse=True)[:2]
    
    # 상위 2개 검색 전략
    top_strategies = sorted(strategy_results, key=lambda x: x['f1_score'], reverse=True)[:2]
    
    # 최적 조합 실험
    for emb in top_embeddings:
        # 원래 모델 설정 찾기
        emb_config = next((m for m in EMBEDDING_MODELS if m["name"] == emb["name"]), None)
        if not emb_config:
            continue
        
        for strat in top_strategies:
            # 원래 전략 설정 찾기
            strat_config = next((s for s in SEARCH_STRATEGIES if s["name"] == strat["name"]), None)
            if not strat_config:
                continue
            
            combined_name = f"{emb['name']}_{strat['name']}"
            output_path = f"results/combined/{combined_name}_{timestamp}.json"
            
            config = {
                "name": combined_name,
                "model": emb_config["model"],
                "index": "ci-20060401",  # 인덱스 이름 수정
                "hybrid_search": strat_config["hybrid_search"],
                "reranking": strat_config["reranking"],
                "top_k": strat_config["top_k"]
            }
            
            success = run_evaluation(config, output_path)
            
            if success:
                with open(output_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                result = {
                    "name": combined_name,
                    "description": f"{emb['description']} + {strat['description']}",
                    "precision": data['total']['precision'],
                    "recall": data['total']['recall'],
                    "f1_score": data['total']['f1_score'],
                    "ndcg": data['total']['ndcg'],
                    "has_answer_rate": data['total']['has_answer_rate'],
                    "retrieval_time": data['total']['avg_retrieval_time_ms']
                }
                results.append(result)
    
    # 결과 정렬 및 저장
    if results:
        results.sort(key=lambda x: x['f1_score'], reverse=True)
        result_path = f"results/combined_best_comparison_{timestamp}.json"
        
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"최적 조합 비교 결과 저장: {result_path}")
        
        # 결과 테이블 출력
        print("\n===== 최적 조합 비교 결과 =====")
        headers = ['조합명', '설명', '정밀도', '재현율', 'F1 점수', 'NDCG', '답변포함', '시간(ms)']
        
        table_data = []
        for r in results:
            table_data.append([
                r['name'], 
                r['description'], 
                f"{r['precision']:.4f}", 
                f"{r['recall']:.4f}", 
                f"{r['f1_score']:.4f}", 
                f"{r['ndcg']:.4f}", 
                f"{r['has_answer_rate']:.4f}", 
                f"{r['retrieval_time']:.1f}"
            ])
        
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
        
        # 결과 시각화
        plot_comparison_chart(results, 'combined_best', ['precision', 'recall', 'f1_score', 'ndcg', 'has_answer_rate'])
    
    return results

def plot_comparison_chart(results: List[Dict[str, Any]], chart_type: str, metrics: List[str]):
    """
    비교 차트 생성
    
    Args:
        results: 비교 결과 목록
        chart_type: 차트 유형 (embedding_models, search_strategies, combined_best)
        metrics: 비교할 메트릭 목록
    """
    if not results:
        logger.warning(f"No results to plot for {chart_type}")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # 데이터 추출
        model_names = [r['name'] for r in results]
        metrics_data = {metric: [r.get(metric, 0) for r in results] for metric in metrics}
        
        # 차트 데이터 준비
        df = pd.DataFrame(metrics_data, index=model_names)
        
        # 성능 메트릭 차트
        fig, ax = plt.figure(figsize=(12, 6)), plt.subplot(111)
        df.plot(kind='bar', ax=ax, width=0.8)
        
        plt.title(f'{chart_type.replace("_", " ").title()} Comparison')
        plt.ylabel('Score')
        plt.ylim(0, 1.0)
        plt.legend(title='Metrics')
        plt.tight_layout()
        
        # 차트 저장
        chart_path = f"results/charts/{chart_type}_comparison_{timestamp}.png"
        plt.savefig(chart_path)
        logger.info(f"차트 저장됨: {chart_path}")
        
        plt.close()
        
        # 시간 성능 차트 (있을 경우)
        if 'retrieval_time' in metrics_data or any('time' in m for m in metrics):
            time_metric = next((m for m in metrics if 'time' in m), None)
            if time_metric:
                fig, ax = plt.figure(figsize=(10, 5)), plt.subplot(111)
                times = [r.get(time_metric, 0) for r in results]
                
                plt.bar(model_names, times, color='skyblue')
                plt.title(f'{chart_type.replace("_", " ").title()} Retrieval Time')
                plt.ylabel('Time (ms)')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                # 차트 저장
                time_chart_path = f"results/charts/{chart_type}_time_comparison_{timestamp}.png"
                plt.savefig(time_chart_path)
                logger.info(f"시간 차트 저장됨: {time_chart_path}")
                
                plt.close()
    
    except Exception as e:
        logger.error(f"차트 생성 중 오류 발생: {e}")

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='RAG 시스템 평가 자동화')
    parser.add_argument('--model', type=str, help='특정 모델만 평가할 경우 모델 이름 지정')
    args = parser.parse_args()
    
    ensure_dirs()
    
    # 특정 모델만 평가하는 경우
    if args.model:
        model_found = False
        for model_config in EMBEDDING_MODELS:
            if model_config["name"] == args.model:
                model_found = True
                logger.info(f"특정 모델 '{args.model}' 평가 시작")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"results/embedding_models/{args.model}_{timestamp}.json"
                
                config = {
                    "name": model_config["name"],
                    "model": model_config["model"],
                    "index": "ci-20060401",
                    "hybrid_search": True,
                    "reranking": True,
                    "top_k": 10  # top_k 10으로 설정
                }
                
                run_evaluation(config, output_path)
                logger.info(f"모델 '{args.model}' 평가 완료")
                return
                
        if not model_found:
            logger.error(f"모델 '{args.model}'을(를) 찾을 수 없습니다.")
            return
    
    # 전체 평가
    embedding_results = evaluate_embedding_models()
    strategy_results = evaluate_search_strategies()
    evaluate_combined_best(embedding_results, strategy_results)
    
    print("\n평가가 완료되었습니다.")
    print("평가 결과는 results/ 디렉토리에서 확인할 수 있습니다.")

if __name__ == "__main__":
    main()
