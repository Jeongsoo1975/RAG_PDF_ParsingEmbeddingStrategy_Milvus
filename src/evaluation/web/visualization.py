#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RAGAS 평가 결과 시각화 모듈 (Part 1: 기본 설정 및 데이터 로딩)
"""

import os
import json
import logging
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from pathlib import Path

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'visualization.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('visualization')

# 폰트 및 스타일 설정
plt.style.use('seaborn-v0_8-whitegrid')

# 한글 폰트 설정
try:
    import platform
    if platform.system() == 'Windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'
    else:
        plt.rcParams['font.family'] = 'DejaVu Sans'
except:
    plt.rcParams['font.family'] = 'sans-serif'
    
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

# 색상 설정
COLOR_PALETTE = {
    'embedding': '#1f77b4',  # 파란색
    'reranker': '#ff7f0e',   # 주황색 
    'combined': '#2ca02c',   # 초록색
    'default': '#7f7f7f',    # 회색
    'highlight': '#d62728',  # 빨간색
    'background': '#f5f5f5'  # 연한 회색
}

# 평가 지표 설정
METRICS = [
    'context_precision', 
    'context_recall',
    'faithfulness', 
    'answer_relevancy', 
    'context_relevancy',
    'average_score'
]

# 모델 매핑 정보
MODEL_DISPLAY_NAMES = {
    'sentence-transformers_paraphrase-multilingual-mpnet-base-v2': 'Multilingual MPNet',
    'jhgan_ko-sroberta-multitask': 'Ko-SRoBERTa',
    'BM-K_KoSimCSE-roberta-multitask': 'KoSimCSE',
    'snunlp_KR-SBERT-V40K-klueNLI-augSTS': 'KR-SBERT',
    'cross-encoder_ms-marco-MiniLM-L-6-v2': 'MS-Marco-MiniLM'
}

# 평가 결과 파일 경로
EVAL_RESULTS_DIR = os.path.join(os.getcwd(), 'evaluation_results')

class VisualizationError(Exception):
    """시각화 과정에서 발생하는 오류를 위한 사용자 정의 예외"""
    pass

def load_summary_results(pattern=None):
    """
    요약 결과 파일들(ALL_EVALUATION_RUNS_SUMMARY_*.json)을 로드
    """
    results = {}
    
    if pattern:
        file_pattern = os.path.join(EVAL_RESULTS_DIR, pattern)
    else:
        file_pattern = os.path.join(EVAL_RESULTS_DIR, "ALL_EVALUATION_RUNS_SUMMARY_*.json")
    
    try:
        files = glob.glob(file_pattern)
        if not files:
            logger.warning(f"패턴에 일치하는 파일이 없습니다.")
            return results
            
        for file_path in files:
            file_name = os.path.basename(file_path)
            logger.info(f"요약 파일 로드 중: {file_name}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    parts = file_name.replace('.json', '').split('_')
                    timestamp = '_'.join(parts[-2:])
                    
                    results[file_name] = {
                        'data': data,
                        'timestamp': timestamp,
                        'file_path': file_path
                    }
                except json.JSONDecodeError:
                    logger.error(f"파일을 JSON으로 파싱할 수 없습니다: {file_path}")
        
        logger.info(f"{len(results)}개의 요약 파일을 로드했습니다.")
        return results
    
    except Exception as e:
        logger.error(f"요약 파일 로드 중 오류 발생: {str(e)}")
        return {}

def load_performance_rankings(pattern=None):
    """
    성능 랭킹 파일들(PERFORMANCE_RANKING_*.json)을 로드
    """
    results = {}
    
    if pattern:
        file_pattern = os.path.join(EVAL_RESULTS_DIR, pattern)
    else:
        file_pattern = os.path.join(EVAL_RESULTS_DIR, "PERFORMANCE_RANKING_*.json")
    
    try:
        files = glob.glob(file_pattern)
        if not files:
            logger.warning(f"패턴에 일치하는 파일이 없습니다.")
            return results
            
        for file_path in files:
            file_name = os.path.basename(file_path)
            logger.info(f"랭킹 파일 로드 중: {file_name}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    parts = file_name.replace('.json', '').split('_')
                    timestamp = '_'.join(parts[-2:])
                    
                    results[file_name] = {
                        'data': data,
                        'timestamp': timestamp,
                        'file_path': file_path
                    }
                except json.JSONDecodeError:
                    logger.error(f"파일을 JSON으로 파싱할 수 없습니다: {file_path}")
        
        logger.info(f"{len(results)}개의 랭킹 파일을 로드했습니다.")
        return results
    
    except Exception as e:
        logger.error(f"랭킹 파일 로드 중 오류 발생: {str(e)}")
        return {}

def load_evaluation_results(pattern=None):
    """간단한 평가 결과 로더"""
    return {}
