#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RAGAS 평가 결과 시각화 모듈 (Part 3: 추가 시각화 및 웹 인터페이스 함수)
"""

import os
import json
import logging
import glob
import base64
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from datetime import datetime
from jinja2 import Environment, FileSystemLoader

# Part 1, 2 모듈에서 함수 및 상수 가져오기
from .visualization import (
    logger, COLOR_PALETTE, METRICS, MODEL_DISPLAY_NAMES, 
    EVAL_RESULTS_DIR, VisualizationError,
    load_evaluation_results, load_summary_results, load_performance_rankings
)
from .visualization_part2 import (
    process_evaluation_data, process_summary_data, process_ranking_data,
    plot_metrics_comparison, plot_radar_chart, plot_ranking_heatmap,
    get_model_name_from_config, get_display_name
)

def plot_metric_trends(df, metric='average_score', output_dir='results'):
    """
    시간에 따른 특정 지표의 변화 추이를 선 그래프로 시각화
    
    Args:
        df (pandas.DataFrame): 처리된 평가 데이터
        metric (str, optional): 표시할 평가 지표
        output_dir (str, optional): 결과 저장 디렉토리
        
    Returns:
        str: 저장된 이미지 파일 경로
    """
    if df.empty:
        logger.warning("시각화할 데이터가 없습니다.")
        return None
    
    # 타임스탬프별로 데이터 정렬
    df_sorted = df.sort_values('timestamp')
    
    # 구성 레이블 생성
    df_sorted['config_label'] = df_sorted.apply(
        lambda row: f"{row['embedding_display']}" + 
                  (f" + {row['reranker_display']}" if pd.notna(row['reranker_display']) else ""),
        axis=1
    )
    
    # 그래프 생성
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 각 구성에 대한 라인 플롯 생성
    for config, group in df_sorted.groupby('config_label'):
        ax.plot(
            group['timestamp'], 
            group[metric], 
            marker='o', 
            linewidth=2, 
            label=config
        )
    
    # 메트릭 표시 이름 변환
    metric_display = {
        'context_precision': 'Context Precision',
        'context_recall': 'Context Recall',
        'faithfulness': 'Faithfulness',
        'answer_relevancy': 'Answer Relevancy',
        'context_relevancy': 'Context Relevancy',
        'average_score': 'Average Score'
    }
    
    # 그래프 스타일 설정
    ax.set_title(f'{metric_display.get(metric, metric)} 시간별 추이', fontsize=16)
    ax.set_xlabel('평가 시간', fontsize=14)
    ax.set_ylabel('점수', fontsize=14)
    ax.set_ylim(0, 1.0)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # X축 레이블 회전
    plt.xticks(rotation=45)
    
    # 범례 위치 조정
    plt.legend(title='구성', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # 결과 저장
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{metric}_trend.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"{metric} 추이 그래프가 저장되었습니다: {output_path}")
    return output_path