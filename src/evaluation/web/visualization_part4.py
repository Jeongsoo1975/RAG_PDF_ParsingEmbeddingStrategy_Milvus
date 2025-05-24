#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RAGAS 평가 결과 시각화 모듈 (Part 4: 시각화 함수 추가)
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .visualization import logger, COLOR_PALETTE

def plot_embedding_reranker_comparison(df, metric='average_score', output_dir='results'):
    """
    임베딩 모델과 리랭커 조합의 효과를 히트맵으로 시각화
    """
    if df.empty:
        logger.warning("시각화할 데이터가 없습니다.")
        return None
    
    # 가장 최근 타임스탬프 사용
    latest_timestamp = df['timestamp'].max()
    filtered_df = df[df['timestamp'] == latest_timestamp].copy()
    
    # 각 임베딩-리랭커 조합에 대한 평균 성능
    pivot_df = filtered_df.pivot_table(
        index='embedding_display',
        columns='reranker_display',
        values=metric,
        aggfunc='mean'
    )
    
    # NaN 값을 추가 (임베딩만 사용한 경우를 위해)
    pivot_df.fillna(-1, inplace=True)
    
    # 리랭커 없는 경우의 성능 추가
    no_reranker_df = filtered_df[filtered_df['reranker_model'].isna()]
    for _, row in no_reranker_df.iterrows():
        pivot_df.loc[row['embedding_display'], 'No Reranker'] = row[metric]
    
    # -1 값(실제 NaN)을 다시 NaN으로 변환
    pivot_df = pivot_df.replace(-1, np.nan)
    
    # 그래프 생성
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 히트맵 그리기
    sns.heatmap(
        pivot_df, 
        annot=True, 
        cmap='viridis', 
        fmt='.3f',
        linewidths=.5, 
        ax=ax,
        vmin=0, 
        vmax=1,
        cbar_kws={'label': '점수'}
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
    ax.set_title(f'임베딩-리랭커 조합 {metric_display.get(metric, metric)} 비교', fontsize=16)
    ax.set_xlabel('리랭커', fontsize=14)
    ax.set_ylabel('임베딩 모델', fontsize=14)
    
    plt.tight_layout()
    
    # 결과 저장
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'embedding_reranker_heatmap_{metric}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"임베딩-리랭커 조합 히트맵이 저장되었습니다: {output_path}")
    return output_path

def encode_image_to_base64(image_path):
    """
    이미지 파일을 Base64로 인코딩
    """
    import base64
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string
