#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RAGAS 평가 결과 시각화 모듈 (Part 2: 데이터 처리 및 시각화 함수)
"""

import os
import json
import logging
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Part 1 모듈에서 상수 및 기본 함수 가져오기
from .visualization import (
    logger, COLOR_PALETTE, METRICS, MODEL_DISPLAY_NAMES, 
    EVAL_RESULTS_DIR, VisualizationError
)

def get_model_name_from_config(config_str):
    """
    설정 문자열에서 모델 이름 추출
    
    Args:
        config_str (str): 설정 문자열 (예: 'emb_jhgan_ko-sroberta-multitask_reranker_...')
        
    Returns:
        tuple: (임베딩 모델명, 리랭커 모델명)
    """
    try:
        parts = config_str.split('_')
        
        # 임베딩 모델 정보 추출
        emb_idx = parts.index('emb')
        embedding_model = None
        
        # emb 다음 부분부터 처리
        for i in range(emb_idx + 1, len(parts)):
            if parts[i] in ['reranker', 'top']:
                break
            if embedding_model is None:
                embedding_model = parts[i]
            else:
                embedding_model += '_' + parts[i]
        
        # 리랭커 정보 추출 (있는 경우)
        reranker_model = None
        if 'reranker' in parts:
            reranker_idx = parts.index('reranker')
            for i in range(reranker_idx + 1, len(parts)):
                if parts[i] in ['top']:
                    break
                if reranker_model is None:
                    reranker_model = parts[i]
                else:
                    reranker_model += '_' + parts[i]
        
        return embedding_model, reranker_model
    
    except Exception as e:
        logger.warning(f"설정 문자열 '{config_str}'에서 모델 정보를 추출할 수 없습니다: {str(e)}")
        return None, None

def get_display_name(model_name):
    """
    모델 이름에 대한 표시용 이름 반환
    
    Args:
        model_name (str): 모델 이름
        
    Returns:
        str: 표시용 이름
    """
    if model_name in MODEL_DISPLAY_NAMES:
        return MODEL_DISPLAY_NAMES[model_name]
    return model_name

def process_evaluation_data(eval_results):
    """
    평가 결과를 분석하기 쉬운 데이터프레임으로 변환
    
    Args:
        eval_results (dict): 로드된 평가 결과 사전
        
    Returns:
        pandas.DataFrame: 처리된 데이터프레임
    """
    all_data = []
    
    for file_name, result_info in eval_results.items():
        data = result_info['data']
        timestamp = result_info['timestamp']
        
        try:
            # 설정 정보 추출
            config_str = '_'.join(file_name.split('_')[3:]).replace('.json', '')
            emb_model, reranker_model = get_model_name_from_config(config_str)
            
            # 평가 지표 추출
            metrics_data = data.get('ragas_metrics', {})
            
            # 각 지표에 대한 평균 계산
            row = {
                'file_name': file_name,
                'timestamp': timestamp,
                'embedding_model': emb_model,
                'reranker_model': reranker_model,
                'embedding_display': get_display_name(emb_model),
                'reranker_display': get_display_name(reranker_model) if reranker_model else None,
                'config': config_str
            }
            
            # 평가 지표 추가
            for metric in METRICS[:-1]:  # average_score 제외
                if metric in metrics_data:
                    values = metrics_data[metric]
                    if isinstance(values, list):
                        row[metric] = np.mean(values)
                    else:
                        row[metric] = values
                else:
                    row[metric] = None
            
            # 평균 점수 계산 (추가된 지표)
            values = [row.get(m) for m in METRICS[:-1] if row.get(m) is not None]
            if values:
                row['average_score'] = np.mean(values)
            else:
                row['average_score'] = None
            
            all_data.append(row)
            
        except Exception as e:
            logger.warning(f"파일 '{file_name}' 처리 중 오류 발생: {str(e)}")
    
    # 데이터프레임 생성 및 정렬
    if all_data:
        df = pd.DataFrame(all_data)
        df = df.sort_values(['timestamp', 'embedding_model', 'reranker_model'])
        return df
    else:
        logger.warning("처리할 데이터가 없습니다.")
        return pd.DataFrame()

def process_summary_data(summary_results):
    """
    요약 결과를 분석하기 쉬운 데이터프레임으로 변환
    
    Args:
        summary_results (dict): 로드된 요약 결과 사전
        
    Returns:
        pandas.DataFrame: 처리된 데이터프레임
    """
    all_data = []
    
    for file_name, result_info in summary_results.items():
        data = result_info['data']
        timestamp = result_info['timestamp']
        
        try:
            # 각 설정에 대한 결과 추출
            for config, metrics in data.items():
                emb_model, reranker_model = get_model_name_from_config(config)
                
                row = {
                    'file_name': file_name,
                    'timestamp': timestamp,
                    'embedding_model': emb_model,
                    'reranker_model': reranker_model,
                    'embedding_display': get_display_name(emb_model),
                    'reranker_display': get_display_name(reranker_model) if reranker_model else None,
                    'config': config
                }
                
                # 평가 지표 추가
                for metric in METRICS[:-1]:  # average_score 제외
                    if metric in metrics:
                        row[metric] = metrics[metric]
                    else:
                        row[metric] = None
                
                # 평균 점수 계산 (있다면 사용, 없으면 계산)
                if 'average_score' in metrics:
                    row['average_score'] = metrics['average_score']
                else:
                    values = [row.get(m) for m in METRICS[:-1] if row.get(m) is not None]
                    if values:
                        row['average_score'] = np.mean(values)
                    else:
                        row['average_score'] = None
                
                all_data.append(row)
                
        except Exception as e:
            logger.warning(f"요약 파일 '{file_name}' 처리 중 오류 발생: {str(e)}")
    
    # 데이터프레임 생성 및 정렬
    if all_data:
        df = pd.DataFrame(all_data)
        df = df.sort_values(['timestamp', 'embedding_model', 'reranker_model'])
        return df
    else:
        logger.warning("처리할 요약 데이터가 없습니다.")
        return pd.DataFrame()

def process_ranking_data(ranking_results):
    """
    랭킹 결과를 분석하기 쉬운 데이터프레임으로 변환
    
    Args:
        ranking_results (dict): 로드된 랭킹 결과 사전
        
    Returns:
        pandas.DataFrame: 처리된 데이터프레임
    """
    all_data = []
    
    for file_name, result_info in ranking_results.items():
        data = result_info['data']
        timestamp = result_info['timestamp']
        
        try:
            # 각 지표별 랭킹 추출
            for metric, rankings in data.items():
                if metric == 'overall_ranking':
                    metric_name = 'average_score'
                else:
                    metric_name = metric
                
                for rank_info in rankings:
                    config = rank_info['config']
                    score = rank_info['score']
                    rank = rank_info['rank']
                    
                    emb_model, reranker_model = get_model_name_from_config(config)
                    
                    row = {
                        'file_name': file_name,
                        'timestamp': timestamp,
                        'embedding_model': emb_model,
                        'reranker_model': reranker_model,
                        'embedding_display': get_display_name(emb_model),
                        'reranker_display': get_display_name(reranker_model) if reranker_model else None,
                        'config': config,
                        'metric': metric_name,
                        'score': score,
                        'rank': rank
                    }
                    
                    all_data.append(row)
                    
        except Exception as e:
            logger.warning(f"랭킹 파일 '{file_name}' 처리 중 오류 발생: {str(e)}")
    
    # 데이터프레임 생성 및 정렬
    if all_data:
        df = pd.DataFrame(all_data)
        df = df.sort_values(['timestamp', 'metric', 'rank'])
        return df
    else:
        logger.warning("처리할 랭킹 데이터가 없습니다.")
        return pd.DataFrame()

def plot_metrics_comparison(df, timestamp=None, output_dir='results'):
    """
    다양한 구성의 평가 지표를 비교하는 막대 그래프 생성
    
    Args:
        df (pandas.DataFrame): 처리된 평가 데이터
        timestamp (str, optional): 특정 타임스탬프 데이터만 표시
        output_dir (str, optional): 결과 저장 디렉토리
        
    Returns:
        str: 저장된 이미지 파일 경로
    """
    # 특정 타임스탬프 필터링
    if timestamp:
        filtered_df = df[df['timestamp'] == timestamp].copy()
        if filtered_df.empty:
            logger.warning(f"타임스탬프 '{timestamp}'에 해당하는 데이터가 없습니다.")
            return None
    else:
        # 가장 최근 타임스탬프 사용
        latest_timestamp = df['timestamp'].max()
        filtered_df = df[df['timestamp'] == latest_timestamp].copy()
    
    # 구성 레이블 생성
    filtered_df['config_label'] = filtered_df.apply(
        lambda row: f"{row['embedding_display']}" + 
                   (f" + {row['reranker_display']}" if pd.notna(row['reranker_display']) else ""),
        axis=1
    )
    
    # 그래프 생성
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 데이터 구조 변환 (long format)
    plot_data = pd.melt(
        filtered_df, 
        id_vars=['config_label'], 
        value_vars=METRICS,
        var_name='Metric', 
        value_name='Score'
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
    
    plot_data['Metric'] = plot_data['Metric'].map(metric_display)
    
    # 그래프 그리기
    sns.barplot(
        data=plot_data, 
        x='Metric', 
        y='Score', 
        hue='config_label',
        palette='viridis',
        ax=ax
    )
    
    # 그래프 스타일 설정
    ax.set_title('RAGAS 평가 지표 비교', fontsize=16)
    ax.set_xlabel('평가 지표', fontsize=14)
    ax.set_ylabel('점수', fontsize=14)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 범례 위치 조정
    plt.legend(title='구성', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # 결과 저장
    os.makedirs(output_dir, exist_ok=True)
    timestamp_str = timestamp or filtered_df['timestamp'].iloc[0]
    output_path = os.path.join(output_dir, f'metrics_comparison_{timestamp_str}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"평가 지표 비교 그래프가 저장되었습니다: {output_path}")
    return output_path

def plot_radar_chart(df, timestamp=None, output_dir='results'):
    """
    각 구성의 성능을 레이더 차트로 시각화
    
    Args:
        df (pandas.DataFrame): 처리된 평가 데이터
        timestamp (str, optional): 특정 타임스탬프 데이터만 표시
        output_dir (str, optional): 결과 저장 디렉토리
        
    Returns:
        str: 저장된 이미지 파일 경로
    """
    # 특정 타임스탬프 필터링
    if timestamp:
        filtered_df = df[df['timestamp'] == timestamp].copy()
        if filtered_df.empty:
            logger.warning(f"타임스탬프 '{timestamp}'에 해당하는 데이터가 없습니다.")
            return None
    else:
        # 가장 최근 타임스탬프 사용
        latest_timestamp = df['timestamp'].max()
        filtered_df = df[df['timestamp'] == latest_timestamp].copy()
    
    # 구성 레이블 생성
    filtered_df['config_label'] = filtered_df.apply(
        lambda row: f"{row['embedding_display']}" + 
                   (f" + {row['reranker_display']}" if pd.notna(row['reranker_display']) else ""),
        axis=1
    )
    
    # 레이더 차트용 지표 선택 (average_score 제외)
    metrics = METRICS[:-1]
    
    # 각 구성에 대한 데이터 준비
    configs = filtered_df['config_label'].unique()
    
    # 메트릭 표시 이름 변환
    metric_display = {
        'context_precision': 'Context Precision',
        'context_recall': 'Context Recall',
        'faithfulness': 'Faithfulness',
        'answer_relevancy': 'Answer Relevancy',
        'context_relevancy': 'Context Relevancy'
    }
    
    # 그래프 설정
    n_metrics = len(metrics)
    angles = np.linspace(0, 2*np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # 첫 번째 요소를 마지막에 추가하여 폐곡선 완성
    
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'polar': True})
    
    # 각 구성에 대한 레이더 차트 그리기
    for i, config in enumerate(configs):
        values = filtered_df[filtered_df['config_label'] == config][metrics].values[0].tolist()
        values += values[:1]  # 첫 번째 값을 마지막에 추가
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=config)
        ax.fill(angles, values, alpha=0.1)
    
    # 축 레이블 설정
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([metric_display[m] for m in metrics])
    
    # 그래프 스타일 설정
    ax.set_ylim(0, 1)
    plt.title('RAGAS 평가 지표 레이더 차트', fontsize=16, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # 결과 저장
    os.makedirs(output_dir, exist_ok=True)
    timestamp_str = timestamp or filtered_df['timestamp'].iloc[0]
    output_path = os.path.join(output_dir, f'radar_chart_{timestamp_str}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"레이더 차트가 저장되었습니다: {output_path}")
    return output_path

def plot_ranking_heatmap(df, timestamp=None, output_dir='results'):
    """
    각 지표별 랭킹 결과를 히트맵으로 시각화
    
    Args:
        df (pandas.DataFrame): 처리된 랭킹 데이터
        timestamp (str, optional): 특정 타임스탬프 데이터만 표시
        output_dir (str, optional): 결과 저장 디렉토리
        
    Returns:
        str: 저장된 이미지 파일 경로
    """
    # 특정 타임스탬프 필터링
    if timestamp:
        filtered_df = df[df['timestamp'] == timestamp].copy()
        if filtered_df.empty:
            logger.warning(f"타임스탬프 '{timestamp}'에 해당하는 데이터가 없습니다.")
            return None
    else:
        # 가장 최근 타임스탬프 사용
        latest_timestamp = df['timestamp'].max()
        filtered_df = df[df['timestamp'] == latest_timestamp].copy()
    
    # 구성 레이블 생성
    filtered_df['config_label'] = filtered_df.apply(
        lambda row: f"{row['embedding_display']}" + 
                   (f" + {row['reranker_display']}" if pd.notna(row['reranker_display']) else ""),
        axis=1
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
    
    # 피벗 테이블 생성
    pivot_df = filtered_df.pivot_table(
        index='config_label', 
        columns='metric', 
        values='rank',
        aggfunc='first'
    )
    
    # 열 이름 변경
    pivot_df.columns = [metric_display.get(col, col) for col in pivot_df.columns]
    
    # 그래프 생성
    fig, ax = plt.subplots(figsize=(14, len(pivot_df) * 0.8 + 2))
    
    # 히트맵 그리기
    sns.heatmap(
        pivot_df, 
        annot=True, 
        cmap='YlGnBu_r',  # 역순 색상 (낮은 랭크(1등)이 더 진한 색)
        fmt='d',
        linewidths=.5, 
        ax=ax,
        cbar_kws={'label': '랭킹 (낮을수록 좋음)'}
    )
    
    # 그래프 스타일 설정
    ax.set_title('RAGAS 평가 지표별 랭킹', fontsize=16)
    ax.set_xlabel('평가 지표', fontsize=14)
    ax.set_ylabel('모델 구성', fontsize=14)
    
    plt.tight_layout()
    
    # 결과 저장
    os.makedirs(output_dir, exist_ok=True)
    timestamp_str = timestamp or filtered_df['timestamp'].iloc[0]
    output_path = os.path.join(output_dir, f'ranking_heatmap_{timestamp_str}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"랭킹 히트맵이 저장되었습니다: {output_path}")
    return output_path
