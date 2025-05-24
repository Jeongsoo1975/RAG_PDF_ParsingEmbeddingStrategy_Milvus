#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Retrieval 평가 결과 시각화 스크립트 (임시)
"""

import os
import sys
import json
import glob
import matplotlib
matplotlib.use('Agg')  # GUI 백엔드 비활성화
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path

def load_summary_data():
    """요약 데이터 로드"""
    eval_dir = "evaluation_results"
    pattern = os.path.join(eval_dir, "ALL_EVALUATION_RUNS_SUMMARY_*.json")
    files = glob.glob(pattern)
    
    all_data = []
    
    for file_path in files:
        print(f"파일 로드: {os.path.basename(file_path)}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 파일명에서 타임스탬프 추출
        filename = os.path.basename(file_path)
        timestamp = filename.split('_')[-1].replace('.json', '')
        
        # evaluation_run_summaries 처리
        if 'evaluation_run_summaries' in data:
            for summary in data['evaluation_run_summaries']:
                row = {
                    'timestamp': timestamp,
                    'embedding_model': summary.get('embedding_model', ''),
                    'reranker_model': summary.get('reranker_model', None),
                    'precision': summary.get('precision', 0),
                    'recall': summary.get('recall', 0),
                    'f1_score': summary.get('f1_score', 0),
                    'ndcg': summary.get('ndcg', 0),
                    'mrr': summary.get('mrr', 0),
                    'has_answer_rate': summary.get('has_answer_rate', 0)
                }
                # 평균 점수 계산
                scores = [row['precision'], row['recall'], row['f1_score'], row['ndcg'], row['mrr']]
                row['average_score'] = np.mean([s for s in scores if s > 0])
                
                all_data.append(row)
    
    return pd.DataFrame(all_data)

def create_model_labels(df):
    """모델 레이블 생성"""
    df['embedding_display'] = df['embedding_model'].str.replace('jhgan/', '').str.replace('-', ' ')
    df['reranker_display'] = df['reranker_model'].fillna('None').str.replace('cross-encoder/', '').str.replace('-', ' ')
    df['config_label'] = df['embedding_display'] + ' + ' + df['reranker_display']
    df.loc[df['reranker_model'].isna(), 'config_label'] = df.loc[df['reranker_model'].isna(), 'embedding_display']
    return df

def plot_metrics_comparison(df, output_dir='results'):
    """메트릭 비교 차트"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 최신 데이터만 사용
    latest_timestamp = df['timestamp'].max()
    latest_df = df[df['timestamp'] == latest_timestamp].copy()
    
    # 중복 제거 (같은 설정의 여러 실행이 있을 수 있음)
    latest_df = latest_df.drop_duplicates(subset=['embedding_model', 'reranker_model'])
    
    # 메트릭 선택
    metrics = ['precision', 'recall', 'f1_score', 'ndcg', 'mrr']
    
    # 데이터 재구성
    plot_data = []
    for _, row in latest_df.iterrows():
        for metric in metrics:
            plot_data.append({
                'config': row['config_label'][:30],  # 라벨 길이 제한
                'metric': metric.upper(),
                'score': row[metric]
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    # 플롯 생성
    plt.figure(figsize=(14, 8))
    sns.barplot(data=plot_df, x='metric', y='score', hue='config', palette='viridis')
    plt.ylabel('Score', fontsize=14)
    plt.xlabel('Metrics', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(0, 1.0)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'metrics_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"메트릭 비교 차트 저장: {output_path}")
    return output_path
    return output_path

def plot_performance_summary(df, output_dir='results'):
    """성능 요약 테이블"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 최신 데이터만 사용
    latest_timestamp = df['timestamp'].max()
    latest_df = df[df['timestamp'] == latest_timestamp].copy()
    
    # 중복 제거
    summary_df = latest_df.drop_duplicates(subset=['embedding_model', 'reranker_model'])
    
    # 점수순 정렬
    summary_df = summary_df.sort_values('average_score', ascending=False)
    
    # 테이블 생성
    plt.figure(figsize=(12, 6))
    plt.axis('tight')
    plt.axis('off')
    
    # 데이터 준비
    table_data = []
    for _, row in summary_df.iterrows():
        table_data.append([
            row['config_label'][:40],
            f"{row['precision']:.3f}",
            f"{row['recall']:.3f}",
            f"{row['f1_score']:.3f}",
            f"{row['ndcg']:.3f}",
            f"{row['mrr']:.3f}",
            f"{row['average_score']:.3f}"
        ])
    
    headers = ['Configuration', 'Precision', 'Recall', 'F1', 'NDCG', 'MRR', 'Average']
    
    table = plt.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    
    output_path = os.path.join(output_dir, 'performance_summary.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"성능 요약 테이블 저장: {output_path}")
    return output_path
    plt.title('Performance Summary', fontsize=16, pad=20)
    
    output_path = os.path.join(output_dir, 'performance_summary.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 성능 요약 테이블 저장: {output_path}")
    return output_path

def main():
    try:
        print("Retrieval 평가 결과 시각화 시작...")
        
        # 데이터 로드
        df = load_summary_data()
        if df.empty:
            print("로드할 데이터가 없습니다.")
            return 1
        
        print(f"{len(df)}개의 평가 결과 로드 완료")
        
        # 모델 레이블 생성
        df = create_model_labels(df)
        
        # 시각화 생성
        plot_metrics_comparison(df)
        plot_performance_summary(df)
        
        print("시각화 완료! results/ 폴더를 확인하세요.")
        return 0
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
if __name__ == '__main__':
    sys.exit(main())
