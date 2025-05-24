#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RAGAS 평가 결과 시각화 메인 모듈
"""

from .visualization import (
    load_evaluation_results, load_summary_results, load_performance_rankings
)
from .visualization_part2 import (
    process_evaluation_data, process_summary_data, process_ranking_data,
    plot_metrics_comparison, plot_radar_chart, plot_ranking_heatmap
)
from .visualization_part3 import plot_metric_trends
from .visualization_part4 import plot_embedding_reranker_comparison
from .visualization_part6 import generate_html_report

def create_visualization_report(timestamp=None, output_dir='results'):
    """
    RAGAS 평가 결과의 완전한 시각화 보고서 생성
    
    Args:
        timestamp (str, optional): 특정 타임스탬프 데이터만 표시
        output_dir (str, optional): 결과 저장 디렉토리
        
    Returns:
        str: 생성된 HTML 보고서 파일 경로
    """
    # 데이터 로드 및 처리
    summary_results = load_summary_results()
    ranking_results = load_performance_rankings()
    
    summary_df = process_summary_data(summary_results)
    ranking_df = process_ranking_data(ranking_results)
    
    # HTML 보고서 생성
    report_path = generate_html_report(
        summary_df=summary_df,
        ranking_df=ranking_df,
        timestamp=timestamp,
        output_dir=output_dir
    )
    
    return report_path

# 편의 함수들
def plot_all_charts(summary_df=None, ranking_df=None, timestamp=None, output_dir='results'):
    """
    모든 차트를 생성하는 편의 함수
    """
    if summary_df is None:
        summary_results = load_summary_results()
        summary_df = process_summary_data(summary_results)
    
    if ranking_df is None:
        ranking_results = load_performance_rankings()
        ranking_df = process_ranking_data(ranking_results)
    
    # 차트 생성
    charts = {}
    
    if not summary_df.empty:
        charts['metrics_comparison'] = plot_metrics_comparison(summary_df, timestamp, output_dir)
        charts['radar_chart'] = plot_radar_chart(summary_df, timestamp, output_dir)
        charts['metric_trends'] = plot_metric_trends(summary_df, 'average_score', output_dir)
        charts['embedding_reranker'] = plot_embedding_reranker_comparison(summary_df, 'average_score', output_dir)
    
    if not ranking_df.empty:
        charts['ranking_heatmap'] = plot_ranking_heatmap(ranking_df, timestamp, output_dir)
    
    return charts

# 익스포트할 함수들
__all__ = [
    'create_visualization_report',
    'plot_all_charts',
    'load_evaluation_results',
    'load_summary_results', 
    'load_performance_rankings',
    'process_evaluation_data',
    'process_summary_data',
    'process_ranking_data',
    'plot_metrics_comparison',
    'plot_radar_chart',
    'plot_ranking_heatmap',
    'plot_metric_trends',
    'plot_embedding_reranker_comparison',
    'generate_html_report'
]
