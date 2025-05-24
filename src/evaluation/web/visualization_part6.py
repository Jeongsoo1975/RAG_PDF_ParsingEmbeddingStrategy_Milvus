#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RAGAS 평가 결과 시각화 모듈 (Part 6: HTML 리포트 생성 메인 함수)
"""

import os
import pandas as pd
from datetime import datetime
from jinja2 import Environment, BaseLoader

from .visualization import (
    logger, load_summary_results, load_performance_rankings
)
from .visualization_part2 import (
    process_summary_data, process_ranking_data,
    plot_metrics_comparison, plot_radar_chart, plot_ranking_heatmap
)
from .visualization_part3 import plot_metric_trends
from .visualization_part4 import plot_embedding_reranker_comparison, encode_image_to_base64
from .visualization_part5 import generate_summary_table, create_html_template

def generate_html_report(summary_df=None, ranking_df=None, timestamp=None, output_dir='results'):
    """
    모든 시각화 결과를 포함하는 HTML 보고서 생성
    
    Args:
        summary_df (pandas.DataFrame, optional): 처리된 요약 데이터
        ranking_df (pandas.DataFrame, optional): 처리된 랭킹 데이터
        timestamp (str, optional): 특정 타임스탬프 데이터만 표시
        output_dir (str, optional): 결과 저장 디렉토리
        
    Returns:
        str: 저장된 HTML 파일 경로
    """
    # 필요한 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 결과 로드
    if summary_df is None:
        try:
            summary_results = load_summary_results()
            summary_df = process_summary_data(summary_results)
        except Exception as e:
            logger.error(f"요약 데이터 로드 중 오류 발생: {str(e)}")
            summary_df = pd.DataFrame()
    
    if ranking_df is None:
        try:
            ranking_results = load_performance_rankings()
            ranking_df = process_ranking_data(ranking_results)
        except Exception as e:
            logger.error(f"랭킹 데이터 로드 중 오류 발생: {str(e)}")
            ranking_df = pd.DataFrame()
    
    # 타임스탬프 사용: 지정된 타임스탬프가 없으면 가장 최근 것 사용
    if timestamp is None and not summary_df.empty:
        timestamp = summary_df['timestamp'].max()
    
    # 시각화 실행
    base64_images = {}
    try:
        if not summary_df.empty:
            metrics_comparison_path = plot_metrics_comparison(summary_df, timestamp, output_dir)
            if metrics_comparison_path:
                base64_images['metrics_comparison_img'] = encode_image_to_base64(metrics_comparison_path)
                
            radar_chart_path = plot_radar_chart(summary_df, timestamp, output_dir)
            if radar_chart_path:
                base64_images['radar_chart_img'] = encode_image_to_base64(radar_chart_path)
                
            metric_trend_path = plot_metric_trends(summary_df, 'average_score', output_dir)
            if metric_trend_path:
                base64_images['metric_trend_img'] = encode_image_to_base64(metric_trend_path)
                
            embedding_reranker_path = plot_embedding_reranker_comparison(summary_df, 'average_score', output_dir)
            if embedding_reranker_path:
                base64_images['embedding_reranker_img'] = encode_image_to_base64(embedding_reranker_path)
        
        if not ranking_df.empty:
            ranking_heatmap_path = plot_ranking_heatmap(ranking_df, timestamp, output_dir)
            if ranking_heatmap_path:
                base64_images['ranking_heatmap_img'] = encode_image_to_base64(ranking_heatmap_path)
        
        # 요약 테이블 생성
        summary_table_html = generate_summary_table(summary_df, timestamp)
        
        # HTML 템플릿 로드 및 렌더링
        template_str = create_html_template()
        env = Environment(loader=BaseLoader())
        template = env.from_string(template_str)
        
        # HTML 렌더링
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_html = template.render(
            generation_time=now,
            evaluation_time=timestamp or 'N/A',
            summary_table_html=summary_table_html,
            **base64_images
        )
        
        # HTML 파일 저장
        timestamp_str = timestamp.replace(':', '').replace(' ', '_') if timestamp else 'latest'
        output_path = os.path.join(output_dir, f'ragas_report_{timestamp_str}.html')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_html)
        
        logger.info(f"HTML 보고서가 생성되었습니다: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"HTML 보고서 생성 중 오류 발생: {str(e)}")
        return None
