#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RAGAS 평가 결과 시각화 모듈 (Part 5: HTML 리포트 생성)
"""

import os
import pandas as pd
from datetime import datetime
from jinja2 import Environment, FileSystemLoader

from .visualization import logger, METRICS
from .visualization_part4 import encode_image_to_base64

def generate_summary_table(df, timestamp=None):
    """
    평가 결과 요약 테이블 생성
    """
    if df.empty:
        return "<p>데이터가 없습니다.</p>"
    
    # 특정 타임스탬프 필터링
    if timestamp:
        filtered_df = df[df['timestamp'] == timestamp].copy()
        if filtered_df.empty:
            return f"<p>타임스탬프 '{timestamp}'에 해당하는 데이터가 없습니다.</p>"
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
    
    # 표시할 열 선택
    display_columns = ['config_label'] + METRICS
    
    # 열 이름 변경
    display_df = filtered_df[display_columns].copy()
    display_df.columns = ['Configuration'] + [metric_display.get(m, m) for m in METRICS]
    
    # 평균 점수 기준 내림차순 정렬
    display_df = display_df.sort_values('Average Score', ascending=False)
    
    # 숫자 형식 지정
    for col in display_df.columns[1:]:  # 첫 번째 열(Configuration)은 제외
        display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
    
    # HTML 테이블 생성
    html_table = display_df.to_html(index=False, classes="table table-striped table-bordered")
    
    return html_table

def create_html_template():
    """
    HTML 템플릿 반환
    """
    return """
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>RAGAS 평가 결과 보고서</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.6.0/dist/css/bootstrap.min.css">
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 20px; }
            .chart-container { margin-bottom: 30px; }
            h1, h2, h3 { color: #333; }
            .timestamp { color: #666; font-size: 0.9em; margin-bottom: 20px; }
            .chart-img { max-width: 100%; height: auto; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
            .table th { background-color: #f8f9fa; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="mt-4 mb-4">RAGAS 평가 결과 보고서</h1>
            <p class="timestamp">생성 시간: {{ generation_time }}<br>평가 시간: {{ evaluation_time }}</p>
            
            <div class="card mb-4">
                <div class="card-header">
                    <h2>성능 요약</h2>
                </div>
                <div class="card-body">
                    {{ summary_table_html|safe }}
                </div>
            </div>
            
            {% if metrics_comparison_img %}
            <div class="chart-container">
                <h3>지표별 성능 비교</h3>
                <img src="data:image/png;base64,{{ metrics_comparison_img }}" alt="지표별 성능 비교" class="chart-img">
            </div>
            {% endif %}
            
            {% if radar_chart_img %}
            <div class="chart-container">
                <h3>레이더 차트</h3>
                <img src="data:image/png;base64,{{ radar_chart_img }}" alt="레이더 차트" class="chart-img">
            </div>
            {% endif %}
            
            {% if ranking_heatmap_img %}
            <div class="chart-container">
                <h3>랭킹 히트맵</h3>
                <img src="data:image/png;base64,{{ ranking_heatmap_img }}" alt="랭킹 히트맵" class="chart-img">
            </div>
            {% endif %}
            
            <footer class="mt-5 mb-3 text-center text-muted">
                <hr>
                <p>Copyright © 2025 RAG 평가 시스템</p>
            </footer>
        </div>
        <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@4.6.0/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    """
