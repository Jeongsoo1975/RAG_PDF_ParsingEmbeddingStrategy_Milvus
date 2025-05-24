#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RAGAS 평가 결과 시각화 실행 스크립트
"""

import os
import sys
import argparse

# 프로젝트 루트를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.web.visualization_main import create_visualization_report

def main():
    parser = argparse.ArgumentParser(description='RAGAS 평가 결과 시각화 도구')
    parser.add_argument(
        '--timestamp', 
        type=str, 
        help='특정 타임스탬프의 결과만 시각화 (예: 20250519_203504)',
        default=None
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='결과를 저장할 디렉토리',
        default='results'
    )
    
    args = parser.parse_args()
    
    print("RAGAS 평가 결과 시각화를 시작합니다...")
    print(f"타임스탬프: {args.timestamp or '최신'}")
    print(f"출력 디렉토리: {args.output_dir}")
    
    try:
        report_path = create_visualization_report(
            timestamp=args.timestamp,
            output_dir=args.output_dir
        )
        
        if report_path:
            print("\n성공! 시각화 보고서가 성공적으로 생성되었습니다!")
            print(f"보고서 위치: {report_path}")
            print(f"\n브라우저에서 다음 파일을 열어 결과를 확인하세요:")
            print(f"file://{os.path.abspath(report_path)}")
        else:
            print("오류: 보고서 생성에 실패했습니다.")
            return 1
            
    except Exception as e:
        print(f"오류가 발생했습니다: {str(e)}")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
