#!/usr/bin/env python3
"""
단계별 RAG 평가: 2단계 - RAGAS 구조 검증
수동 답변 데이터와 Milvus 검색 결과를 활용한 간소화된 RAGAS 평가
"""

import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def print_section_header(title: str):
    """섹션 헤더 출력"""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")

def validate_prerequisites() -> Dict[str, bool]:
    """필수 조건 확인"""
    print_section_header("필수 조건 확인")
    
    checks = {
        'evaluation_dataset': False,
        'step2_evaluator': False,
        'milvus_connection': False,
        'results_directory': False
    }
    
    # 1. 평가 데이터셋 존재 확인
    dataset_path = project_root / "src" / "evaluation" / "data" / "insurance_eval_dataset.json"
    if dataset_path.exists():
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            question_count = len(dataset.get('questions', []))
            print(f"[OK] 평가 데이터셋: {question_count}개 질문")
            checks['evaluation_dataset'] = True
        except Exception as e:
            print(f"[FAIL] 데이터셋 로드 오류: {e}")
    else:
        print(f"[FAIL] 평가 데이터셋 파일 없음: {dataset_path}")
    
    # 2. 2단계 평가기 모듈 확인
    evaluator_path = project_root / "src" / "evaluation" / "step2_simple_ragas_evaluator.py"
    if evaluator_path.exists():
        try:
            from src.evaluation.step2_simple_ragas_evaluator import Step2SimpleRAGASEvaluator
            print("[OK] 2단계 평가기 모듈")
            checks['step2_evaluator'] = True
        except ImportError as e:
            print(f"[FAIL] 2단계 평가기 import 오류: {e}")
    else:
        print(f"[FAIL] 2단계 평가기 파일 없음: {evaluator_path}")
    
    # 3. Milvus 연결 확인 (선택적)
    try:
        from src.vectordb.milvus_client import MilvusClient
        from src.utils.config import Config
        
        config = Config()
        client = MilvusClient(config)
        
        if client.is_connected():
            collections = client.list_collections()
            if "insurance_ko_sroberta" in collections:
                print("[OK] Milvus 연결 및 컬렉션 확인")
                checks['milvus_connection'] = True
            else:
                print("[WARN] Milvus 연결됨, 대상 컬렉션 없음 (오프라인 모드 사용)")
        else:
            print("[WARN] Milvus 연결 실패 (오프라인 모드 사용)")
        
        client.close()
        
    except Exception as e:
        print(f"[WARN] Milvus 연결 확인 중 오류: {e} (오프라인 모드 사용)")
    
    # 4. 결과 디렉토리 확인
    results_dir = project_root / "evaluation_results" / "step2_ragas"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"[OK] 결과 저장 디렉토리: {results_dir}")
    checks['results_directory'] = True
    
    return checks

def print_prerequisites_summary(checks: Dict[str, bool]):
    """필수 조건 확인 결과 요약"""
    print(f"\n필수 조건 확인 결과:")
    for check_name, status in checks.items():
        status_str = "[OK]" if status else "[FAIL]"
        check_display = {
            'evaluation_dataset': '평가 데이터셋',
            'step2_evaluator': '2단계 평가기',
            'milvus_connection': 'Milvus 연결',
            'results_directory': '결과 디렉토리'
        }
        print(f"   {status_str} {check_display.get(check_name, check_name)}")
    
    # 필수 조건 만족 여부 확인
    required_checks = ['evaluation_dataset', 'step2_evaluator', 'results_directory']
    can_proceed = all(checks.get(check, False) for check in required_checks)
    
    if can_proceed:
        print(f"\\n[OK] 모든 필수 조건이 만족되었습니다.")
        if not checks.get('milvus_connection', False):
            print(f"[INFO] Milvus 연결되지 않음, 오프라인 모드로 평가 진행됩니다.")
    else:
        print(f"\\n[FAIL] 필수 조건이 만족되지 않아 평가를 진행할 수 없습니다.")
    
    return can_proceed

def run_step2_evaluation() -> Optional[Dict[str, Any]]:
    """
    2단계 RAGAS 평가 실행
    
    Returns:
        Optional[Dict[str, Any]]: 평가 결과
    """
    print_section_header("2단계 RAGAS 평가 실행")
    
    try:
        from src.evaluation.step2_simple_ragas_evaluator import Step2SimpleRAGASEvaluator
        
        print("[INFO] 2단계 평가기 초기화 중...")
        evaluator = Step2SimpleRAGASEvaluator()
        
        print("[INFO] 평가 시작...")
        start_time = time.time()
        
        # 평가 실행
        result = evaluator.run_evaluation()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        if result.get('success', False):
            print(f"\n[OK] 2단계 평가 성공적으로 완료!")
            print(f"   - 처리된 질문: {result.get('processed_count', 0)}/{result.get('total_count', 0)}")
            print(f"   - 소요 시간: {total_time:.2f}초")
            print(f"   - 결과 파일: {result.get('result_file', 'Unknown')}")
            
            # 전체 지표 요약
            overall_metrics = result.get('overall_metrics', {})
            if overall_metrics:
                print(f"   - 전체 평균 점수: {overall_metrics.get('overall_score', 0):.3f}")
            
            return result
        else:
            print(f"\n[FAIL] 2단계 평가 실패")
            print(f"   - 오류: {result.get('error', 'Unknown error')}")
            print(f"   - 처리된 질문: {result.get('processed_count', 0)}")
            return None
            
    except Exception as e:
        print(f"[FAIL] 평가 실행 중 예외 발생: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_with_step1_results():
    """
    1단계 결과와의 비교 및 연계성 확인
    """
    print_section_header("1단계 결과와의 비교")
    
    # 1단계 결과 파일 찾기
    results_dir = project_root / "evaluation_results"
    step1_files = list(results_dir.glob("step1_milvus_retrieval_*.json"))
    
    if step1_files:
        # 가장 최근 파일 선택
        latest_step1 = max(step1_files, key=lambda x: x.stat().st_mtime)
        
        try:
            with open(latest_step1, 'r', encoding='utf-8') as f:
                step1_data = json.load(f)
            
            step1_metrics = step1_data.get('performance_metrics', {})
            
            print(f"[INFO] 1단계 결과 참조: {latest_step1.name}")
            print(f"   - Milvus 검색 성공률: {step1_metrics.get('success_rate', 0):.1%}")
            print(f"   - 평균 검색 시간: {step1_metrics.get('average_search_time', 0):.3f}초")
            print(f"   - 성공한 검색: {step1_metrics.get('successful_searches', 0)}개")
            
            print(f"\n[INFO] 1단계에서 검증된 Milvus 검색 기능을 2단계에서 활용하여")
            print(f"        실제 답변 데이터와 비교한 RAGAS 평가를 수행했습니다.")
            
        except Exception as e:
            print(f"[WARN] 1단계 결과 로드 실패: {e}")
    else:
        print(f"[WARN] 1단계 결과 파일을 찾을 수 없습니다.")
        print(f"        1단계를 먼저 실행하신 후 2단계를 진행하시기 바랍니다.")

def print_next_steps():
    """
    다음 단계 안내
    """
    print_section_header("다음 단계 안내")
    
    print(f"[NEXT] 3단계: 검색 기반 평가 메트릭 개발")
    print(f"   - 사용자 질문에 대한 자동 답변 생성")
    print(f"   - 생성된 답변과 수동 답변 비교")
    print(f"   - 다양한 RAGAS 지표로 성능 평가")
    print(f"   - 최종 RAG 시스템 성능 보고서 작성")
    
    print(f"\n[TIP] 현재까지의 진행 상황:")
    print(f"   ✓ 1단계: Milvus 검색 기능 검증 완료")
    print(f"   ✓ 2단계: RAGAS 평가 구조 검증 완료")
    print(f"   ▷ 3단계: 검색 기반 평가 (다음)")

def main():
    """
    메인 실행 함수
    """
    print_section_header("RAG 평가 2단계: RAGAS 구조 검증")
    print("수동 답변 데이터와 Milvus 검색 결과를 활용한 간소화된 RAGAS 평가를 시작합니다.")
    print("이 단계에서는 PyTorch 의존성 없이 키워드 매칭 기반 평가를 수행합니다.")
    
    # 1. 필수 조건 확인
    checks = validate_prerequisites()
    can_proceed = print_prerequisites_summary(checks)
    
    if not can_proceed:
        print("\n[ABORT] 필수 조건이 만족되지 않아 평가를 중단합니다.")
        return
    
    # 2. 1단계 결과와의 비교
    compare_with_step1_results()
    
    # 3. 2단계 평가 실행
    result = run_step2_evaluation()
    
    if result:
        # 4. 평가 성공 시 다음 단계 안내
        print_next_steps()
        
        print(f"\n" + "="*80)
        print(f"  2단계 평가 완료!")
        print(f"="*80)
        print(f"[SUCCESS] 간소화된 RAGAS 평가 구조가 성공적으로 검증되었습니다.")
        print(f"[FILE] 상세 결과: {result.get('result_file', 'Unknown')}")
        
        overall_score = result.get('overall_metrics', {}).get('overall_score', 0)
        if overall_score >= 0.5:
            print(f"[PERFORMANCE] 평가 점수({overall_score:.3f})가 양호하여 3단계 진행이 권장됩니다.")
        else:
            print(f"[PERFORMANCE] 평가 점수({overall_score:.3f})가 낮습니다. 데이터나 설정을 검토해보세요.")
        
    else:
        # 5. 평가 실패 시 문제 해결 안내
        print(f"\n" + "="*80)
        print(f"  2단계 평가 실패")
        print(f"="*80)
        print(f"[FAIL] 평가가 완료되지 않았습니다.")
        print(f"[SOLUTION] 다음 사항을 확인해보세요:")
        print(f"   1. Milvus 서버가 실행 중인지 확인 (docker-compose up -d)")
        print(f"   2. 평가 데이터셋 파일이 존재하는지 확인")
        print(f"   3. logs/ 디렉토리의 로그 파일 확인")
        print(f"   4. 필요시 오프라인 모드로 재시도")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n[INTERRUPTED] 사용자에 의해 평가가 중단되었습니다.")
    except Exception as e:
        print(f"\n[ERROR] 예기치 못한 오류가 발생했습니다: {e}")
        import traceback
        traceback.print_exc()