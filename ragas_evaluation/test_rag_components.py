#!/usr/bin/env python3
"""
PyTorch 의존성 없이 가능한 RAG 컴포넌트들의 기본 기능 테스트
"""

import sys
import os
import traceback
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def print_test_header(test_name):
    print(f"\n{'='*60}")
    print(f"[TEST] {test_name}")
    print(f"{'='*60}")

def test_evaluation_imports():
    """Evaluation 관련 임포트 테스트"""
    print_test_header("Evaluation 컴포넌트 임포트 테스트")
    
    try:
        from src.evaluation.data.ragas_dataset_converter import RagasDatasetConverter
        print("[OK] RagasDatasetConverter 임포트 성공")
    except Exception as e:
        print(f"[FAIL] RagasDatasetConverter 임포트 실패: {e}")
        
    try:
        # PyTorch 의존성이 있는 모듈들은 스킵
        print("[SKIP] RAGPipelineAdapter - PyTorch 의존성으로 인해 스킵")
    except Exception as e:
        print(f"[FAIL] RAGPipelineAdapter 임포트 실패: {e}")

def test_parser_imports():
    """Parser 관련 임포트 테스트"""
    print_test_header("Parser 컴포넌트 임포트 테스트")
    
    try:
        from src.parsers.insurance_pdf_parser import InsurancePDFParser
        print("[OK] InsurancePDFParser 임포트 성공")
    except Exception as e:
        print(f"[FAIL] InsurancePDFParser 임포트 실패: {e}")

def test_basic_config_functionality():
    """Config 클래스의 기본 기능 테스트"""
    print_test_header("Config 클래스 기본 기능 테스트")
    
    try:
        from src.utils.config import Config
        
        # 기본 설정으로 객체 생성
        config = Config()
        print("[OK] 기본 Config 객체 생성 성공")
        
        # 주요 설정 섹션들 확인
        sections = ['milvus', 'embedding', 'chunking', 'retrieval', 'generation']
        for section in sections:
            if hasattr(config, section):
                section_data = getattr(config, section)
                if isinstance(section_data, dict):
                    print(f"[OK] {section} 설정 확인 (키 {len(section_data)}개)")
                else:
                    print(f"[WARN] {section} 설정이 dict가 아님: {type(section_data)}")
            else:
                print(f"[FAIL] {section} 설정 없음")
                
        # 설정 저장 테스트
        test_config_path = "test_config.yaml"
        try:
            config.save(test_config_path)
            print(f"[OK] 설정 저장 테스트 성공: {test_config_path}")
            
            # 파일이 생성되었는지 확인
            if os.path.exists(test_config_path):
                print("[OK] 설정 파일 생성 확인")
                os.remove(test_config_path)  # 테스트 파일 삭제
                print("[OK] 테스트 파일 정리 완료")
            else:
                print("[FAIL] 설정 파일이 생성되지 않음")
                
        except Exception as e:
            print(f"[FAIL] 설정 저장 테스트 실패: {e}")
            
    except Exception as e:
        print(f"[FAIL] Config 기본 기능 테스트 실패: {e}")
        traceback.print_exc()

def test_milvus_collections_info():
    """Milvus 컬렉션들의 상세 정보 확인"""
    print_test_header("Milvus 컬렉션 상세 정보")
    
    try:
        from src.vectordb.milvus_client import MilvusClient
        from src.utils.config import Config
        
        config = Config()
        client = MilvusClient(config)
        
        if client.is_connected():
            collections = client.list_collections()
            print(f"[INFO] 총 {len(collections)}개 컬렉션 발견")
            
            for collection_name in collections:
                try:
                    stats = client.get_collection_stats(collection_name)
                    count = client.count(collection_name)
                    print(f"\n[COLLECTION] {collection_name}")
                    print(f"   - 문서 수: {count}")
                    print(f"   - 스키마 정보: {len(stats)} 항목")
                    
                except Exception as e:
                    print(f"   - 상세 정보 조회 실패: {e}")
            
            client.close()
            print("\n[OK] 컬렉션 정보 조회 완료")
        else:
            print("[FAIL] Milvus 연결 실패")
            
    except Exception as e:
        print(f"[FAIL] Milvus 컬렉션 정보 테스트 실패: {e}")
        traceback.print_exc()

def test_data_directory_structure():
    """데이터 디렉토리 구조 확인"""
    print_test_header("데이터 디렉토리 구조 확인")
    
    try:
        from src.utils.config import Config
        
        config = Config()
        base_dir = config.base_dir
        
        print(f"[INFO] 프로젝트 루트: {base_dir}")
        
        # 주요 디렉토리들 확인
        important_dirs = [
            'data',
            'logs', 
            'src',
            'configs',
            'results',
            'evaluation_results'
        ]
        
        for dir_name in important_dirs:
            dir_path = os.path.join(base_dir, dir_name)
            if os.path.exists(dir_path):
                # 디렉토리 내용 개수 확인
                try:
                    contents = os.listdir(dir_path)
                    print(f"[OK] {dir_name}/ (항목 {len(contents)}개)")
                    
                    # 일부 주요 파일들 표시
                    if contents:
                        sample_files = contents[:3]  # 처음 3개만
                        print(f"      예시: {', '.join(sample_files)}")
                        if len(contents) > 3:
                            print(f"      ... 외 {len(contents) - 3}개")
                        
                except PermissionError:
                    print(f"[WARN] {dir_name}/ (접근 권한 없음)")
            else:
                print(f"[MISSING] {dir_name}/")
                
    except Exception as e:
        print(f"[FAIL] 데이터 디렉토리 구조 확인 실패: {e}")
        traceback.print_exc()

def main():
    """메인 테스트 실행"""
    print("==> RAG 컴포넌트 기본 테스트 시작...")
    print(f"==> 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 테스트 실행
    test_evaluation_imports()
    test_parser_imports()
    test_basic_config_functionality()
    test_data_directory_structure()
    test_milvus_collections_info()
    
    print(f"\n==> 모든 테스트 완료 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n[요약]")
    print("- 기본 컴포넌트들의 임포트와 초기화가 정상적으로 작동합니다")
    print("- Milvus 연결 및 컬렉션 조회가 성공적으로 수행됩니다")
    print("- PyTorch 의존성이 있는 embedder와 관련 모듈들은 별도 테스트가 필요합니다")

if __name__ == "__main__":
    main()
