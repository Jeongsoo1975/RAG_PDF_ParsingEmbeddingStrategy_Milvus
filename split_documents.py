#!/usr/bin/env python3
"""
amnesty_qa_documents.json을 개별 문서 파일로 분할하여 embedder가 처리할 수 있도록 함
"""

import json
import os
from pathlib import Path

def split_documents():
    # 입력 파일 경로
    input_file = Path("data/amnesty_qa/amnesty_qa_documents.json")
    output_dir = Path("data/amnesty_qa/chunks")
    
    # 출력 디렉토리 생성
    output_dir.mkdir(exist_ok=True)
    
    # JSON 파일 로드
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"총 {len(data['documents'])}개 문서 처리 중...")
    
    # 각 문서를 개별 파일로 저장
    for i, document in enumerate(data['documents']):
        # 파일명 생성 (embedder가 찾는 _chunks.json 형식)
        filename = f"amnesty_qa_document_{i:02d}_chunks.json"
        output_path = output_dir / filename
        
        # 문서를 embedder가 예상하는 형식으로 변환
        doc_for_embedder = {
            "doc_id": document["doc_id"],
            "source": document["source"],
            "title": document["title"],
            "chunks": document["chunks"],
            "parent_chunks": document.get("parent_chunks"),
            "metadata": document.get("metadata", {})
        }
        
        # 파일 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(doc_for_embedder, f, ensure_ascii=False, indent=2)
        
        print(f"문서 {i+1}/10 저장: {filename}")
    
    print(f"\n모든 문서가 {output_dir}에 저장되었습니다.")
    print(f"이제 다음 명령어로 임베딩을 실행하세요:")
    print(f"python src/rag/embedder.py --input_path {output_dir} --store_in_db")

if __name__ == "__main__":
    split_documents()
