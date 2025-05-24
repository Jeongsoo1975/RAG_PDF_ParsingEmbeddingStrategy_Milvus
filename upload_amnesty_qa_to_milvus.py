#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
amnesty_qa 데이터를 Milvus에 업로드하는 스크립트
실제 사용 시나리오 테스트를 위해 테스트 데이터를 Milvus에 저장합니다.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# 프로젝트 루트 경로 추가
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

try:
    from src.utils.config import Config
    from src.vectordb.milvus_client import MilvusClient
    from src.utils.logger import setup_logger
    # sentence-transformers를 직접 사용
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# 로그 설정
logger = setup_logger("amnesty_qa_uploader")

def main():
    """amnesty_qa 데이터를 Milvus에 업로드"""
    try:
        logger.info("=== amnesty_qa 데이터 Milvus 업로드 시작 ===")
        
        # Config 로드
        config = Config()
        logger.info("Config 로드 완료")
        
        # Milvus 클라이언트와 임베딩 모델 초기화
        milvus_client = MilvusClient(config)
        embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  # 384차원 모델로 변경
        logger.info("Milvus 클라이언트 및 임베딩 모델 초기화 완료")
        
        # 1. amnesty_qa 데이터 로드
        data_path = Path("data/amnesty_qa/amnesty_qa_evaluation.json")
        
        if not data_path.exists():
            logger.error(f"데이터 파일이 없습니다: {data_path}")
            return False
        
        with open(data_path, 'r', encoding='utf-8') as f:
            amnesty_data = json.load(f)
        
        logger.info(f"amnesty_qa 데이터 로드 완료: {len(amnesty_data['questions'])}개 질문")
        
        # 2. 컬렉션 생성 (기존 삭제 후 재생성)
        collection_name = "test_amnesty_qa"
        
        try:
            # 기존 컬렉션 삭제
            milvus_client.drop_collection(collection_name)
            logger.info(f"기존 컬렉션 '{collection_name}' 삭제")
        except Exception as e:
            logger.info(f"기존 컬렉션이 없음 (정상): {e}")
        
        # 새 컬렉션 생성
        success = milvus_client.create_collection(
            collection_name=collection_name,
            dimension=384,  # all-MiniLM-L6-v2 모델의 실제 차원
            metric_type="COSINE"
        )
        if success:
            logger.info(f"새 컬렉션 '{collection_name}' 생성 완료")
        else:
            logger.error(f"컬렉션 '{collection_name}' 생성 실패")
            return False
        
        # 3. 각 질문-답변 쌍을 문서로 변환하여 업로드
        documents = []
        
        for i, (question, context_list, ground_truth_list, metadata) in enumerate(zip(
            amnesty_data['questions'],
            amnesty_data['contexts'], 
            amnesty_data['ground_truths'],
            amnesty_data['metadata']
        )):
            # 질문을 문서로 추가
            question_doc = {
                'id': f"question_{i}",
                'content': f"Question: {question}",
                'chunk_type': 'question',
                'source': 'amnesty_qa',
                'title': metadata.get('title', f'Question {i}'),
                'index': i
            }
            documents.append(question_doc)
            
            # 관련 문맥을 문서로 추가
            for j, context in enumerate(context_list):
                context_doc = {
                    'id': f"context_{i}_{j}",
                    'content': context,
                    'chunk_type': 'context',
                    'source': 'amnesty_qa',
                    'title': metadata.get('title', f'Context {i}_{j}'),
                    'question_id': f"question_{i}",
                    'index': i
                }
                documents.append(context_doc)
            
            # 정답을 문서로 추가  
            for j, ground_truth in enumerate(ground_truth_list):
                answer_doc = {
                    'id': f"answer_{i}_{j}",
                    'content': f"Answer: {ground_truth}",
                    'chunk_type': 'answer',
                    'source': 'amnesty_qa',
                    'title': metadata.get('title', f'Answer {i}_{j}'),
                    'question_id': f"question_{i}",
                    'index': i
                }
                documents.append(answer_doc)
        
        logger.info(f"총 {len(documents)}개 문서 생성")
        
        # 4. 문서들을 임베딩하고 Milvus에 업로드
        logger.info("문서 임베딩 및 업로드 시작...")
        
        batch_size = 10
        total_uploaded = 0
        
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            
            try:
                # 임베딩 생성
                texts = [doc['content'] for doc in batch_docs]
                embeddings = embedding_model.encode(texts)
                
                # ID와 메타데이터 준비
                ids = [doc['id'] for doc in batch_docs]
                vectors = [embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding) for embedding in embeddings]
                metadata = []
                
                for doc in batch_docs:
                    meta = {
                        'text': doc['content'],
                        'chunk_type': doc['chunk_type'],
                        'source': doc['source'],
                        'doc_id': doc['id'],
                        'page_num': doc['index'],
                        'chunk_id': doc['id'],
                        'parent_chunk_id': '',
                        'article_title': doc['title'],
                        'item_marker': ''
                    }
                    metadata.append(meta)
                
                # Milvus에 업로드
                inserted_count = milvus_client.insert(
                    collection_name=collection_name,
                    ids=ids,
                    vectors=vectors,
                    metadata=metadata
                )
                
                total_uploaded += inserted_count
                logger.info(f"배치 {i//batch_size + 1} 업로드 완료: {total_uploaded}/{len(documents)}")
                
            except Exception as e:
                logger.error(f"배치 {i//batch_size + 1} 업로드 실패: {str(e)}")
                continue
        
        # 5. 업로드 결과 확인
        logger.info("업로드 완료, 결과 확인 중...")
        
        # 컬렉션 로드 (검색을 위해 필요) - MilvusClient에는 load_collection 메서드가 없을 수 있음
        try:
            # Collection 객체를 직접 사용해서 로드
            from pymilvus import Collection
            collection = Collection(collection_name)
            collection.load()
            logger.info(f"컬렉션 '{collection_name}' 로드 완료")
        except Exception as e:
            logger.warning(f"컬렉션 로드 실패: {e}")
        
        # 간단한 검색 테스트
        test_query = "human rights"
        test_embedding = embedding_model.encode([test_query])[0]
        
        search_results = milvus_client.search(
            collection_name=collection_name,
            query_vector=test_embedding.tolist() if hasattr(test_embedding, 'tolist') else list(test_embedding),
            top_k=5,
            output_fields=['text', 'chunk_type', 'source']
        )
        
        logger.info(f"테스트 검색 '{test_query}': {len(search_results) if search_results else 0}개 결과")
        
        if search_results and len(search_results) > 0:
            for i, result in enumerate(search_results[:3]):
                logger.info(f"  결과 {i+1}: {result.get('text', '')[:100]}... (유사도: {result.get('score', 0):.3f})")
            
            logger.info("✅ amnesty_qa 데이터 업로드 및 검색 테스트 성공!")
            return True
        else:
            logger.warning("⚠️ 업로드는 완료되었으나 검색 결과가 없습니다.")
            return False
        
    except Exception as e:
        logger.error(f"업로드 중 오류 발생: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ amnesty_qa 데이터 Milvus 업로드 성공!")
        sys.exit(0)
    else:
        print("\n❌ amnesty_qa 데이터 Milvus 업로드 실패!")
        sys.exit(1)
