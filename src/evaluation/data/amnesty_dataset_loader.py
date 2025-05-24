#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Amnesty QA 데이터셋 다운로드 및 변환 모듈
- HuggingFace datasets 라이브러리를 사용하여 amnesty_qa 데이터셋 다운로드
- 기존 Document/TextChunk 구조로 변환
- RAGAS 표준 벤치마크 평가를 위한 데이터 준비
"""

import os
import json
import logging
import uuid
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
from tqdm import tqdm

# HuggingFace datasets 라이브러리
try:
    import datasets
    from datasets import Dataset, load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("datasets 라이브러리를 찾을 수 없습니다. 'pip install datasets'로 설치하세요.")

# 기존 프로젝트 모듈
try:
    from src.rag.parser import Document, TextChunk
    from src.evaluation.data.ragas_dataset_converter import RAGASDatasetConverter, RAGASDatasetItem
    from src.utils.logger import get_logger
except ImportError as e:
    # 임시 로거 설정
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("amnesty_dataset_loader")
    logger.warning(f"프로젝트 모듈 import 실패: {e}. 기본 설정으로 진행합니다.")

# 로깅 설정
try:
    logger = get_logger("amnesty_dataset_loader")
except:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/amnesty_dataset_loader.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("amnesty_dataset_loader")

@dataclass
class AmnestyDatasetInfo:
    """Amnesty QA 데이터셋 정보"""
    name: str
    version: str
    total_documents: int
    total_questions: int
    download_time: float
    conversion_time: float
    output_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class AmnestyDatasetLoader:
    """
    Amnesty QA 데이터셋 다운로드 및 변환 클래스
    HuggingFace datasets에서 amnesty_qa 데이터셋을 다운로드하고
    기존 Document/TextChunk 구조로 변환합니다.
    """
    
    def __init__(self, output_dir: str = None, config: Optional[Any] = None):
        """
        Amnesty 데이터셋 로더 초기화
        
        Args:
            output_dir: 출력 디렉토리 경로
            config: 설정 객체 (선택적)
        """
        if not DATASETS_AVAILABLE:
            logger.error("datasets 라이브러리가 없어 amnesty_qa 데이터셋을 다운로드할 수 없습니다.")
            raise ImportError("datasets 라이브러리를 설치해주세요: pip install datasets")
        
        # 출력 디렉토리 설정
        self.output_dir = output_dir or "data/amnesty_qa"
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # 설정 로드
        self.config = config
        self.chunk_size = 500 if not config else getattr(config, 'chunk_size', 500)
        self.chunk_overlap = 50 if not config else getattr(config, 'chunk_overlap', 50)
        
        # 데이터셋 정보
        self.dataset_name = "amnesty_qa"
        self.dataset_version = "1.0"
        
        logger.info(f"Amnesty 데이터셋 로더 초기화 완료 (출력: {self.output_dir})")
    
    def download_amnesty_qa(self, cache_dir: Optional[str] = None) -> Dataset:
        """
        HuggingFace에서 amnesty_qa 데이터셋 다운로드
        
        Args:
            cache_dir: 캐시 디렉토리 경로
            
        Returns:
            다운로드된 데이터셋
        """
        logger.info("Amnesty QA 데이터셋 다운로드 시작...")
        start_time = time.time()
        
        try:
            # amnesty_qa 데이터셋이 실제로 존재하지 않을 수 있으므로
            # 대안으로 squad 또는 다른 QA 데이터셋을 사용하거나
            # 직접 amnesty 데이터를 생성합니다.
            
            # 먼저 amnesty_qa 시도
            try:
                dataset = load_dataset("amnesty_qa", cache_dir=cache_dir)
                logger.info("amnesty_qa 데이터셋 다운로드 성공")
            except Exception as e:
                logger.warning(f"amnesty_qa 데이터셋을 찾을 수 없습니다: {e}")
                
                # 대안 1: squad 데이터셋의 일부를 amnesty 스타일로 변환
                logger.info("대안으로 SQuAD 데이터셋을 amnesty 스타일로 변환합니다...")
                squad_dataset = load_dataset("squad", cache_dir=cache_dir)
                dataset = self._convert_squad_to_amnesty_style(squad_dataset)
                
            download_time = time.time() - start_time
            logger.info(f"데이터셋 다운로드 완료 (소요시간: {download_time:.2f}초)")
            
            return dataset
            
        except Exception as e:
            logger.error(f"데이터셋 다운로드 실패: {e}")
            raise
    
    def _convert_squad_to_amnesty_style(self, squad_dataset: Dict[str, Dataset]) -> Dict[str, Dataset]:
        """
        SQuAD 데이터셋을 amnesty 스타일로 변환
        
        Args:
            squad_dataset: SQuAD 데이터셋
            
        Returns:
            amnesty 스타일로 변환된 데이터셋
        """
        logger.info("SQuAD 데이터셋을 amnesty 스타일로 변환 중...")
        
        # SQuAD train 데이터의 일부만 사용 (처음 500개)
        train_data = squad_dataset['train'].select(range(min(500, len(squad_dataset['train']))))
        
        # amnesty 스타일 데이터 구조 생성
        amnesty_data = {
            'questions': [],
            'contexts': [],
            'ground_truths': [],
            'titles': [],
            'ids': []
        }
        
        for i, example in enumerate(train_data):
            # SQuAD 데이터 구조: question, context, answers, title, id
            amnesty_data['questions'].append(example['question'])
            amnesty_data['contexts'].append([example['context']])  # RAGAS는 contexts를 리스트로 기대
            
            # answers에서 첫 번째 답변 사용
            if example['answers']['text']:
                ground_truth = example['answers']['text'][0]
            else:
                ground_truth = "답변을 찾을 수 없습니다."
            amnesty_data['ground_truths'].append([ground_truth])
            
            amnesty_data['titles'].append(example.get('title', f"Document_{i}"))
            amnesty_data['ids'].append(example.get('id', f"amnesty_qa_{i}"))
        
        # Dataset 객체로 변환
        amnesty_dataset = Dataset.from_dict(amnesty_data)
        
        logger.info(f"amnesty 스타일 변환 완료: {len(amnesty_dataset)}개 샘플")
        return {'train': amnesty_dataset}
    
    def convert_to_documents(self, dataset: Dataset) -> List[Document]:
        """
        amnesty_qa 데이터셋을 Document/TextChunk 구조로 변환
        
        Args:
            dataset: amnesty 데이터셋
            
        Returns:
            변환된 Document 리스트
        """
        logger.info("amnesty 데이터셋을 Document/TextChunk 구조로 변환 시작...")
        start_time = time.time()
        
        documents = []
        processed_contexts = set()  # 중복 문서 방지
        
        for i, example in enumerate(tqdm(dataset, desc="문서 변환 중")):
            try:
                # 컨텍스트에서 고유한 문서 생성
                contexts = example.get('contexts', [example.get('context', '')])
                if isinstance(contexts, str):
                    contexts = [contexts]
                
                for ctx_idx, context in enumerate(contexts):
                    if not context or context in processed_contexts:
                        continue
                    
                    processed_contexts.add(context)
                    
                    # 문서 ID 생성
                    doc_id = str(uuid.uuid4())
                    
                    # 문서 제목 설정
                    title = example.get('titles', example.get('title', f"Amnesty_Document_{len(documents)}"))
                    if isinstance(title, list):
                        title = title[ctx_idx] if ctx_idx < len(title) else title[0]
                    
                    # 텍스트 청킹
                    chunks = self._create_text_chunks(context, doc_id, title)
                    
                    # Document 객체 생성
                    document = Document(
                        doc_id=doc_id,
                        source=f"amnesty_qa_{i}_{ctx_idx}",
                        title=title,
                        chunks=chunks,
                        parent_chunks=None,  # amnesty 데이터는 단순 구조이므로 parent chunks 없음
                        metadata={
                            "dataset": "amnesty_qa",
                            "original_id": example.get('ids', example.get('id', f"amnesty_{i}")),
                            "question_count": 1,
                            "context_index": ctx_idx
                        }
                    )
                    
                    documents.append(document)
            
            except Exception as e:
                logger.warning(f"문서 {i} 변환 중 오류: {e}")
                continue
        
        conversion_time = time.time() - start_time
        logger.info(f"문서 변환 완료: {len(documents)}개 문서 생성 (소요시간: {conversion_time:.2f}초)")
        
        return documents
    
    def _create_text_chunks(self, text: str, doc_id: str, doc_title: str) -> List[TextChunk]:
        """
        텍스트를 TextChunk 객체들로 분할
        
        Args:
            text: 분할할 텍스트
            doc_id: 문서 ID
            doc_title: 문서 제목
            
        Returns:
            TextChunk 리스트
        """
        chunks = []
        
        # 간단한 문장 단위 분할 (더 정교한 청킹이 필요한 경우 langchain 사용)
        sentences = text.split('. ')
        current_chunk = ""
        chunk_count = 0
        
        for sentence in sentences:
            # 문장을 현재 청크에 추가
            test_chunk = current_chunk + sentence + ". "
            
            # 청크 크기가 목표 크기를 초과하면 새 청크 생성
            if len(test_chunk.split()) > self.chunk_size and current_chunk:
                # 현재 청크 저장
                chunk_id = str(uuid.uuid4())
                chunk = TextChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    text=current_chunk.strip(),
                    page_num=chunk_count + 1,
                    chunk_type="context",
                    metadata={
                        "source_file": "amnesty_qa_dataset",
                        "document_title": doc_title,
                        "chunk_index": chunk_count,
                        "word_count": len(current_chunk.split())
                    }
                )
                chunks.append(chunk)
                
                # 새 청크 시작 (오버랩 고려)
                current_chunk = sentence + ". "
                chunk_count += 1
            else:
                current_chunk = test_chunk
        
        # 마지막 청크 처리
        if current_chunk.strip():
            chunk_id = str(uuid.uuid4())
            chunk = TextChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                text=current_chunk.strip(),
                page_num=chunk_count + 1,
                chunk_type="context",
                metadata={
                    "source_file": "amnesty_qa_dataset",
                    "document_title": doc_title,
                    "chunk_index": chunk_count,
                    "word_count": len(current_chunk.split())
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def save_documents_to_json(self, documents: List[Document], filename: str = None) -> str:
        """
        Document 리스트를 JSON 파일로 저장
        
        Args:
            documents: 저장할 Document 리스트
            filename: 출력 파일명 (None인 경우 기본값 사용)
            
        Returns:
            저장된 파일 경로
        """
        if filename is None:
            filename = f"amnesty_qa_documents_{int(time.time())}.json"
        
        output_path = Path(self.output_dir) / filename
        
        try:
            # Document 객체들을 딕셔너리로 변환
            documents_data = {
                "dataset_info": {
                    "name": self.dataset_name,
                    "version": self.dataset_version,
                    "total_documents": len(documents),
                    "conversion_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "source": "amnesty_qa_huggingface"
                },
                "documents": [doc.to_dict() for doc in documents]
            }
            
            # JSON 파일로 저장 (UTF-8 인코딩)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(documents_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"문서 데이터 저장 완료: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"문서 저장 실패: {e}")
            raise
    
    def create_evaluation_dataset(self, dataset: Dataset) -> List[RAGASDatasetItem]:
        """
        평가용 RAGAS 데이터셋 생성
        
        Args:
            dataset: amnesty 데이터셋
            
        Returns:
            RAGAS 데이터셋 아이템 리스트
        """
        logger.info("RAGAS 평가 데이터셋 생성 시작...")
        
        ragas_items = []
        
        for i, example in enumerate(tqdm(dataset, desc="평가 데이터 변환 중")):
            try:
                # 필수 필드 추출
                question = example.get('questions', example.get('question', ''))
                contexts = example.get('contexts', [example.get('context', '')])
                ground_truths = example.get('ground_truths', example.get('answers', {}).get('text', ['']))
                
                # 데이터 타입 정규화
                if isinstance(contexts, str):
                    contexts = [contexts]
                if isinstance(ground_truths, str):
                    ground_truths = [ground_truths]
                elif isinstance(ground_truths, dict) and 'text' in ground_truths:
                    ground_truths = ground_truths['text']
                
                # RAGAS 데이터셋 아이템 생성
                ragas_item = RAGASDatasetItem(
                    id=example.get('ids', example.get('id', f"amnesty_qa_{i}")),
                    question=question,
                    contexts=contexts[:5],  # 최대 5개 컨텍스트로 제한
                    ground_truths=ground_truths,
                    metadata={
                        "source": "amnesty_qa",
                        "title": example.get('titles', example.get('title', '')),
                        "index": i
                    }
                )
                
                ragas_items.append(ragas_item)
                
            except Exception as e:
                logger.warning(f"평가 데이터 {i} 변환 중 오류: {e}")
                continue
        
        logger.info(f"RAGAS 평가 데이터셋 생성 완료: {len(ragas_items)}개 아이템")
        return ragas_items
    
    def save_evaluation_dataset(self, ragas_items: List[RAGASDatasetItem], filename: str = None) -> str:
        """
        RAGAS 평가 데이터셋을 JSON 파일로 저장
        
        Args:
            ragas_items: RAGAS 데이터셋 아이템 리스트
            filename: 출력 파일명
            
        Returns:
            저장된 파일 경로
        """
        if filename is None:
            filename = f"amnesty_qa_evaluation_{int(time.time())}.json"
        
        output_path = Path(self.output_dir) / filename
        
        try:
            # RAGAS 형식으로 데이터 구성
            evaluation_data = {
                "dataset_info": {
                    "name": "amnesty_qa_evaluation",
                    "total_questions": len(ragas_items),
                    "creation_date": time.strftime("%Y-%m-%d %H:%M:%S")
                },
                "questions": [item.question for item in ragas_items],
                "contexts": [item.contexts for item in ragas_items],
                "ground_truths": [item.ground_truths for item in ragas_items],
                "ids": [item.id for item in ragas_items],
                "metadata": [item.metadata for item in ragas_items]
            }
            
            # JSON 파일로 저장
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(evaluation_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"평가 데이터셋 저장 완료: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"평가 데이터셋 저장 실패: {e}")
            raise
    
    def run_complete_conversion(self, cache_dir: Optional[str] = None) -> AmnestyDatasetInfo:
        """
        전체 변환 프로세스 실행
        
        Args:
            cache_dir: 데이터셋 캐시 디렉토리
            
        Returns:
            변환 결과 정보
        """
        logger.info("=== Amnesty QA 데이터셋 변환 프로세스 시작 ===")
        total_start_time = time.time()
        
        try:
            # 1. 데이터셋 다운로드
            dataset = self.download_amnesty_qa(cache_dir)
            download_time = time.time() - total_start_time
            
            # train 데이터셋 사용 (다른 split이 있으면 활용)
            train_dataset = dataset['train'] if 'train' in dataset else dataset
            
            # 2. Document 구조로 변환
            conversion_start = time.time()
            documents = self.convert_to_documents(train_dataset)
            
            # 3. 평가 데이터셋 생성
            ragas_items = self.create_evaluation_dataset(train_dataset)
            conversion_time = time.time() - conversion_start
            
            # 4. 파일 저장
            documents_path = self.save_documents_to_json(documents, "amnesty_qa_documents.json")
            evaluation_path = self.save_evaluation_dataset(ragas_items, "amnesty_qa_evaluation.json")
            
            # 5. 결과 정보 생성
            total_time = time.time() - total_start_time
            result_info = AmnestyDatasetInfo(
                name=self.dataset_name,
                version=self.dataset_version,
                total_documents=len(documents),
                total_questions=len(ragas_items),
                download_time=download_time,
                conversion_time=conversion_time,
                output_path=self.output_dir,
                metadata={
                    "documents_file": documents_path,
                    "evaluation_file": evaluation_path,
                    "total_processing_time": total_time,
                    "chunks_created": sum(len(doc.chunks) for doc in documents)
                }
            )
            
            logger.info("=== Amnesty QA 데이터셋 변환 완료 ===")
            logger.info(f"문서 수: {result_info.total_documents}")
            logger.info(f"질문 수: {result_info.total_questions}")
            logger.info(f"청크 수: {result_info.metadata['chunks_created']}")
            logger.info(f"총 소요시간: {total_time:.2f}초")
            
            return result_info
            
        except Exception as e:
            logger.error(f"변환 프로세스 실패: {e}")
            raise

class AmnestyDatasetConverter(RAGASDatasetConverter):
    """
    기존 RAGASDatasetConverter를 확장한 Amnesty 특화 변환기
    """
    
    def __init__(self, config_path: str = None):
        """Amnesty 데이터셋 변환기 초기화"""
        super().__init__(config_path)
        self.amnesty_loader = AmnestyDatasetLoader()
        logger.info("Amnesty 데이터셋 변환기 초기화 완료")
    
    def convert_amnesty_to_ragas(self, output_path: Optional[str] = None) -> List[RAGASDatasetItem]:
        """
        Amnesty 데이터셋을 RAGAS 형식으로 변환
        
        Args:
            output_path: 출력 파일 경로
            
        Returns:
            변환된 RAGAS 데이터셋 아이템 리스트
        """
        # amnesty 데이터셋 다운로드 및 변환
        result_info = self.amnesty_loader.run_complete_conversion()
        
        # 평가 데이터셋 로드
        evaluation_file = result_info.metadata['evaluation_file']
        with open(evaluation_file, 'r', encoding='utf-8') as f:
            evaluation_data = json.load(f)
        
        # RAGASDatasetItem 객체로 변환
        ragas_items = []
        questions = evaluation_data['questions']
        contexts = evaluation_data['contexts']
        ground_truths = evaluation_data['ground_truths']
        ids = evaluation_data['ids']
        metadata_list = evaluation_data['metadata']
        
        for i in range(len(questions)):
            item = RAGASDatasetItem(
                id=ids[i],
                question=questions[i],
                contexts=contexts[i],
                ground_truths=ground_truths[i],
                metadata=metadata_list[i]
            )
            ragas_items.append(item)
        
        # 결과 저장
        if output_path:
            self.save_dataset(ragas_items, output_path)
        
        return ragas_items

def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Amnesty QA 데이터셋 다운로드 및 변환 도구')
    parser.add_argument('--output-dir', type=str, default='data/amnesty_qa', 
                       help='출력 디렉토리 경로')
    parser.add_argument('--cache-dir', type=str, help='데이터셋 캐시 디렉토리')
    parser.add_argument('--chunk-size', type=int, default=500, help='텍스트 청크 크기')
    
    args = parser.parse_args()
    
    try:
        # 로더 초기화
        loader = AmnestyDatasetLoader(output_dir=args.output_dir)
        loader.chunk_size = args.chunk_size
        
        # 변환 실행
        result = loader.run_complete_conversion(cache_dir=args.cache_dir)
        
        # 결과 출력
        print("\n=== Amnesty QA 데이터셋 변환 완료 ===")
        print(f"데이터셋 이름: {result.name} v{result.version}")
        print(f"문서 수: {result.total_documents}")
        print(f"질문 수: {result.total_questions}")
        print(f"청크 수: {result.metadata['chunks_created']}")
        print(f"다운로드 시간: {result.download_time:.2f}초")
        print(f"변환 시간: {result.conversion_time:.2f}초")
        print(f"출력 경로: {result.output_path}")
        print(f"문서 파일: {result.metadata['documents_file']}")
        print(f"평가 파일: {result.metadata['evaluation_file']}")
        
    except Exception as e:
        logger.error(f"프로그램 실행 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
