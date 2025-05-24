#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RAGAS 평가를 위한 데이터셋 변환 모듈
- 기존 RAG 평가 데이터셋을 RAGAS 형식으로 변환
- RAGAS 평가를 위한 데이터셋 생성 및 관리
"""

import os
import json
import logging
import yaml
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
import pandas as pd
from tqdm import tqdm

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/ragas_dataset_converter.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ragas_dataset_converter")

@dataclass
class RAGASDatasetItem:
    """RAGAS 데이터셋 아이템 데이터 클래스"""
    question: str
    contexts: List[str]
    ground_truths: List[str]
    answer: Optional[str] = None
    id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class RAGASDatasetConverter:
    """RAGAS 데이터셋 변환 클래스"""
    
    def __init__(self, config_path: str = None):
        """
        RAGAS 데이터셋 변환기 초기화
        
        Args:
            config_path: RAGAS 설정 파일 경로
        """
        self.config = self._load_config(config_path)
        
        # 기본 설정값
        self.output_dir = self.config.get("datasets", {}).get("path", "evaluation_results/ragas_datasets")
        self.threshold = self.config.get("datasets", {}).get("conversion", {}).get("threshold", 0.7)
        self.max_contexts = self.config.get("datasets", {}).get("conversion", {}).get("max_contexts", 5)
        self.text_preprocessing = self.config.get("datasets", {}).get("conversion", {}).get("text_preprocessing", True)
        
        # 출력 디렉토리 생성
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"RAGAS 데이터셋 변환기 초기화 완료 (출력 디렉토리: {self.output_dir})")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        RAGAS 설정 파일 로드
        
        Args:
            config_path: 설정 파일 경로
            
        Returns:
            설정 딕셔너리
        """
        default_config_path = "configs/ragas_config.yaml"
        
        if config_path is None:
            if os.path.exists(default_config_path):
                config_path = default_config_path
            else:
                logger.warning(f"기본 설정 파일({default_config_path})을 찾을 수 없습니다. 기본값을 사용합니다.")
                return {}
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"설정 파일 로드 완료: {config_path}")
            return config
        except Exception as e:
            logger.error(f"설정 파일 로드 실패: {str(e)}")
            return {}
    
    def convert_from_json(self, input_path: str, output_path: Optional[str] = None) -> List[RAGASDatasetItem]:
        """
        기존 평가 데이터셋 JSON 파일을 RAGAS 형식으로 변환
        
        Args:
            input_path: 입력 JSON 파일 경로
            output_path: 출력 JSON 파일 경로 (None인 경우 기본 출력 경로 사용)
            
        Returns:
            변환된 RAGAS 데이터셋 아이템 리스트
        """
        logger.info(f"JSON 데이터셋 변환 시작: {input_path}")
        
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 데이터셋 정보 로깅
            dataset_info = data.get('dataset_info', {})
            logger.info(f"데이터셋 제목: {dataset_info.get('document_title', '알 수 없음')}")
            logger.info(f"질문 수: {dataset_info.get('total_questions', len(data.get('questions', [])))}")
            
            # 데이터셋 아이템 변환
            ragas_items = []
            for q in tqdm(data.get('questions', []), desc="질문 변환 중"):
                # 기본 필드 추출
                question_id = q.get('id', '')
                question_text = q.get('text', '')
                
                # 관련 문서 추출 (RAGAS의 contexts)
                contexts = self._extract_contexts(q)
                
                # Ground truth 추출
                ground_truths = self._extract_ground_truths(q)
                
                # RAGAS 데이터셋 아이템 생성
                ragas_item = RAGASDatasetItem(
                    id=question_id,
                    question=question_text,
                    contexts=contexts[:self.max_contexts],  # 최대 문맥 수 제한
                    ground_truths=ground_truths,
                    metadata={
                        "type": q.get('type', ''),
                        "difficulty": q.get('difficulty', ''),
                        "related_sections": q.get('related_sections', []),
                        "page_numbers": q.get('page_numbers', [])
                    }
                )
                ragas_items.append(ragas_item)
            
            # 데이터셋 저장
            if output_path is None:
                output_filename = os.path.basename(input_path).replace('.json', '_ragas.json')
                output_path = os.path.join(self.output_dir, output_filename)
            
            self.save_dataset(ragas_items, output_path)
            
            logger.info(f"JSON 데이터셋 변환 완료: {len(ragas_items)}개 아이템 생성")
            return ragas_items
            
        except Exception as e:
            logger.error(f"JSON 데이터셋 변환 실패: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def _extract_contexts(self, question_data: Dict[str, Any]) -> List[str]:
        """
        질문 데이터에서 문맥(contexts) 추출
        
        Args:
            question_data: 질문 데이터
            
        Returns:
            문맥 리스트
        """
        contexts = []
        
        # 소스 인용문이 있으면 추가
        source_quote = question_data.get('gold_standard', {}).get('source_quote', '')
        if source_quote:
            contexts.append(source_quote)
        
        # 관련 문서가 있으면 추가 (기존 데이터셋의 구조에 따라 다름)
        related_docs = question_data.get('related_documents', [])
        for doc in related_docs:
            if isinstance(doc, dict) and 'content' in doc:
                contexts.append(doc['content'])
            elif isinstance(doc, str):
                contexts.append(doc)
        
        # 관련 섹션이 있으면 추가 (기존 데이터셋의 구조에 따라 다름)
        related_sections = question_data.get('related_sections', [])
        if isinstance(related_sections, list) and len(related_sections) > 0:
            # 이 부분은 실제 데이터셋 구조에 따라 수정 필요
            pass
        
        # 문맥이 비어있으면 기본 문맥 추가
        if not contexts:
            essential_elements = question_data.get('gold_standard', {}).get('essential_elements', [])
            if essential_elements:
                contexts.append(" ".join(essential_elements))
        
        # 문맥 전처리
        if self.text_preprocessing:
            contexts = [self._preprocess_text(ctx) for ctx in contexts]
        
        return contexts
    
    def _extract_ground_truths(self, question_data: Dict[str, Any]) -> List[str]:
        """
        질문 데이터에서 정답(ground_truths) 추출
        
        Args:
            question_data: 질문 데이터
            
        Returns:
            정답 리스트
        """
        ground_truths = []
        
        # 기본 답변 추가
        answer = question_data.get('gold_standard', {}).get('answer', '')
        if answer:
            ground_truths.append(answer)
        
        # 필수 요소 추가
        essential_elements = question_data.get('gold_standard', {}).get('essential_elements', [])
        if essential_elements and not answer:
            # 필수 요소를 결합하여 정답으로 사용
            ground_truths.append(" ".join(essential_elements))
        
        # 정답이 비어있으면 기본 정답 추가
        if not ground_truths:
            ground_truths.append("답변을 찾을 수 없습니다.")
        
        # 정답 전처리
        if self.text_preprocessing:
            ground_truths = [self._preprocess_text(gt) for gt in ground_truths]
        
        return ground_truths
    
    def _preprocess_text(self, text: str) -> str:
        """
        텍스트 전처리
        
        Args:
            text: 전처리할 텍스트
            
        Returns:
            전처리된 텍스트
        """
        if not text:
            return ""
        
        # 여러 줄의 공백 제거
        text = ' '.join(text.split())
        
        # 특수 문자 처리 (필요한 경우)
        
        return text
    
    def save_dataset(self, items: List[RAGASDatasetItem], output_path: str):
        """
        RAGAS 데이터셋 저장
        
        Args:
            items: RAGAS 데이터셋 아이템 리스트
            output_path: 출력 파일 경로
        """
        try:
            # 출력 디렉토리 생성
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # 데이터셋 포맷을 RAGAS에서 사용하는 형식으로 변환
            dataset_dict = {
                "questions": [item.question for item in items],
                "contexts": [item.contexts for item in items],
                "ground_truths": [item.ground_truths for item in items],
                "ids": [item.id for item in items],
                "metadata": [item.metadata for item in items]
            }
            
            # 답변이 있는 경우 추가
            answers = [item.answer for item in items]
            if any(answer is not None for answer in answers):
                dataset_dict["answers"] = answers
            
            # JSON 파일로 저장
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(dataset_dict, f, ensure_ascii=False, indent=2)
            
            logger.info(f"데이터셋 저장 완료: {output_path}")
            
        except Exception as e:
            logger.error(f"데이터셋 저장 실패: {str(e)}")
    
    def convert_to_huggingface_dataset(self, items: List[RAGASDatasetItem]) -> Dict[str, List[Any]]:
        """
        RAGAS 데이터셋 아이템을 Hugging Face 데이터셋 형식으로 변환
        
        Args:
            items: RAGAS 데이터셋 아이템 리스트
            
        Returns:
            Hugging Face 데이터셋 형식의 딕셔너리
        """
        # 필수 필드 추출
        dataset_dict = {
            "question": [item.question for item in items],
            "contexts": [item.contexts for item in items],
            "ground_truths": [item.ground_truths for item in items]
        }
        
        # 답변이 있는 경우 추가
        answers = [item.answer for item in items]
        if any(answer is not None for answer in answers):
            dataset_dict["answer"] = answers
        
        # ID가 있는 경우 추가
        ids = [item.id for item in items]
        if any(id is not None for id in ids):
            dataset_dict["id"] = ids
        
        return dataset_dict
    
    def convert_to_pandas_dataframe(self, items: List[RAGASDatasetItem]) -> pd.DataFrame:
        """
        RAGAS 데이터셋 아이템을 Pandas DataFrame으로 변환
        
        Args:
            items: RAGAS 데이터셋 아이템 리스트
            
        Returns:
            Pandas DataFrame
        """
        # 항목을 딕셔너리로 변환
        dict_items = []
        for item in items:
            item_dict = asdict(item)
            # 리스트 필드 처리
            item_dict['contexts'] = json.dumps(item_dict['contexts'], ensure_ascii=False)
            item_dict['ground_truths'] = json.dumps(item_dict['ground_truths'], ensure_ascii=False)
            item_dict['metadata'] = json.dumps(item_dict['metadata'], ensure_ascii=False)
            dict_items.append(item_dict)
        
        # DataFrame 생성
        df = pd.DataFrame(dict_items)
        return df
    
    @staticmethod
    def from_huggingface_dataset(dataset_dict: Dict[str, List[Any]]) -> List[RAGASDatasetItem]:
        """
        Hugging Face 데이터셋 형식에서 RAGAS 데이터셋 아이템 생성
        
        Args:
            dataset_dict: Hugging Face 데이터셋 형식의 딕셔너리
            
        Returns:
            RAGAS 데이터셋 아이템 리스트
        """
        items = []
        
        # 필수 필드 확인
        if 'question' not in dataset_dict or 'contexts' not in dataset_dict:
            logger.error("필수 필드(question, contexts)가 없습니다.")
            return items
        
        # 레코드 수 확인
        n_records = len(dataset_dict['question'])
        
        for i in range(n_records):
            # 필수 필드 추출
            question = dataset_dict['question'][i]
            contexts = dataset_dict['contexts'][i]
            
            # 선택적 필드 추출
            id = dataset_dict.get('id', [None] * n_records)[i]
            answer = dataset_dict.get('answer', [None] * n_records)[i]
            ground_truths = dataset_dict.get('ground_truths', [[]] * n_records)[i]
            # ground_truths가 비어 있는 경우 answer를 ground_truth로 사용
            if not ground_truths and answer:
                ground_truths = [answer]
            
            # 메타데이터 추출
            metadata = dataset_dict.get('metadata', [{}] * n_records)[i]
            
            # RAGAS 데이터셋 아이템 생성
            item = RAGASDatasetItem(
                id=id,
                question=question,
                contexts=contexts,
                answer=answer,
                ground_truths=ground_truths,
                metadata=metadata
            )
            items.append(item)
        
        return items

def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='RAGAS 데이터셋 변환 도구')
    parser.add_argument('--input', type=str, required=True, help='입력 데이터셋 파일 경로')
    parser.add_argument('--output', type=str, help='출력 데이터셋 파일 경로')
    parser.add_argument('--config', type=str, help='RAGAS 설정 파일 경로')
    
    args = parser.parse_args()
    
    # 변환기 초기화
    converter = RAGASDatasetConverter(args.config)
    
    # 데이터셋 변환
    if args.input.endswith('.json'):
        items = converter.convert_from_json(args.input, args.output)
    else:
        logger.error(f"지원하지 않는 파일 형식: {args.input}")
        return
    
    # 변환 결과 보고
    print("\n=== RAGAS 데이터셋 변환 결과 ===")
    print(f"전체 아이템 수: {len(items)}")
    if items:
        print(f"첫 번째 아이템:")
        print(f"  질문: {items[0].question}")
        print(f"  문맥 수: {len(items[0].contexts)}")
        print(f"  정답 수: {len(items[0].ground_truths)}")
        print(f"  메타데이터: {items[0].metadata}")

if __name__ == "__main__":
    main()