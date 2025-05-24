#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
수동 답변 데이터 로더 및 처리기
- 평가용 데이터셋 로드 및 처리
- RAGAS 평가에 필요한 형태로 전처리
- gold_standard 데이터 활용한 평가용 데이터셋 준비
"""

import os
import json
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/manual_data_loader.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("manual_data_loader")

@dataclass
class ManualEvaluationItem:
    """수동 평가 데이터 아이템"""
    question_id: str
    question: str
    ground_truth_answer: str
    essential_keywords: List[str]
    reference_context: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DatasetInfo:
    """데이터셋 정보"""
    document_title: str
    total_questions: int
    source_path: str
    load_time: str

class ManualAnswerDataLoader:
    """수동 답변 데이터 로더 클래스"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        수동 답변 데이터 로더 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        self.config = config or {}
        
        # 데이터 품질 설정
        self.min_answer_length = self.config.get("min_answer_length", 10)
        self.min_keywords_count = self.config.get("min_keywords_count", 1)
        self.text_preprocessing = self.config.get("text_preprocessing", True)
        
        # 데이터셋 경로 설정 (TODO: 실제 데이터셋 경로 설정 필요)
        self.default_dataset_path = "src/evaluation/data/eval_dataset.json"
        
        logger.info("수동 답변 데이터 로더 초기화 완료")
    
    def load_dataset(self, dataset_path: str = None) -> Tuple[List[ManualEvaluationItem], DatasetInfo]:
        """
        평가 데이터셋 로드
        
        Args:
            dataset_path: 데이터셋 파일 경로 (None인 경우 기본 경로 사용)
            
        Returns:
            평가 아이템 리스트, 데이터셋 정보
        """
        if dataset_path is None:
            dataset_path = self.default_dataset_path
        
        logger.info(f"평가 데이터셋 로드 시작: {dataset_path}")
        
        try:
            # 절대 경로로 변환
            if not os.path.isabs(dataset_path):
                dataset_path = os.path.abspath(dataset_path)
            
            # 파일 존재 확인
            if not os.path.exists(dataset_path):
                logger.error(f"데이터셋 파일을 찾을 수 없습니다: {dataset_path}")
                return [], DatasetInfo("", 0, dataset_path, "")
            
            # JSON 파일 로드
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 데이터셋 정보 추출
            dataset_info_raw = data.get('dataset_info', {})
            dataset_info = DatasetInfo(
                document_title=dataset_info_raw.get('document_title', '알 수 없음'),
                total_questions=dataset_info_raw.get('total_questions', 0),
                source_path=dataset_path,
                load_time=self._get_current_time()
            )
            
            logger.info(f"데이터셋 정보: {dataset_info.document_title}, {dataset_info.total_questions}개 질문")
            
            # 질문 데이터 추출
            questions = data.get('questions', [])
            if not questions:
                logger.warning("질문 데이터가 비어있습니다.")
                return [], dataset_info
            
            # 평가 아이템 추출
            evaluation_items = self.extract_evaluation_items(questions)
            
            logger.info(f"평가 아이템 추출 완료: {len(evaluation_items)}개 아이템")
            
            return evaluation_items, dataset_info
            
        except Exception as e:
            logger.error(f"데이터셋 로드 실패: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return [], DatasetInfo("", 0, dataset_path, "")
    
    def extract_evaluation_items(self, questions: List[Dict[str, Any]]) -> List[ManualEvaluationItem]:
        """
        각 질문별로 평가 아이템 추출
        
        Args:
            questions: 질문 데이터 리스트
            
        Returns:
            평가 아이템 리스트
        """
        evaluation_items = []
        
        for i, question_data in enumerate(questions):
            try:
                # 기본 필드 추출
                question_id = question_data.get('id', f'Q{i+1:03d}')
                question_text = question_data.get('text', '')
                
                # gold_standard 데이터 추출
                gold_standard = question_data.get('gold_standard', {})
                
                if not gold_standard:
                    logger.warning(f"질문 {question_id}: gold_standard 데이터가 없습니다.")
                    continue
                
                # 답변, 필수 요소, 참조 컨텍스트 추출
                ground_truth_answer = gold_standard.get('answer', '')
                essential_elements = gold_standard.get('essential_elements', [])
                source_quote = gold_standard.get('source_quote', '')
                
                # 데이터 품질 검증
                if not self._validate_evaluation_item(question_id, ground_truth_answer, essential_elements):
                    continue
                
                # 텍스트 전처리
                if self.text_preprocessing:
                    question_text = self.preprocess_text(question_text)
                    ground_truth_answer = self.preprocess_text(ground_truth_answer)
                    source_quote = self.preprocess_text(source_quote)
                    essential_elements = [self.preprocess_text(elem) for elem in essential_elements]
                
                # 메타데이터 구성
                metadata = {
                    "type": question_data.get('type', ''),
                    "difficulty": question_data.get('difficulty', ''),
                    "related_sections": question_data.get('related_sections', []),
                    "page_numbers": question_data.get('page_numbers', []),
                    "original_index": i
                }
                
                # 평가 아이템 생성
                evaluation_item = ManualEvaluationItem(
                    question_id=question_id,
                    question=question_text,
                    ground_truth_answer=ground_truth_answer,
                    essential_keywords=essential_elements,
                    reference_context=source_quote,
                    metadata=metadata
                )
                
                evaluation_items.append(evaluation_item)
                
                logger.debug(f"질문 {question_id} 추출 완료")
                
            except Exception as e:
                logger.error(f"질문 {i+1} 처리 중 오류: {str(e)}")
                continue
        
        logger.info(f"평가 아이템 추출 완료: {len(evaluation_items)}/{len(questions)} 성공")
        
        return evaluation_items
    
    def _validate_evaluation_item(self, question_id: str, answer: str, keywords: List[str]) -> bool:
        """
        평가 아이템 데이터 품질 검증
        
        Args:
            question_id: 질문 ID
            answer: 정답
            keywords: 필수 키워드
            
        Returns:
            검증 성공 여부
        """
        # 답변 길이 검증
        if len(answer) < self.min_answer_length:
            logger.warning(f"질문 {question_id}: 답변이 너무 짧습니다 (길이: {len(answer)})")
            return False
        
        # 키워드 수 검증
        if len(keywords) < self.min_keywords_count:
            logger.warning(f"질문 {question_id}: 필수 키워드가 부족합니다 (개수: {len(keywords)})")
            return False
        
        # 키워드가 모두 비어있지 않은지 확인
        valid_keywords = [kw for kw in keywords if kw and kw.strip()]
        if len(valid_keywords) < self.min_keywords_count:
            logger.warning(f"질문 {question_id}: 유효한 키워드가 부족합니다")
            return False
        
        return True
    
    def preprocess_text(self, text: str) -> str:
        """
        텍스트 정규화 및 전처리
        
        Args:
            text: 입력 텍스트
            
        Returns:
            전처리된 텍스트
        """
        if not text:
            return ""
        
        # 여러 공백을 하나로 변환
        text = re.sub(r'\s+', ' ', text)
        
        # 앞뒤 공백 제거
        text = text.strip()
        
        # 특수문자 정리 (한국어 문서 고려)
        # 불필요한 특수문자 제거하되 문장부호는 보존
        text = re.sub(r'[^\w\s가-힣.,!?()""''\"\':-]', ' ', text)
        
        # 다시 여러 공백을 하나로 변환
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def create_ragas_format(self, evaluation_items: List[ManualEvaluationItem]) -> Dict[str, List[Any]]:
        """
        RAGAS 평가 형식으로 변환
        
        Args:
            evaluation_items: 평가 아이템 리스트
            
        Returns:
            RAGAS 형식 데이터셋
        """
        logger.info("RAGAS 형식으로 변환 시작")
        
        ragas_dataset = {
            "questions": [],
            "ground_truth_answers": [],
            "essential_keywords": [],
            "reference_contexts": [],
            "question_ids": [],
            "metadata": []
        }
        
        for item in evaluation_items:
            ragas_dataset["questions"].append(item.question)
            ragas_dataset["ground_truth_answers"].append(item.ground_truth_answer)
            ragas_dataset["essential_keywords"].append(item.essential_keywords)
            ragas_dataset["reference_contexts"].append([item.reference_context])  # 리스트로 감싸기
            ragas_dataset["question_ids"].append(item.question_id)
            ragas_dataset["metadata"].append(item.metadata)
        
        logger.info(f"RAGAS 형식 변환 완료: {len(evaluation_items)}개 아이템")
        
        return ragas_dataset
    
    def save_processed_dataset(
        self, 
        evaluation_items: List[ManualEvaluationItem], 
        dataset_info: DatasetInfo,
        output_path: str
    ):
        """
        처리된 데이터셋 저장
        
        Args:
            evaluation_items: 평가 아이템 리스트
            dataset_info: 데이터셋 정보
            output_path: 출력 파일 경로
        """
        try:
            # 출력 디렉토리 생성
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # RAGAS 형식으로 변환
            ragas_dataset = self.create_ragas_format(evaluation_items)
            
            # 데이터셋 정보 추가
            output_data = {
                "dataset_info": {
                    "document_title": dataset_info.document_title,
                    "total_questions": dataset_info.total_questions,
                    "processed_questions": len(evaluation_items),
                    "source_path": dataset_info.source_path,
                    "load_time": dataset_info.load_time,
                    "processing_config": self.config
                },
                "ragas_dataset": ragas_dataset
            }
            
            # JSON 파일로 저장
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"처리된 데이터셋 저장 완료: {output_path}")
            
        except Exception as e:
            logger.error(f"데이터셋 저장 실패: {str(e)}")
    
    def get_evaluation_summary(self, evaluation_items: List[ManualEvaluationItem]) -> Dict[str, Any]:
        """
        평가 데이터 요약 정보 생성
        
        Args:
            evaluation_items: 평가 아이템 리스트
            
        Returns:
            요약 정보 딕셔너리
        """
        if not evaluation_items:
            return {}
        
        # 기본 통계
        total_items = len(evaluation_items)
        
        # 질문 유형별 분포
        type_distribution = {}
        difficulty_distribution = {}
        
        for item in evaluation_items:
            q_type = item.metadata.get('type', 'unknown')
            difficulty = item.metadata.get('difficulty', 'unknown')
            
            type_distribution[q_type] = type_distribution.get(q_type, 0) + 1
            difficulty_distribution[difficulty] = difficulty_distribution.get(difficulty, 0) + 1
        
        # 답변 길이 통계
        answer_lengths = [len(item.ground_truth_answer) for item in evaluation_items]
        avg_answer_length = sum(answer_lengths) / len(answer_lengths)
        
        # 키워드 수 통계
        keyword_counts = [len(item.essential_keywords) for item in evaluation_items]
        avg_keyword_count = sum(keyword_counts) / len(keyword_counts)
        
        summary = {
            "total_items": total_items,
            "type_distribution": type_distribution,
            "difficulty_distribution": difficulty_distribution,
            "answer_statistics": {
                "average_length": round(avg_answer_length, 1),
                "min_length": min(answer_lengths),
                "max_length": max(answer_lengths)
            },
            "keyword_statistics": {
                "average_count": round(avg_keyword_count, 1),
                "min_count": min(keyword_counts),
                "max_count": max(keyword_counts)
            }
        }
        
        return summary
    
    def _get_current_time(self) -> str:
        """현재 시간 문자열 반환"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 테스트 함수
def test_manual_data_loader():
    """수동 데이터 로더 테스트"""
    print("=== 수동 답변 데이터 로더 테스트 ===")
    
    # 데이터 로더 초기화
    loader = ManualAnswerDataLoader()
    
    # 데이터셋 로드
    evaluation_items, dataset_info = loader.load_dataset()
    
    if not evaluation_items:
        print("데이터셋 로드 실패")
        return
    
    print(f"데이터셋 제목: {dataset_info.document_title}")
    print(f"총 질문 수: {dataset_info.total_questions}")
    print(f"처리된 질문 수: {len(evaluation_items)}")
    
    # 첫 번째 아이템 확인
    if evaluation_items:
        first_item = evaluation_items[0]
        print(f"\n첫 번째 질문 ({first_item.question_id}):")
        print(f"질문: {first_item.question[:100]}...")
        print(f"정답: {first_item.ground_truth_answer[:100]}...")
        print(f"필수 키워드: {first_item.essential_keywords[:5]}")
        print(f"참조 컨텍스트: {first_item.reference_context[:100]}...")
    
    # 요약 정보
    summary = loader.get_evaluation_summary(evaluation_items)
    print(f"\n요약 정보:")
    print(f"- 평균 답변 길이: {summary['answer_statistics']['average_length']}자")
    print(f"- 평균 키워드 수: {summary['keyword_statistics']['average_count']}개")
    print(f"- 질문 유형 분포: {summary['type_distribution']}")

if __name__ == "__main__":
    test_manual_data_loader()
