#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
간소화된 RAGAS 평가 지표 구현
- PyTorch 의존성 없는 키워드 매칭 기반 RAGAS 평가 지표
- Context Relevancy, Faithfulness, Answer Relevancy 구현
- 한국어 텍스트 처리 특화
"""

import re
import logging
from typing import List, Dict, Any, Set, Optional, Tuple
from dataclasses import dataclass, field
import json

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/simple_ragas_metrics.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("simple_ragas_metrics")

@dataclass
class SimpleRAGASResult:
    """간소화된 RAGAS 평가 결과 데이터 클래스"""
    question_id: str
    context_relevancy: float
    context_precision: float
    context_recall: float
    faithfulness: float  
    answer_relevancy: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class SimpleRAGASMetrics:
    """간소화된 RAGAS 평가 지표 클래스"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        간소화된 RAGAS 평가 지표 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        self.config = config or {}
        
        # 키워드 추출 설정
        self.min_keyword_length = self.config.get("min_keyword_length", 2)
        self.max_keywords = self.config.get("max_keywords", 20)
        self.use_stemming = self.config.get("use_stemming", False)
        
        # 한국어 불용어 리스트
        self.korean_stopwords = {
            '이', '그', '저', '것', '들', '의', '가', '를', '에', '는', '은', '과', '와', '로', '으로',
            '이다', '있다', '하다', '되다', '않다', '없다', '같다', '다른', '많다', '적다', '크다', '작다',
            '좋다', '나쁘다', '새다', '옛다', '첫다', '끝다', '빠르다', '느리다', '높다', '낮다',
            '그래서', '그러나', '하지만', '그리고', '또한', '또는', '그런데', '따라서', '즉', '예를',
            '위해', '통해', '대해', '관해', '때문', '까지', '부터', '동안', '사이', '중에', '안에',
            '밖에', '위에', '아래', '앞에', '뒤에', '옆에', '근처', '주변', '내부', '외부',
            '이런', '그런', '저런', '어떤', '모든', '각각', '여러', '다양', '특별', '일반',
            '보험', '계약', '회사', '계약자', '피보험자', '보험금', '보험료', '약관', '조항'
        }
        
        logger.info("간소화된 RAGAS 평가 지표 초기화 완료")
    
    def _extract_keywords(self, text: str) -> Set[str]:
        """
        텍스트에서 중요 키워드 추출
        
        Args:
            text: 입력 텍스트
            
        Returns:
            키워드 집합
        """
        if not text:
            return set()
        
        # 텍스트 정규화
        text = self._normalize_text(text)
        
        # 단어 분할 (한국어 고려)
        words = self._tokenize_korean(text)
        
        # 키워드 필터링
        keywords = set()
        for word in words:
            # 길이 필터링
            if len(word) < self.min_keyword_length:
                continue
            
            # 불용어 제거
            if word in self.korean_stopwords:
                continue
            
            # 숫자만 있는 단어 제거
            if word.isdigit():
                continue
            
            # 특수문자만 있는 단어 제거
            if re.match(r'^[^\w가-힣]+$', word):
                continue
            
            keywords.add(word)
        
        # 키워드 수 제한
        if len(keywords) > self.max_keywords:
            # 길이가 긴 키워드 우선 선택
            keywords = set(sorted(keywords, key=len, reverse=True)[:self.max_keywords])
        
        return keywords
    
    def _normalize_text(self, text: str) -> str:
        """
        텍스트 정규화
        
        Args:
            text: 입력 텍스트
            
        Returns:
            정규화된 텍스트
        """
        if not text:
            return ""
        
        # 여러 공백을 하나로 변환
        text = re.sub(r'\s+', ' ', text)
        
        # 특수문자 정리 (한국어 문자는 보존)
        text = re.sub(r'[^\w\s가-힣]', ' ', text)
        
        # 소문자 변환
        text = text.lower()
        
        # 앞뒤 공백 제거
        text = text.strip()
        
        return text
    
    def _tokenize_korean(self, text: str) -> List[str]:
        """
        한국어 텍스트 토큰화
        
        Args:
            text: 입력 텍스트
            
        Returns:
            토큰 리스트
        """
        # 간단한 공백 기반 토큰화 (향후 개선 가능)
        words = text.split()
        
        # 추가적인 한국어 처리
        processed_words = []
        for word in words:
            # 길이가 너무 긴 단어 분할
            if len(word) > 10:
                # 한글과 영문이 섞인 경우 분할
                parts = re.findall(r'[가-힣]+|[a-zA-Z]+|\d+', word)
                processed_words.extend(parts)
            else:
                processed_words.append(word)
        
        return processed_words
    
    def calculate_context_relevancy(
        self, 
        question: str, 
        contexts: List[str]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        질문과 컨텍스트 간 관련성 계산
        
        Args:
            question: 질문 텍스트
            contexts: 컨텍스트 리스트
            
        Returns:
            관련성 점수 (0.0-1.0), 상세 정보
        """
        if not question or not contexts:
            return 0.0, {"error": "Empty question or contexts"}
        
        # 질문에서 키워드 추출
        question_keywords = self._extract_keywords(question)
        
        if not question_keywords:
            return 0.0, {"error": "No keywords in question"}
        
        # 각 컨텍스트와의 관련성 계산
        context_scores = []
        context_details = []
        
        for i, context in enumerate(contexts):
            context_keywords = self._extract_keywords(context)
            
            if not context_keywords:
                context_scores.append(0.0)
                context_details.append({
                    "context_index": i,
                    "overlap_keywords": [],
                    "score": 0.0
                })
                continue
            
            # 키워드 겹침 계산
            overlap_keywords = question_keywords.intersection(context_keywords)
            overlap_ratio = len(overlap_keywords) / len(question_keywords)
            
            context_scores.append(overlap_ratio)
            context_details.append({
                "context_index": i,
                "overlap_keywords": list(overlap_keywords),
                "question_keywords": list(question_keywords),
                "context_keywords": list(context_keywords),
                "score": overlap_ratio
            })
        
        # 최고 점수를 최종 점수로 사용
        final_score = max(context_scores) if context_scores else 0.0
        
        metadata = {
            "question_keywords": list(question_keywords),
            "context_details": context_details,
            "best_context_index": context_scores.index(final_score) if context_scores else -1
        }
        
        return min(final_score, 1.0), metadata
    
    def calculate_faithfulness(
        self, 
        contexts: List[str], 
        answer: str
    ) -> Tuple[float, Dict[str, Any]]:
        """
        컨텍스트와 답변 간 충실성 계산
        
        Args:
            contexts: 컨텍스트 리스트
            answer: 답변 텍스트
            
        Returns:
            충실성 점수 (0.0-1.0), 상세 정보
        """
        if not contexts or not answer:
            return 0.0, {"error": "Empty contexts or answer"}
        
        # 답변에서 키워드 추출
        answer_keywords = self._extract_keywords(answer)
        
        if not answer_keywords:
            return 0.0, {"error": "No keywords in answer"}
        
        # 모든 컨텍스트에서 키워드 추출
        all_context_keywords = set()
        context_keyword_details = []
        
        for i, context in enumerate(contexts):
            context_keywords = self._extract_keywords(context)
            all_context_keywords.update(context_keywords)
            context_keyword_details.append({
                "context_index": i,
                "keywords": list(context_keywords)
            })
        
        if not all_context_keywords:
            return 0.0, {"error": "No keywords in contexts"}
        
        # 답변 키워드가 컨텍스트에서 지원되는 비율 계산
        supported_keywords = answer_keywords.intersection(all_context_keywords)
        faithfulness_score = len(supported_keywords) / len(answer_keywords)
        
        metadata = {
            "answer_keywords": list(answer_keywords),
            "context_keywords": list(all_context_keywords),
            "supported_keywords": list(supported_keywords),
            "unsupported_keywords": list(answer_keywords - supported_keywords),
            "context_details": context_keyword_details
        }
        
        return min(faithfulness_score, 1.0), metadata
    
    def calculate_answer_relevancy(
        self, 
        question: str, 
        answer: str
    ) -> Tuple[float, Dict[str, Any]]:
        """
        질문과 답변 간 관련성 계산
        
        Args:
            question: 질문 텍스트
            answer: 답변 텍스트
            
        Returns:
            관련성 점수 (0.0-1.0), 상세 정보
        """
        if not question or not answer:
            return 0.0, {"error": "Empty question or answer"}
        
        # 질문과 답변에서 키워드 추출
        question_keywords = self._extract_keywords(question)
        answer_keywords = self._extract_keywords(answer)
        
        if not question_keywords or not answer_keywords:
            return 0.0, {"error": "No keywords in question or answer"}
        
        # 키워드 겹침 계산
        overlap_keywords = question_keywords.intersection(answer_keywords)
        
        # 양방향 관련성 계산
        question_coverage = len(overlap_keywords) / len(question_keywords)
        answer_coverage = len(overlap_keywords) / len(answer_keywords)
        
        # 조화 평균으로 최종 점수 계산
        if question_coverage + answer_coverage > 0:
            relevancy_score = 2 * (question_coverage * answer_coverage) / (question_coverage + answer_coverage)
        else:
            relevancy_score = 0.0
        
        metadata = {
            "question_keywords": list(question_keywords),
            "answer_keywords": list(answer_keywords),
            "overlap_keywords": list(overlap_keywords),
            "question_coverage": question_coverage,
            "answer_coverage": answer_coverage
        }
        
        return min(relevancy_score, 1.0), metadata
    
    def calculate_context_precision(
        self, 
        question: str, 
        contexts: List[str], 
        answer: str
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Context Precision 계산 - 검색된 컨텍스트 중에서 실제로 답변 생성에 유용한 비율
        
        Args:
            question: 질문 텍스트
            contexts: 컨텍스트 리스트 (순서대로 정렬됨)
            answer: 생성된 답변
            
        Returns:
            precision 점수 (0.0-1.0), 상세 정보
        """
        if not question or not contexts or not answer:
            return 0.0, {"error": "Empty inputs"}
        
        # 답변에서 키워드 추출
        answer_keywords = self._extract_keywords(answer)
        if not answer_keywords:
            return 0.0, {"error": "No keywords in answer"}
        
        # 질문에서 키워드 추출 (답변의 관련성 판단을 위해)
        question_keywords = self._extract_keywords(question)
        
        relevant_contexts = 0
        context_details = []
        
        for i, context in enumerate(contexts):
            context_keywords = self._extract_keywords(context)
            
            # 컨텍스트가 답변 생성에 유용한지 판단
            # 1. 답변 키워드와의 겹침
            answer_overlap = context_keywords.intersection(answer_keywords)
            answer_overlap_ratio = len(answer_overlap) / len(answer_keywords) if answer_keywords else 0
            
            # 2. 질문 키워드와의 겹침
            question_overlap = context_keywords.intersection(question_keywords)
            question_overlap_ratio = len(question_overlap) / len(question_keywords) if question_keywords else 0
            
            # 컨텍스트가 유용한지 판단 (임계값 기반)
            # 답변과 10% 이상 겹치거나 질문과 20% 이상 겹치면 유용한 것으로 판단
            is_relevant = answer_overlap_ratio >= 0.1 or question_overlap_ratio >= 0.2
            
            if is_relevant:
                relevant_contexts += 1
            
            context_details.append({
                "context_index": i,
                "answer_overlap_ratio": answer_overlap_ratio,
                "question_overlap_ratio": question_overlap_ratio,
                "is_relevant": is_relevant,
                "answer_overlap_keywords": list(answer_overlap),
                "question_overlap_keywords": list(question_overlap)
            })
        
        # Precision = 유용한 컨텍스트 수 / 전체 컨텍스트 수
        precision = relevant_contexts / len(contexts)
        
        metadata = {
            "total_contexts": len(contexts),
            "relevant_contexts": relevant_contexts,
            "context_details": context_details,
            "answer_keywords": list(answer_keywords),
            "question_keywords": list(question_keywords)
        }
        
        return min(precision, 1.0), metadata
    
    def calculate_context_recall(
        self, 
        question: str, 
        contexts: List[str], 
        ground_truth_contexts: List[str] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Context Recall 계산 - 답변에 필요한 모든 정보가 검색된 컨텍스트에 포함된 비율
        
        Args:
            question: 질문 텍스트
            contexts: 검색된 컨텍스트 리스트
            ground_truth_contexts: 정답 컨텍스트 리스트 (제공되지 않으면 질문 기반으로 추정)
            
        Returns:
            recall 점수 (0.0-1.0), 상세 정보
        """
        if not question or not contexts:
            return 0.0, {"error": "Empty question or contexts"}
        
        # 질문에서 키워드 추출 (필요한 정보 추정)
        question_keywords = self._extract_keywords(question)
        if not question_keywords:
            return 0.0, {"error": "No keywords in question"}
        
        # Ground truth가 제공되지 않은 경우, 질문 키워드를 기준으로 판단
        if ground_truth_contexts is None:
            required_keywords = question_keywords
        else:
            # Ground truth 컨텍스트에서 키워드 추출
            required_keywords = set()
            for gt_context in ground_truth_contexts:
                gt_keywords = self._extract_keywords(gt_context)
                required_keywords.update(gt_keywords)
        
        if not required_keywords:
            return 0.0, {"error": "No required keywords identified"}
        
        # 검색된 컨텍스트에서 키워드 추출
        retrieved_keywords = set()
        context_details = []
        
        for i, context in enumerate(contexts):
            context_keywords = self._extract_keywords(context)
            retrieved_keywords.update(context_keywords)
            
            # 각 컨텍스트가 필요한 정보 중 얼마나 제공하는지 계산
            provided_keywords = context_keywords.intersection(required_keywords)
            provision_ratio = len(provided_keywords) / len(required_keywords) if required_keywords else 0
            
            context_details.append({
                "context_index": i,
                "provided_keywords": list(provided_keywords),
                "provision_ratio": provision_ratio,
                "context_keywords": list(context_keywords)
            })
        
        # Recall = 검색된 키워드 중 필요한 키워드 비율
        covered_keywords = retrieved_keywords.intersection(required_keywords)
        recall = len(covered_keywords) / len(required_keywords)
        
        metadata = {
            "required_keywords": list(required_keywords),
            "retrieved_keywords": list(retrieved_keywords),
            "covered_keywords": list(covered_keywords),
            "missing_keywords": list(required_keywords - covered_keywords),
            "context_details": context_details,
            "total_contexts": len(contexts),
            "used_ground_truth": ground_truth_contexts is not None
        }
        
        return min(recall, 1.0), metadata
    
    def evaluate_single_item(
        self,
        question_id: str,
        question: str,
        contexts: List[str],
        answer: str,
        ground_truth_contexts: List[str] = None
    ) -> SimpleRAGASResult:
        """
        단일 항목에 대한 모든 RAGAS 지표 계산
        
        Args:
            question_id: 질문 ID
            question: 질문 텍스트
            contexts: 컨텍스트 리스트
            answer: 답변 텍스트
            ground_truth_contexts: 정답 컨텍스트 리스트 (선택적)
            
        Returns:
            평가 결과
        """
        try:
            # Context Relevancy 계산
            context_relevancy, context_metadata = self.calculate_context_relevancy(question, contexts)
            
            # Context Precision 계산
            context_precision, precision_metadata = self.calculate_context_precision(question, contexts, answer)
            
            # Context Recall 계산
            context_recall, recall_metadata = self.calculate_context_recall(question, contexts, ground_truth_contexts)
            
            # Faithfulness 계산
            faithfulness, faithfulness_metadata = self.calculate_faithfulness(contexts, answer)
            
            # Answer Relevancy 계산
            answer_relevancy, answer_metadata = self.calculate_answer_relevancy(question, answer)
            
            # 메타데이터 통합
            metadata = {
                "context_relevancy_details": context_metadata,
                "context_precision_details": precision_metadata,
                "context_recall_details": recall_metadata,
                "faithfulness_details": faithfulness_metadata,
                "answer_relevancy_details": answer_metadata,
                "question_length": len(question),
                "answer_length": len(answer),
                "contexts_count": len(contexts)
            }
            
            result = SimpleRAGASResult(
                question_id=question_id,
                context_relevancy=context_relevancy,
                context_precision=context_precision,
                context_recall=context_recall,
                faithfulness=faithfulness,
                answer_relevancy=answer_relevancy,
                metadata=metadata
            )
            
            logger.debug(f"질문 {question_id} 평가 완료: CR={context_relevancy:.3f}, CP={context_precision:.3f}, CRe={context_recall:.3f}, F={faithfulness:.3f}, AR={answer_relevancy:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"질문 {question_id} 평가 중 오류: {str(e)}")
            
            # 오류 발생 시 기본값 반환
            return SimpleRAGASResult(
                question_id=question_id,
                context_relevancy=0.0,
                context_precision=0.0,
                context_recall=0.0,
                faithfulness=0.0,
                answer_relevancy=0.0,
                metadata={"error": str(e)}
            )
    
    def evaluate_batch(
        self,
        evaluation_items: List[Dict[str, Any]]
    ) -> List[SimpleRAGASResult]:
        """
        배치 평가 실행
        
        Args:
            evaluation_items: 평가 항목 리스트
                각 항목은 question_id, question, contexts, answer 포함
                
        Returns:
            평가 결과 리스트
        """
        results = []
        
        for item in evaluation_items:
            try:
                result = self.evaluate_single_item(
                    question_id=item.get("question_id", "unknown"),
                    question=item.get("question", ""),
                    contexts=item.get("contexts", []),
                    answer=item.get("answer", "")
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"평가 항목 처리 중 오류: {str(e)}")
                # 오류 발생 시에도 기본 결과 추가
                results.append(SimpleRAGASResult(
                    question_id=item.get("question_id", "error"),
                    context_relevancy=0.0,
                    context_precision=0.0,
                    context_recall=0.0,
                    faithfulness=0.0,
                    answer_relevancy=0.0,
                    metadata={"error": str(e)}
                ))
        
        logger.info(f"배치 평가 완료: {len(results)}개 항목 처리")
        
        return results
    
    def calculate_overall_metrics(self, results: List[SimpleRAGASResult]) -> Dict[str, float]:
        """
        전체 평가 지표 계산
        
        Args:
            results: 개별 평가 결과 리스트
            
        Returns:
            전체 평가 지표 딕셔너리
        """
        if not results:
            return {
                "context_relevancy": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "overall_score": 0.0
            }
        
        # 각 지표의 평균 계산
        context_relevancy_scores = [r.context_relevancy for r in results]
        context_precision_scores = [r.context_precision for r in results]
        context_recall_scores = [r.context_recall for r in results]
        faithfulness_scores = [r.faithfulness for r in results]
        answer_relevancy_scores = [r.answer_relevancy for r in results]
        
        avg_context_relevancy = sum(context_relevancy_scores) / len(context_relevancy_scores)
        avg_context_precision = sum(context_precision_scores) / len(context_precision_scores)
        avg_context_recall = sum(context_recall_scores) / len(context_recall_scores)
        avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores)
        avg_answer_relevancy = sum(answer_relevancy_scores) / len(answer_relevancy_scores)
        
        # 전체 점수 (5개 지표의 평균)
        overall_score = (avg_context_relevancy + avg_context_precision + avg_context_recall + avg_faithfulness + avg_answer_relevancy) / 5
        
        return {
            "context_relevancy": avg_context_relevancy,
            "context_precision": avg_context_precision,
            "context_recall": avg_context_recall,
            "faithfulness": avg_faithfulness,
            "answer_relevancy": avg_answer_relevancy,
            "overall_score": overall_score
        }
    
    def evaluate(self, question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
        """
        전체 RAGAS 평가 메인 인터페이스
        
        Args:
            question: 질문 텍스트
            answer: 답변 텍스트  
            contexts: 컨텍스트 리스트
            
        Returns:
            평가 결과 딕셔너리
        """
        try:
            # Context Relevancy 계산
            context_relevancy, _ = self.calculate_context_relevancy(question, contexts)
            
            # Context Precision 계산
            context_precision, _ = self.calculate_context_precision(question, contexts, answer)
            
            # Context Recall 계산
            context_recall, _ = self.calculate_context_recall(question, contexts)
            
            # Faithfulness 계산
            faithfulness, _ = self.calculate_faithfulness(contexts, answer)
            
            # Answer Relevancy 계산
            answer_relevancy, _ = self.calculate_answer_relevancy(question, answer)
            
            return {
                'context_relevancy': context_relevancy,
                'context_precision': context_precision,
                'context_recall': context_recall,
                'faithfulness': faithfulness,
                'answer_relevancy': answer_relevancy,
                'overall_score': (context_relevancy + context_precision + context_recall + faithfulness + answer_relevancy) / 5.0
            }
            
        except Exception as e:
            logger.error(f"평가 중 오류 발생: {e}")
            return {
                'context_relevancy': 0.0,
                'context_precision': 0.0,
                'context_recall': 0.0,
                'faithfulness': 0.0,
                'answer_relevancy': 0.0,
                'overall_score': 0.0
            }

# 테스트 함수
def test_simple_ragas_metrics():
    """간소화된 RAGAS 지표 테스트"""
    print("=== 간소화된 RAGAS 지표 테스트 ===")
    
    # 테스트 데이터
    test_question = "보험계약은 어떻게 성립되나요?"
    test_contexts = [
        "보험계약은 보험계약자의 청약과 보험회사의 승낙으로 이루어집니다.",
        "계약자는 청약을 한 날 또는 제1회 보험료를 납입한 날부터 15일 이내에 청약을 철회할 수 있습니다."
    ]
    test_answer = "보험계약은 보험계약자의 청약과 보험회사의 승낙으로 성립됩니다."
    
    # 평가기 초기화
    metrics = SimpleRAGASMetrics()
    
    # 단일 항목 평가
    result = metrics.evaluate_single_item("test_001", test_question, test_contexts, test_answer)
    
    print(f"질문: {test_question}")
    print(f"답변: {test_answer}")
    print(f"Context Relevancy: {result.context_relevancy:.3f}")
    print(f"Context Precision: {result.context_precision:.3f}")
    print(f"Context Recall: {result.context_recall:.3f}")
    print(f"Faithfulness: {result.faithfulness:.3f}")
    print(f"Answer Relevancy: {result.answer_relevancy:.3f}")
    
    # 전체 지표 계산
    overall_metrics = metrics.calculate_overall_metrics([result])
    print(f"Overall Score: {overall_metrics['overall_score']:.3f}")

if __name__ == "__main__":
    test_simple_ragas_metrics()
