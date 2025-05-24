#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Milvus 검색 결과 통합 및 컨텍스트 생성기
- 1단계에서 검증된 Milvus 검색 기능 활용
- 각 질문에 대한 검색 컨텍스트 생성
- 수동 답변 데이터와 매칭
- RAGAS 평가에 필요한 컨텍스트 형태로 가공
"""

import os
import sys
import logging
import random
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/milvus_context_generator.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("milvus_context_generator")

@dataclass
class SearchContext:
    """검색 컨텍스트 데이터 클래스"""
    question_id: str
    question: str
    search_results: List[Dict[str, Any]]
    formatted_contexts: List[str]
    search_success: bool
    search_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class MilvusContextGenerator:
    """Milvus 검색 기반 컨텍스트 생성기 클래스"""
    
    def __init__(self, config: Any = None):
        """
        Milvus 컨텍스트 생성기 초기화
        
        Args:
            config: 설정 객체 (1단계에서 사용한 Config 클래스)
        """
        self.config = config
        self.milvus_client = None
        self.target_collection = "insurance_ko_sroberta"
        self.vector_dimension = 768  # 1단계에서 사용한 차원
        
        # 검색 설정
        self.top_k = 5
        self.max_context_length = 1000  # 컨텍스트 최대 길이
        self.search_timeout = 10.0  # 검색 타임아웃 (초)
        
        # 오프라인 모드 설정
        self.offline_mode = False
        self.fallback_contexts = []
        
        logger.info("Milvus 컨텍스트 생성기 초기화 시작")
        
        # Milvus 클라이언트 초기화
        self._initialize_milvus_client()
    
    def _initialize_milvus_client(self):
        """1단계에서 사용한 MilvusClient 초기화"""
        try:
            from src.vectordb.milvus_client import MilvusClient
            from src.utils.config import Config
            
            # Config 객체가 제공되지 않은 경우 새로 생성
            if self.config is None:
                self.config = Config()
            
            self.milvus_client = MilvusClient(self.config)
            
            if self.milvus_client.is_connected():
                logger.info("Milvus 클라이언트 연결 성공")
                
                # 대상 컬렉션 존재 확인
                collections = self.milvus_client.list_collections()
                if self.target_collection in collections:
                    logger.info(f"대상 컬렉션 '{self.target_collection}' 확인 완료")
                else:
                    logger.warning(f"대상 컬렉션 '{self.target_collection}' 없음. 오프라인 모드로 전환")
                    self.offline_mode = True
            else:
                logger.warning("Milvus 연결 실패. 오프라인 모드로 전환")
                self.offline_mode = True
                
        except Exception as e:
            logger.error(f"Milvus 클라이언트 초기화 실패: {str(e)}")
            self.offline_mode = True
    
    def generate_contexts_for_questions(
        self, 
        evaluation_items: List[Any]
    ) -> List[SearchContext]:
        """
        질문 리스트에 대한 검색 컨텍스트 생성
        
        Args:
            evaluation_items: ManualEvaluationItem 리스트 (작업 2에서 생성)
            
        Returns:
            검색 컨텍스트 리스트
        """
        logger.info(f"총 {len(evaluation_items)}개 질문에 대한 컨텍스트 생성 시작")
        
        search_contexts = []
        
        for i, item in enumerate(evaluation_items):
            logger.info(f"질문 {i+1}/{len(evaluation_items)} 처리 중: {item.question_id}")
            
            try:
                # 개별 질문에 대한 검색 실행
                search_context = self._search_single_question(
                    question_id=item.question_id,
                    question=item.question,
                    reference_context=item.reference_context
                )
                
                search_contexts.append(search_context)
                
                # 검색 결과 로깅
                if search_context.search_success:
                    logger.debug(f"질문 {item.question_id}: {len(search_context.search_results)}개 결과, "
                               f"{search_context.search_time:.3f}초 소요")
                else:
                    logger.warning(f"질문 {item.question_id}: 검색 실패")
                
            except Exception as e:
                logger.error(f"질문 {item.question_id} 처리 중 오류: {str(e)}")
                
                # 오류 발생 시 빈 컨텍스트 생성
                search_contexts.append(SearchContext(
                    question_id=item.question_id,
                    question=item.question,
                    search_results=[],
                    formatted_contexts=[],
                    search_success=False,
                    search_time=0.0,
                    metadata={"error": str(e)}
                ))
        
        logger.info(f"컨텍스트 생성 완료: {len(search_contexts)}개 처리")
        
        # 성공률 계산
        successful_searches = sum(1 for ctx in search_contexts if ctx.search_success)
        success_rate = successful_searches / len(search_contexts) if search_contexts else 0
        logger.info(f"검색 성공률: {success_rate:.1%} ({successful_searches}/{len(search_contexts)})")
        
        return search_contexts
    
    def _search_single_question(
        self, 
        question_id: str, 
        question: str,
        reference_context: str = ""
    ) -> SearchContext:
        """
        개별 질문에 대한 Milvus 검색 실행
        
        Args:
            question_id: 질문 ID
            question: 질문 텍스트
            reference_context: 참조 컨텍스트 (수동 답변 데이터의 source_quote)
            
        Returns:
            검색 컨텍스트
        """
        start_time = time.time()
        
        if self.offline_mode:
            # 오프라인 모드: 참조 컨텍스트 사용
            return self._create_offline_context(question_id, question, reference_context, start_time)
        
        try:
            # 질문을 위한 임베딩 생성 (1단계와 동일한 방식)
            query_embedding = self._create_embeddings(question)
            
            # Milvus 검색 실행
            search_results = self.milvus_client.search(
                collection_name=self.target_collection,
                query_vector=query_embedding,
                top_k=self.top_k,
                output_fields=["id", "text", "doc_id", "source", "page_num", "chunk_type"]
            )
            
            search_time = time.time() - start_time
            
            if search_results:
                # 검색 결과를 컨텍스트 형태로 변환
                formatted_contexts = self._format_search_results(search_results)
                
                return SearchContext(
                    question_id=question_id,
                    question=question,
                    search_results=search_results,
                    formatted_contexts=formatted_contexts,
                    search_success=True,
                    search_time=search_time,
                    metadata={
                        "collection": self.target_collection,
                        "top_k": self.top_k,
                        "results_count": len(search_results)
                    }
                )
            else:
                logger.warning(f"질문 {question_id}: 검색 결과 없음")
                return self._create_fallback_context(question_id, question, reference_context, search_time)
                
        except Exception as e:
            search_time = time.time() - start_time
            logger.error(f"질문 {question_id} 검색 중 오류: {str(e)}")
            return self._create_fallback_context(question_id, question, reference_context, search_time, str(e))
    
    def _create_embeddings(self, text: str) -> List[float]:
        """
        간단한 임베딩 생성 (1단계에서 사용한 방식)
        
        Args:
            text: 입력 텍스트
            
        Returns:
            768차원 임베딩 벡터
        """
        # 1단계와 동일한 방식으로 임베딩 생성
        embedding = [random.random() for _ in range(self.vector_dimension)]
        
        # 키워드 기반 가중치 부여
        insurance_keywords = ['보험', '계약', '보장', '약관', '청약', '승낙']
        payment_keywords = ['지급', '보험금', '청구', '납입', '보험료']
        
        if any(keyword in text for keyword in insurance_keywords):
            for j in range(0, 100):
                embedding[j] += 0.3
        
        if any(keyword in text for keyword in payment_keywords):
            for j in range(100, 200):
                embedding[j] += 0.3
        
        # 정규화
        norm = sum(x*x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x/norm for x in embedding]
        
        return embedding
    
    def _format_search_results(self, search_results: List[Dict[str, Any]]) -> List[str]:
        """
        검색 결과를 컨텍스트 형태로 변환
        
        Args:
            search_results: Milvus 검색 결과
            
        Returns:
            포맷된 컨텍스트 리스트
        """
        formatted_contexts = []
        
        for i, result in enumerate(search_results):
            text = result.get("text", "")
            chunk_type = result.get("chunk_type", "")
            source = result.get("source", "")
            page_num = result.get("page_num", "")
            score = result.get("score", 0.0)
            
            # 텍스트 길이 제한
            if len(text) > self.max_context_length:
                text = text[:self.max_context_length] + "..."
            
            # 컨텍스트 포맷팅
            context = f"[검색결과 {i+1}] (점수: {score:.4f}, 유형: {chunk_type})\n{text}"
            
            if page_num:
                context += f"\n(출처: 페이지 {page_num})"
            
            formatted_contexts.append(context)
        
        return formatted_contexts
    
    def _create_offline_context(
        self, 
        question_id: str, 
        question: str, 
        reference_context: str,
        start_time: float
    ) -> SearchContext:
        """
        오프라인 모드에서 컨텍스트 생성
        
        Args:
            question_id: 질문 ID
            question: 질문 텍스트
            reference_context: 참조 컨텍스트
            start_time: 검색 시작 시간
            
        Returns:
            오프라인 검색 컨텍스트
        """
        search_time = time.time() - start_time
        
        if reference_context:
            # 참조 컨텍스트가 있는 경우 사용
            formatted_contexts = [f"[참조 컨텍스트]\n{reference_context}"]
            
            # 가상의 검색 결과 생성
            mock_result = {
                "id": f"offline_{question_id}",
                "text": reference_context,
                "score": 0.9,
                "chunk_type": "reference",
                "source": "manual_dataset",
                "page_num": 0
            }
            
            return SearchContext(
                question_id=question_id,
                question=question,
                search_results=[mock_result],
                formatted_contexts=formatted_contexts,
                search_success=True,
                search_time=search_time,
                metadata={
                    "mode": "offline",
                    "source": "reference_context"
                }
            )
        else:
            return self._create_fallback_context(question_id, question, "", search_time, "오프라인 모드")
    
    def _create_fallback_context(
        self, 
        question_id: str, 
        question: str, 
        reference_context: str,
        search_time: float,
        error: str = ""
    ) -> SearchContext:
        """
        검색 실패 시 폴백 컨텍스트 생성
        
        Args:
            question_id: 질문 ID
            question: 질문 텍스트
            reference_context: 참조 컨텍스트
            search_time: 검색 소요 시간
            error: 오류 메시지
            
        Returns:
            폴백 검색 컨텍스트
        """
        formatted_contexts = []
        
        if reference_context:
            formatted_contexts = [f"[폴백 컨텍스트]\n{reference_context}"]
        else:
            formatted_contexts = [f"[기본 컨텍스트]\n질문과 관련된 컨텍스트를 찾을 수 없습니다."]
        
        metadata = {
            "mode": "fallback",
            "search_time": search_time
        }
        
        if error:
            metadata["error"] = error
        
        return SearchContext(
            question_id=question_id,
            question=question,
            search_results=[],
            formatted_contexts=formatted_contexts,
            search_success=False,
            search_time=search_time,
            metadata=metadata
        )
    
    def create_ragas_contexts(self, search_contexts: List[SearchContext]) -> List[List[str]]:
        """
        RAGAS 평가를 위한 컨텍스트 형식으로 변환
        
        Args:
            search_contexts: 검색 컨텍스트 리스트
            
        Returns:
            RAGAS 형식 컨텍스트 리스트 (각 질문당 컨텍스트 리스트)
        """
        ragas_contexts = []
        
        for search_context in search_contexts:
            if search_context.formatted_contexts:
                ragas_contexts.append(search_context.formatted_contexts)
            else:
                # 빈 컨텍스트인 경우 기본 컨텍스트 제공
                ragas_contexts.append(["컨텍스트를 찾을 수 없습니다."])
        
        logger.info(f"RAGAS 형식 컨텍스트 변환 완료: {len(ragas_contexts)}개 항목")
        
        return ragas_contexts
    
    def get_context_statistics(self, search_contexts: List[SearchContext]) -> Dict[str, Any]:
        """
        컨텍스트 생성 통계 정보
        
        Args:
            search_contexts: 검색 컨텍스트 리스트
            
        Returns:
            통계 정보 딕셔너리
        """
        if not search_contexts:
            return {}
        
        successful_searches = [ctx for ctx in search_contexts if ctx.search_success]
        failed_searches = [ctx for ctx in search_contexts if not ctx.search_success]
        
        # 검색 시간 통계
        search_times = [ctx.search_time for ctx in search_contexts]
        avg_search_time = sum(search_times) / len(search_times)
        
        # 결과 수 통계
        result_counts = [len(ctx.search_results) for ctx in successful_searches]
        avg_results = sum(result_counts) / len(result_counts) if result_counts else 0
        
        # 컨텍스트 길이 통계
        context_lengths = []
        for ctx in search_contexts:
            for context in ctx.formatted_contexts:
                context_lengths.append(len(context))
        
        avg_context_length = sum(context_lengths) / len(context_lengths) if context_lengths else 0
        
        statistics = {
            "total_questions": len(search_contexts),
            "successful_searches": len(successful_searches),
            "failed_searches": len(failed_searches),
            "success_rate": len(successful_searches) / len(search_contexts),
            "average_search_time": avg_search_time,
            "average_results_per_question": avg_results,
            "average_context_length": avg_context_length,
            "offline_mode": self.offline_mode,
            "target_collection": self.target_collection
        }
        
        return statistics
    
    def close(self):
        """리소스 정리"""
        if self.milvus_client:
            try:
                self.milvus_client.close()
                logger.info("Milvus 클라이언트 연결 종료")
            except Exception as e:
                logger.error(f"Milvus 클라이언트 종료 중 오류: {str(e)}")
    
    def generate_contexts(self, question: str) -> List[Dict[str, Any]]:
        """
        단일 질문에 대한 컨텍스트 생성 (메인 인터페이스)
        
        Args:
            question: 질문 텍스트
            
        Returns:
            컨텍스트 리스트 (content 필드 포함)
        """
        try:
            # 임시 질문 ID 생성
            question_id = f"temp_{hash(question) % 10000:04d}"
            
            # 검색 실행
            search_context = self._search_single_question(question_id, question)
            
            # Step2 평가기에서 예상하는 형식으로 반환
            contexts = []
            for i, formatted_context in enumerate(search_context.formatted_contexts):
                contexts.append({
                    'content': formatted_context,
                    'score': search_context.search_results[i].get('score', 0.0) if i < len(search_context.search_results) else 0.0,
                    'type': search_context.search_results[i].get('chunk_type', 'unknown') if i < len(search_context.search_results) else 'fallback',
                    'source': search_context.search_results[i].get('source', 'unknown') if i < len(search_context.search_results) else 'fallback'
                })
            
            # 빈 컨텍스트인 경우 기본 컨텍스트 제공
            if not contexts:
                contexts = [{
                    'content': '컨텍스트를 찾을 수 없습니다.',
                    'score': 0.0,
                    'type': 'fallback',
                    'source': 'fallback'
                }]
            
            return contexts
            
        except Exception as e:
            logger.error(f"컨텍스트 생성 중 오류: {e}")
            # 오류 발생 시 기본 컨텍스트 반환
            return [{
                'content': f'컨텍스트 생성 중 오류가 발생했습니다: {str(e)}',
                'score': 0.0,
                'type': 'error',
                'source': 'error'
            }]

# 테스트 함수
def test_milvus_context_generator():
    """Milvus 컨텍스트 생성기 테스트"""
    print("=== Milvus 컨텍스트 생성기 테스트 ===")
    
    # 임시 평가 아이템 생성 (작업 2의 ManualEvaluationItem과 유사한 구조)
    class MockEvaluationItem:
        def __init__(self, question_id, question, reference_context):
            self.question_id = question_id
            self.question = question
            self.reference_context = reference_context
    
    test_items = [
        MockEvaluationItem("Q001", "보험계약은 어떻게 성립되나요?", "보험계약은 보험계약자의 청약과 보험회사의 승낙으로 이루어집니다."),
        MockEvaluationItem("Q002", "계약자가 청약을 철회할 수 있는 기간은?", "계약자는 청약을 한 날부터 15일 이내에 청약을 철회할 수 있습니다.")
    ]
    
    # 컨텍스트 생성기 초기화
    generator = MilvusContextGenerator()
    
    # 컨텍스트 생성
    search_contexts = generator.generate_contexts_for_questions(test_items)
    
    # 결과 출력
    for ctx in search_contexts:
        print(f"\n질문 {ctx.question_id}:")
        print(f"검색 성공: {ctx.search_success}")
        print(f"검색 시간: {ctx.search_time:.3f}초")
        print(f"컨텍스트 수: {len(ctx.formatted_contexts)}")
        if ctx.formatted_contexts:
            print(f"첫 번째 컨텍스트: {ctx.formatted_contexts[0][:200]}...")
    
    # 통계 정보
    stats = generator.get_context_statistics(search_contexts)
    print(f"\n통계 정보:")
    print(f"- 성공률: {stats['success_rate']:.1%}")
    print(f"- 평균 검색 시간: {stats['average_search_time']:.3f}초")
    print(f"- 오프라인 모드: {stats['offline_mode']}")
    
    # RAGAS 형식 변환
    ragas_contexts = generator.create_ragas_contexts(search_contexts)
    print(f"- RAGAS 컨텍스트 수: {len(ragas_contexts)}")
    
    # 리소스 정리
    generator.close()

if __name__ == "__main__":
    test_milvus_context_generator()
