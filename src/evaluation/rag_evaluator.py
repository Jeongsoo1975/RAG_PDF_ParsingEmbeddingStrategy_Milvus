#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RAG 시스템 평가 도구
- Pinecone 벡터 검색 성능 검증
- 보험 약관 질의응답 평가
"""

import os
import json
import argparse
import logging
import time
from typing import Dict, Any, List

import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass, field, asdict
import numpy as np
from tqdm import tqdm

# Pinecone 및 임베딩 관련 임포트
try:
    import pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    print("Pinecone 라이브러리를 찾을 수 없습니다. 'pip install pinecone-client'로 설치하세요.")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("SentenceTransformer를 찾을 수 없습니다. 'pip install sentence-transformers'로 설치하세요.")

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("rag_evaluator")

@dataclass
class EvaluationQuestion:
    """평가용 질문 데이터 클래스"""
    id: str
    type: str  # structural, factual, clausal, inferential, relational
    difficulty: str  # easy, medium, hard
    text: str
    related_sections: List[str]
    page_numbers: List[int]
    gold_standard: Dict[str, Any]

@dataclass
class EvaluationResult:
    """평가 결과 데이터 클래스"""
    question_id: str
    question_type: str
    question_difficulty: str
    question_text: str
    retrieved_documents: List[Dict[str, Any]]
    retrieval_time: float
    relevant_docs_count: int
    irrelevant_docs_count: int
    precision: float
    recall: float
    f1_score: float
    ndcg: float
    has_answer: bool
    metrics: Dict[str, Any] = field(default_factory=dict)

class RAGEvaluator:
    """RAG 시스템 평가 클래스"""
    
    def __init__(self, config=None):
        """
        평가 도구 초기화
        
        Args:
            config: 설정 파라미터 딕셔너리
        """
        # 기본값 설정
        default_config = {
            'embedding_model': 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
            'pinecone_index': None,
            'top_k': 5,
            'embedding_path': None
        }
        
        # 사용자 설정과 기본값 병합
        config = {**default_config, **(config or {})}
        
        # 임베딩 모델 로딩
        try:
            self.embedding_model = SentenceTransformer(config['embedding_model'])
            logger.info(f"임베딩 모델 로딩: {config['embedding_model']}")
        except Exception as e:
            logger.error(f"임베딩 모델 로딩 실패: {str(e)}")
            self.embedding_model = None
        
        # Pinecone 인덱스 설정
        original_index_name = config['pinecone_index']
        
        # 인덱스 이름 매핑 - 이제 원래 인덱스 이름이 모두 ci-20060401로 매핑됨
        self.pinecone_index_name = "ci-20060401"
        logger.info(f"인덱스 매핑: {original_index_name} -> {self.pinecone_index_name}")
        
        self.top_k = config['top_k']
        
        # Pinecone 초기화
        self._init_pinecone()
        
        # 로컬 임베딩 로딩
        self.local_embeddings = None
        if config['embedding_path']:
            try:
                with open(config['embedding_path'], 'r', encoding='utf-8') as f:
                    self.local_embeddings = json.load(f)
            except Exception as e:
                logger.error(f"로컬 임베딩 로딩 실패: {str(e)}")
    
    def _init_pinecone(self):
        """Pinecone 초기화"""
        if not PINECONE_AVAILABLE:
            logger.error("Pinecone을 사용할 수 없습니다.")
            self.pinecone_index = None
            return
        
        try:
            # Pinecone API 키 확인
            api_key = os.environ.get('PINECONE_API_KEY')
            environment = os.environ.get('PINECONE_ENVIRONMENT', 'asia-southeast1-gcp')
            
            if not api_key:
                logger.error("PINECONE_API_KEY 환경 변수가 설정되지 않았습니다.")
                self.pinecone_index = None
                return
            
            # Pinecone 클라이언트 버전 확인 및 초기화
            import inspect
            pc_init_params = inspect.signature(pinecone.Pinecone.__init__).parameters
            
            # Pinecone 클라이언트 v3.x
            logger.info("Pinecone 클라이언트 v3.x 사용")
            pc = pinecone.Pinecone(api_key=api_key, environment=environment)
            
            # 인덱스 존재 확인
            index_list = pc.list_indexes()
            index_names = [idx.name for idx in index_list.indexes] if hasattr(index_list, 'indexes') else []
            
            if self.pinecone_index_name not in index_names:
                logger.error(f"Pinecone 인덱스 '{self.pinecone_index_name}' 를 찾을 수 없습니다.")
                self.pinecone_index = None
                return
            
            # 인덱스 연결
            self.pinecone_index = pc.Index(self.pinecone_index_name)
            logger.info(f"Pinecone 인덱스 '{self.pinecone_index_name}' 연결 완료")
            
            try:
                # 인덱스 정보 확인
                index_stats = pc.describe_index(self.pinecone_index_name)
                logger.info(f"인덱스 통계: {index_stats}")
            except Exception as e_stats:
                logger.warning(f"인덱스 통계 조회 실패: {e_stats}")
            
        except Exception as e:
            logger.error(f"Pinecone 초기화 실패: {str(e)}")
            self.pinecone_index = None
    
    def load_questions(self, dataset_path: str) -> List[EvaluationQuestion]:
        """
        평가용 질문 데이터셋 로드
        
        Args:
            dataset_path: 평가 데이터셋 JSON 파일 경로
            
        Returns:
            질문 목록
        """
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 데이터셋 정보 로깅
            logger.info(f"데이터셋 제목: {data.get('dataset_info', {}).get('document_title', '알 수 없음')}")
            logger.info(f"질문 수: {data.get('dataset_info', {}).get('total_questions', len(data.get('questions', [])))}")
            
            # 질문 목록 생성
            questions = []
            for q in data.get('questions', []):
                question = EvaluationQuestion(
                    id=q.get('id', ''),
                    type=q.get('type', ''),
                    difficulty=q.get('difficulty', ''),
                    text=q.get('text', ''),
                    related_sections=q.get('related_sections', []),
                    page_numbers=q.get('page_numbers', []),
                    gold_standard=q.get('gold_standard', {})
                )
                questions.append(question)
            
            logger.info(f"{len(questions)}개 질문 로딩 완료")
            return questions
        
        except Exception as e:
            logger.error(f"질문 데이터셋 로딩 실패: {str(e)}")
            return []
    
    def evaluate_question(self, question: EvaluationQuestion) -> EvaluationResult:
        """
        단일 질문에 대한 RAG 성능 평가
        
        Args:
            question: 평가할 질문
            
        Returns:
            평가 결과
        """
        logger.info(f"질문 평가 시작 [ID: {question.id}]: {question.text}")
        
        # 검색 시작 시간
        start_time = time.time()
        
        # 질문 임베딩 생성
        query_embedding = self.embedding_model.encode(question.text)
        
        # 검색 방법 선택
        retrieved_docs = []
        
        # Pinecone 인덱스 사용 (우선순위 1)
        if self.pinecone_index:
            try:
                retrieved_docs = self._query_pinecone(query_embedding)
            except Exception as e:
                logger.error(f"Pinecone 검색 오류: {str(e)}")
        
        # 로컬 임베딩 사용 (대체 방법)
        elif self.local_embeddings:
            # 코사인 유사도 계산
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            
            # 쿼리 임베딩을 NumPy 배열로 변환
            query_embedding_np = query_embedding.reshape(1, -1)
            
            # 로컬 임베딩과 유사도 계산
            similarities = []
            for chunk in self.local_embeddings.get('chunks', []):
                chunk_embedding = chunk.get('metadata', {}).get('embedding', [])
                if chunk_embedding:
                    sim = cosine_similarity(query_embedding_np, np.array(chunk_embedding).reshape(1, -1))[0][0]
                    similarities.append((sim, chunk))
            
            # 유사도 기준으로 정렬 및 상위 k개 선택
            similarities.sort(key=lambda x: x[0], reverse=True)
            retrieved_docs = [
                {
                    "id": doc['chunk_id'],
                    "score": score,
                    "content": doc.get('text', ''),
                    "metadata": {k: v for k, v in doc.items() if k != 'text'}
                } for score, doc in similarities[:self.top_k]
            ]
        
        # 검색 소요 시간
        retrieval_time = time.time() - start_time
        
        # 관련 문서 평가
        relevant_docs = []
        irrelevant_docs = []
        
        for doc in retrieved_docs:
            is_relevant = self._is_document_relevant(doc, question)
            if is_relevant:
                relevant_docs.append(doc)
            else:
                irrelevant_docs.append(doc)
        
        # 평가 지표 계산
        precision = len(relevant_docs) / max(1, len(retrieved_docs))
        
        # 전체 관련 문서 수는 이상적으로는 gold_standard의 essential_elements 수와 같다고 가정
        total_relevant = len(question.gold_standard.get("essential_elements", []))
        recall = len(relevant_docs) / max(1, total_relevant)
        
        f1_score = 2 * (precision * recall) / max(0.001, precision + recall)
        
        # NDCG 계산
        ndcg = self._calculate_ndcg(retrieved_docs, question)
        
        # 답변 포함 여부 확인 
        has_answer = self._contains_answer(retrieved_docs, question)
        
        # 결과 생성
        result = EvaluationResult(
            question_id=question.id,
            question_type=question.type,
            question_difficulty=question.difficulty,
            question_text=question.text,
            retrieved_documents=retrieved_docs,
            retrieval_time=retrieval_time,
            relevant_docs_count=len(relevant_docs),
            irrelevant_docs_count=len(irrelevant_docs),
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            ndcg=ndcg,
            has_answer=has_answer,
            metrics={
                "retrieval_time_ms": retrieval_time * 1000,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "ndcg": ndcg,
                "has_answer": has_answer
            }
        )
        
        logger.info(f"질문 평가 완료 [ID: {question.id}]: 정밀도 {precision:.2f}, 재현율 {recall:.2f}, F1 {f1_score:.2f}")
        return result
    
    def _is_document_relevant(self, doc: Dict[str, Any], question: EvaluationQuestion) -> bool:
        """
        문서가 질문과 관련이 있는지 평가
        
        Args:
            doc: 평가할 문서
            question: 질문
            
        Returns:
            관련성 여부
        """
        # 관련 섹션 체크
        doc_section = doc.get("metadata", {}).get("section_title", "")
        if any(section in doc_section for section in question.related_sections):
            return True
        
        # 페이지 번호 체크
        doc_page = int(doc.get("metadata", {}).get("page_num", -1))
        if doc_page in question.page_numbers:
            return True
        
        # 중요 요소 체크
        essential_elements = question.gold_standard.get("essential_elements", [])
        content = doc.get("content", "").lower()
        
        # 중요 요소의 절반 이상이 문서에 포함되면 관련 있다고 판단
        matches = sum(1 for element in essential_elements if element.lower() in content)
        if matches >= max(1, len(essential_elements) // 2):
            return True
        
        # 출처 인용 체크
        source_quote = question.gold_standard.get("source_quote", "").lower()
        if source_quote and len(source_quote) > 20:
            # 출처 인용의 일부가 문서에 포함되면 관련 있다고 판단
            quote_parts = source_quote.split()
            for i in range(len(quote_parts) - 3):
                phrase = " ".join(quote_parts[i:i+3])
                if phrase in content:
                    return True
        
        return False
    
    def _calculate_ndcg(self, docs: List[Dict[str, Any]], question: EvaluationQuestion) -> float:
        """
        NDCG(Normalized Discounted Cumulative Gain) 계산
        
        Args:
            docs: 검색된 문서 목록
            question: 질문
            
        Returns:
            NDCG 점수
        """
        if not docs:
            return 0.0
        
        # 각 문서의 관련성 점수 (0~3)
        relevance_scores = []
        
        for i, doc in enumerate(docs):
            score = 0
            
            # 1) 관련 섹션 체크 (+1)
            doc_section = doc.get("metadata", {}).get("section_title", "")
            if any(section in doc_section for section in question.related_sections):
                score += 1
            
            # 2) 페이지 번호 체크 (+1)
            doc_page = int(doc.get("metadata", {}).get("page_num", -1))
            if doc_page in question.page_numbers:
                score += 1
            
            # 3) 중요 요소 체크 (+1)
            essential_elements = question.gold_standard.get("essential_elements", [])
            content = doc.get("content", "").lower()
            
            matches = sum(1 for element in essential_elements if element.lower() in content)
            if matches >= max(1, len(essential_elements) // 3):
                score += 1
            
            relevance_scores.append(score)
        
        # DCG 계산
        dcg = 0
        for i, score in enumerate(relevance_scores):
            dcg += score / np.log2(i + 2)  # i+2는 1-기반 인덱싱을 위함
        
        # 이상적인 정렬 (내림차순)
        ideal_scores = sorted(relevance_scores, reverse=True)
        
        # IDCG 계산
        idcg = 0
        for i, score in enumerate(ideal_scores):
            idcg += score / np.log2(i + 2)
        
        # NDCG 계산
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def _contains_answer(self, docs: List[Dict[str, Any]], question: EvaluationQuestion) -> bool:
        """
        검색된 문서들이 질문의 답변을 포함하는지 확인
        
        Args:
            docs: 검색된 문서 목록
            question: 질문
            
        Returns:
            답변 포함 여부
        """
        # Gold standard 답변 확인
        gold_answer = question.gold_standard.get("answer", "").lower()
        if not gold_answer:
            return False
        
        # 필수 요소 확인
        essential_elements = question.gold_standard.get("essential_elements", [])
        
        # 모든 문서의 텍스트 결합
        all_content = " ".join([doc.get("content", "").lower() for doc in docs])
        
        # 필수 요소의 절반 이상이 포함되면 답변이 있다고 판단
        matches = sum(1 for element in essential_elements if element.lower() in all_content)
        if matches >= max(1, len(essential_elements) * 2 // 3):
            return True
        
        # 출처 인용 체크
        source_quote = question.gold_standard.get("source_quote", "").lower()
        if source_quote and len(source_quote) > 20:
            # 출처 인용의 일부가 문서에 포함되면 답변이 있다고 판단
            quote_parts = source_quote.split()
            for i in range(len(quote_parts) - 5):
                phrase = " ".join(quote_parts[i:i+5])
                if phrase in all_content:
                    return True
        
        return False
    
    def evaluate_dataset(self, questions: List[EvaluationQuestion]) -> Dict[str, Any]:
        """
        전체 데이터셋 평가
        
        Args:
            questions: 평가할 질문 목록
            
        Returns:
            평가 결과 통계
        """
        if not questions:
            logger.error("평가할 질문이 없습니다.")
            return {}
        
        results = []
        
        # 모든 질문 평가
        for question in tqdm(questions, desc="질문 평가 중"):
            result = self.evaluate_question(question)
            results.append(result)
        
        # 결과 분석
        stats = self._analyze_results(results)
        
        return stats
    
    def _analyze_results(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """
        평가 결과 분석
        
        Args:
            results: 평가 결과 목록
            
        Returns:
            분석 결과
        """
        if not results:
            return {}
        
        # 전체 통계
        total_questions = len(results)
        total_precision = sum(r.precision for r in results) / total_questions
        total_recall = sum(r.recall for r in results) / total_questions
        total_f1 = sum(r.f1_score for r in results) / total_questions
        total_ndcg = sum(r.ndcg for r in results) / total_questions
        total_has_answer = sum(1 for r in results if r.has_answer) / total_questions
        total_retrieval_time = sum(r.retrieval_time for r in results) / total_questions
        
        # 질문 유형별 통계
        types = {}
        for result in results:
            q_type = result.question_type
            if q_type not in types:
                types[q_type] = {
                    "count": 0,
                    "precision": 0,
                    "recall": 0,
                    "f1_score": 0,
                    "ndcg": 0,
                    "has_answer": 0
                }
            
            types[q_type]["count"] += 1
            types[q_type]["precision"] += result.precision
            types[q_type]["recall"] += result.recall
            types[q_type]["f1_score"] += result.f1_score
            types[q_type]["ndcg"] += result.ndcg
            types[q_type]["has_answer"] += 1 if result.has_answer else 0
        
        # 평균 계산
        for q_type, stats in types.items():
            count = stats["count"]
            if count > 0:
                stats["precision"] /= count
                stats["recall"] /= count
                stats["f1_score"] /= count
                stats["ndcg"] /= count
                stats["has_answer"] /= count
        
        # 난이도별 통계
        difficulties = {}
        for result in results:
            difficulty = result.question_difficulty
            if difficulty not in difficulties:
                difficulties[difficulty] = {
                    "count": 0,
                    "precision": 0,
                    "recall": 0,
                    "f1_score": 0,
                    "ndcg": 0,
                    "has_answer": 0
                }
            
            difficulties[difficulty]["count"] += 1
            difficulties[difficulty]["precision"] += result.precision
            difficulties[difficulty]["recall"] += result.recall
            difficulties[difficulty]["f1_score"] += result.f1_score
            difficulties[difficulty]["ndcg"] += result.ndcg
            difficulties[difficulty]["has_answer"] += 1 if result.has_answer else 0
        
        # 평균 계산
        for difficulty, stats in difficulties.items():
            count = stats["count"]
            if count > 0:
                stats["precision"] /= count
                stats["recall"] /= count
                stats["f1_score"] /= count
                stats["ndcg"] /= count
                stats["has_answer"] /= count
        
        return {
            "total": {
                "questions": total_questions,
                "precision": total_precision,
                "recall": total_recall,
                "f1_score": total_f1,
                "ndcg": total_ndcg,
                "has_answer_rate": total_has_answer,
                "avg_retrieval_time_ms": total_retrieval_time * 1000
            },
            "by_type": types,
            "by_difficulty": difficulties,
            "results": [asdict(r) for r in results]
        }
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """
        평가 결과 저장
        
        Args:
            results: 평가 결과
            output_path: 결과 저장 경로
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"평가 결과 저장 완료: {output_path}")
        except Exception as e:
            logger.error(f"평가 결과 저장 실패: {str(e)}")

    def _query_pinecone(self, query_embedding, top_k=None):
        """Pinecone에서 쿼리 실행"""
        if not self.pinecone_index:
            logger.warning("Pinecone 인덱스가 초기화되지 않았습니다.")
            return []
        
        if top_k is None:
            top_k = self.top_k
        
        logger.debug(f"Pinecone 쿼리 시작 (top_k={top_k})")
        
        try:
            # Pinecone v3.x 호환 쿼리 - CI_20060401 네임스페이스 사용
            logger.info(f"CI_20060401 네임스페이스에서 검색 수행 중 (top_k={top_k})")
            search_results = self.pinecone_index.query(
                vector=query_embedding.tolist(),
                top_k=top_k,
                include_metadata=True,
                namespace="CI_20060401"  # 대문자 네임스페이스 사용
            )
            
            matches = getattr(search_results, 'matches', [])
            logger.debug(f"CI_20060401 네임스페이스 검색 결과: {len(matches)}개 항목 찾음")
            
            # 검색 결과가 없는 경우 디버그 로그만 출력
            if not matches:
                logger.warning("CI_20060401 네임스페이스에서 결과가 없습니다.")
            else:
                # 성공적으로 결과를 찾은 경우 메타데이터 확인
                for i, match in enumerate(matches[:2]):  # 첫 두 개만 로그
                    if hasattr(match, 'metadata') and match.metadata:
                        metadata_keys = list(match.metadata.keys())
                        logger.debug(f"결과 {i+1} 메타데이터 키: {metadata_keys}")
                        if 'text' in match.metadata:
                            text_snippet = match.metadata['text'][:50] + "..." if len(match.metadata['text']) > 50 else match.metadata['text']
                            logger.debug(f"결과 {i+1} 내용: {text_snippet}")
                    else:
                        logger.debug(f"결과 {i+1}에 메타데이터가 없습니다.")
            
            # 검색 결과를 RetrievedDocument 객체 리스트로 변환
            retrieved_docs = []
            
            for match in matches:
                if hasattr(match, 'metadata') and match.metadata and 'text' in match.metadata:
                    doc = {
                        "id": match.id,
                        "score": match.score,
                        "content": match.metadata.get('text', ''),
                        "metadata": {k: v for k, v in match.metadata.items() if k != 'text'}
                    }
                    retrieved_docs.append(doc)
            
            logger.info(f"변환된 문서 수: {len(retrieved_docs)}")
            return retrieved_docs
            
        except Exception as e:
            logger.error(f"Pinecone 쿼리 중 오류 발생: {e}")
            import traceback
            logger.error(f"상세 오류: {traceback.format_exc()}")
            return []


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='RAG 시스템 평가 도구')
    parser.add_argument('--dataset', type=str, required=True, help='평가 데이터셋 JSON 파일 경로')
    parser.add_argument('--output', type=str, default='rag_evaluation_results.json', help='결과 저장 파일 경로')
    parser.add_argument('--model', type=str, default='sentence-transformers/paraphrase-multilingual-mpnet-base-v2', help='임베딩 모델 이름')
    parser.add_argument('--index', type=str, default='insurance-rag', help='Pinecone 인덱스 이름')
    parser.add_argument('--top-k', type=int, default=5, help='검색 결과 수')
    parser.add_argument('--embedding_path', type=str, help='로컬 임베딩 파일 경로')
    
    args = parser.parse_args()
    
    # 설정
    config = {
        'embedding_model': args.model,
        'pinecone_index': args.index,
        'top_k': args.top_k,
        'embedding_path': args.embedding_path
    }
    
    # 평가 도구 초기화
    evaluator = RAGEvaluator(config)
    
    # 질문 데이터셋 로드
    questions = evaluator.load_questions(args.dataset)
    
    if not questions:
        logger.error("질문 로딩 실패. 종료합니다.")
        return
    
    # 평가 실행
    results = evaluator.evaluate_dataset(questions)
    
    # 결과 저장
    evaluator.save_results(results, args.output)
    
    # 결과 요약 출력
    print("\n=== RAG 시스템 평가 결과 요약 ===")
    print(f"전체 질문 수: {results['total']['questions']}")
    print(f"정밀도: {results['total']['precision']:.4f}")
    print(f"재현율: {results['total']['recall']:.4f}")
    print(f"F1 점수: {results['total']['f1_score']:.4f}")
    print(f"NDCG: {results['total']['ndcg']:.4f}")
    print(f"답변 포함율: {results['total']['has_answer_rate']:.4f}")
    print(f"평균 검색 시간: {results['total']['avg_retrieval_time_ms']:.2f} ms")
    
    # 유형별 결과
    print("\n유형별 성능:")
    for q_type, stats in results['by_type'].items():
        print(f"- {q_type} (질문 수: {stats['count']})")
        print(f"  F1 점수: {stats['f1_score']:.4f}, 답변 포함율: {stats['has_answer']:.4f}")
    
    # 난이도별 결과
    print("\n난이도별 성능:")
    for difficulty, stats in results['by_difficulty'].items():
        print(f"- {difficulty} (질문 수: {stats['count']})")
        print(f"  F1 점수: {stats['f1_score']:.4f}, 답변 포함율: {stats['has_answer']:.4f}")


if __name__ == "__main__":
    main()
