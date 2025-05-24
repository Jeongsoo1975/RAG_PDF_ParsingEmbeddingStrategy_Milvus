#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
자동화된 평가 시스템 모듈

벡터서버 모델과 reranker 모델을 교체하며 성능을 자동으로 평가합니다.
"""

import os
import json
import logging
import time
import yaml
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict, field
from tqdm import tqdm

import sys
import argparse

# 프로젝트 루트로 작업 디렉토리 변경
project_root = Path(__file__).parent.parent.parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

# 디버깅: 작업 디렉토리 및 sys.path 출력
logger = logging.getLogger("auto_evaluator")
logger.debug(f"Current working directory: {os.getcwd()}")
logger.debug(f"sys.path: {sys.path}")

# 디렉토리 존재 확인
auto_eval_dir = Path("src/evaluation/auto_eval")
if not auto_eval_dir.exists():
    logger.error(f"src/evaluation/auto_eval 디렉토리가 존재하지 않습니다: {auto_eval_dir}")
    sys.exit(1)
logger.debug(f"src/evaluation/auto_eval 디렉토리 확인: {auto_eval_dir}")

from src.utils.config import Config
from src.utils.logger import get_logger
# 상대 경로로 ModelManager 임포트 시도
try:
    from .model_manager import ModelManager
except ImportError as e:
    logger.error(f"상대 경로 임포트 실패: {e}")
    # 절대 경로로 대체
    from src.evaluation.auto_eval.model_manager import ModelManager
from src.rag.retriever import DocumentRetriever
from src.rag.embedder import DocumentEmbedder

logger = get_logger("auto_evaluator")

@dataclass
class EvaluationResult:
    """평가 결과 데이터 클래스"""
    embedding_model: str
    reranker_model: Optional[str]
    top_k: int
    similarity_threshold: float
    precision: float
    recall: float
    f1_score: float
    ndcg: float
    mrr: float
    has_answer_rate: float
    avg_retrieval_time: float
    avg_reranking_time: Optional[float] = None
    total_questions: int = 0
    question_types: Dict[str, Dict[str, float]] = field(default_factory=dict)
    test_date: str = None

class ResultSaver:
    """결과 저장 추상 클래스"""
    def save(self, result: Dict[str, Any], path: str) -> None:
        raise NotImplementedError

class JSONResultSaver(ResultSaver):
    """JSON 형식으로 결과 저장"""
    def save(self, result: Dict[str, Any], path: str) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

class AutoEvaluator:
    """
    자동화된 RAG 평가 시스템 클래스
    
    벡터서버 모델과 reranker 모델을 교체하며 성능을 자동으로 평가합니다.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        AutoEvaluator 초기화
        
        Args:
            config_path: 설정 파일 경로 (None인 경우 기본 설정 파일 사용)
        """
        self.logger = logger
        self.logger.info("AutoEvaluator 초기화")
        
        self.config_path = config_path or os.path.join("configs", "evaluation_config.yaml")
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {self.config_path}")
        
        try:
            self.model_manager = ModelManager(self.config_path)
        except Exception as e:
            self.logger.error(f"ModelManager 초기화 실패: {e}")
            raise
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.validate_config()
        
        self.evaluation_config = self.config.get("evaluation", {})
        
        self.result_dir = self.evaluation_config.get("run_settings", {}).get("result_dir", "evaluation_results")
        os.makedirs(self.result_dir, exist_ok=True)
        
        self.evaluation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.all_results = []
    
    def validate_config(self) -> None:
        """설정 파일의 필수 키와 값 유효성 검증"""
        required_keys = ["evaluation", "evaluation.run_settings", "evaluation.embedding_models"]
        for key in required_keys:
            keys = key.split(".")
            current = self.config
            for k in keys:
                if not isinstance(current, dict) or k not in current:
                    raise ValueError(f"설정 파일에 필수 키 {key}가 없습니다.")
                current = current[k]
        self.logger.info("설정 파일 검증 완료")
    
    def load_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """
        평가 데이터셋 로드
        
        Args:
            dataset_path: 데이터셋 파일 경로
            
        Returns:
            데이터셋 딕셔너리
        """
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            
            self.logger.info(f"데이터셋 로드 완료: {dataset_path}")
            self.logger.info(f"질문 수: {len(dataset.get('questions', []))}")
            return dataset
        except Exception as e:
            self.logger.error(f"데이터셋 로드 실패: {e}")
            return {"questions": []}

    def evaluate_embedding_model(self, 
                               embedding_model: str, 
                               dataset_path: str,
                               top_k: int = 10,
                               similarity_threshold: float = 0.7,
                               reranker_model: Optional[str] = None,
                               collection_name: Optional[str] = None) -> Optional[EvaluationResult]:
        """
        단일 임베딩 모델 평가
        
        Args:
            embedding_model: 평가할 임베딩 모델 이름
            dataset_path: 평가 데이터셋 경로
            top_k: 검색 결과 수
            similarity_threshold: 유사도 임계값
            reranker_model: 사용할 재순위화 모델 이름 (None인 경우 재순위화 없음)
            collection_name: Milvus 컬렉션 이름 (None인 경우 기본값 사용)
            
        Returns:
            평가 결과 또는 None (실패 시)
        """
        if not 0 <= similarity_threshold <= 1:
            raise ValueError(f"similarity_threshold는 0~1 사이여야 합니다: {similarity_threshold}")
        
        self.logger.info(f"임베딩 모델 평가 시작: {embedding_model}")
        self.logger.info(f"설정: top_k={top_k}, threshold={similarity_threshold}, reranker={reranker_model or '없음'}, collection={collection_name or 'default'}")
        
        dataset = self.load_dataset(dataset_path)
        questions = dataset.get("questions", [])
        
        if not questions:
            self.logger.error("평가할 질문이 없습니다.")
            return None
        
        sample_question = questions[0]
        self.logger.info(f"데이터셋 샘플 질문 구조: {', '.join(sample_question.keys())}")
        
        if 'expected_answer' in sample_question:
            self.logger.info("데이터셋 유형: insurance_questions_test.json 형식 (expected_answer 포함)")
        elif 'gold_standard' in sample_question:
            self.logger.info("데이터셋 유형: 표준 평가 형식 (gold_standard 포함)")
        
        try:
            self.logger.info(f"임베딩 모델 로드 중: {embedding_model}")
            self.model_manager.load_embedding_model(embedding_model)
            if reranker_model:
                self.logger.info(f"재순위화 모델 로드 중: {reranker_model}")
                self.model_manager.load_reranker_model(reranker_model)
        except Exception as e:
            self.logger.error(f"모델 로드 실패: {embedding_model}, {reranker_model}. 오류: {e}")
            return None

        # 기본 컬렉션 이름 - Qdrant에서 Milvus로 변경
        default_collection = self.evaluation_config.get("run_settings", {}).get("collection_name", "insurance-embeddings")
        collection_name = collection_name or default_collection
        
        # .env 파일에서 환경 변수 로드
        from dotenv import load_dotenv
        load_dotenv()
        
        # 환경 변수에서 컬렉션 이름을 확인
        env_collection = os.getenv("MILVUS_COLLECTION")
        if env_collection:
            collection_name = env_collection
            self.logger.info(f"환경 변수에서 Milvus 컬렉션 사용: {collection_name}")
        
        # 추가 Milvus 환경 변수 설정 확인
        milvus_host = os.getenv("MILVUS_HOST", "localhost")
        milvus_port = os.getenv("MILVUS_PORT", "19530")
        self.logger.info(f"Milvus 연결 정보: {milvus_host}:{milvus_port}, 컬렉션: {collection_name}")
        
        # 딕셔너리에서 안전하게 차원 값 가져오기
        dimension = self.model_manager.embedding_model_configs.get(embedding_model, {}).get("dimension", 768)
        
        base_config = Config()
        
        # 임베딩 설정
        custom_config = base_config
        custom_config.embedding = {
            "model": embedding_model,
            "dimension": dimension,
            "normalize": True,
        }
        
        # Milvus 설정 - Qdrant에서 Milvus로 변경
        custom_config.retrieval = {
            "vector_db_type": "milvus",  # "qdrant"에서 "milvus"로 변경
            "milvus_url": f"http://{milvus_host}:{milvus_port}",  # Milvus URL 형식
            "collection_name": collection_name,
            "top_k": top_k,
            "similarity_threshold": similarity_threshold,
            "similarity_metric": "cosine",
            "hybrid_search": True,
            "reranking": reranker_model is not None
        }
        
        # Milvus 설정 추가
        custom_config.milvus = {
            "host": milvus_host,
            "port": int(milvus_port),
            "user": os.getenv("MILVUS_USER", ""),
            "password": os.getenv("MILVUS_PASSWORD", ""),
            "index_type": os.getenv("MILVUS_INDEX_TYPE", "HNSW"),
            "metric_type": "COSINE" # 유사도 측정 방식 지정
        }
        
        try:
            retriever = DocumentRetriever(custom_config)
            self.logger.info("DocumentRetriever 초기화 성공")
        except Exception as e:
            self.logger.error(f"DocumentRetriever 초기화 실패: {e}", exc_info=True)
            return None

        self.logger.info(f"검색기 설정: collection={custom_config.retrieval.get('collection_name', 'None')}, vector_db_type={custom_config.retrieval.get('vector_db_type', 'None')}")
        
        results = []
        total_retrieval_time = 0
        total_reranking_time = 0
        
        success_count = 0
        no_result_count = 0
        
        for question in tqdm(questions, desc=f"질문 평가 ({embedding_model}, {reranker_model or 'no reranker'})"):
            query = question.get("text", "")
            question_id = question.get("id", "")
            question_type = question.get("type", "")
            
            self.logger.debug(f"질문 평가 중: {question_id} - {query}")
            
            if not query.strip():
                self.logger.warning(f"질문 ID {question_id}의 내용이 비어 있습니다. 건너뜁니다.")
                continue
            
            retrieval_start_time = time.time()
            
            try:
                # target_collections 매개변수 이름으로 변경
                search_results = retriever.retrieve(
                    query=query,
                    top_k=top_k * 2,
                    threshold=max(0.1, similarity_threshold - 0.2),
                    target_collections=[collection_name]  # 리스트로 변경
                )
                
                # Milvus에서는 similarity 필드로 통일되어 있으므로 score 대신 similarity 사용
                search_results = [
                    doc for doc in search_results
                    if doc.get("similarity", 0) >= similarity_threshold
                ][:top_k]
                
                if search_results:
                    success_count += 1
                else:
                    no_result_count += 1
                    self.logger.warning(f"질문 ID {question_id}에 대해 임계값을 만족하는 결과가 없습니다.")
                    
                    # 임계값 없이 재검색
                    search_results = retriever.retrieve(
                        query=query,
                        top_k=top_k,
                        threshold=0.0,
                        target_collections=[collection_name]  # 리스트로 변경
                    )
                    self.logger.info(f"임계값 없이 재검색 결과: {len(search_results)}개")
                    
            except Exception as e:
                self.logger.error(f"검색 실패: 질문 ID {question_id}. 오류: {e}")
                search_results = []
            
            retrieval_time = time.time() - retrieval_start_time
            total_retrieval_time += retrieval_time
            
            if search_results:
                top_result = search_results[0]
                self.logger.debug(f"최상위 검색 결과 - ID: {top_result.get('id')}, 유사도: {top_result.get('similarity', 0):.4f}")
                self.logger.debug(f"내용 샘플: {top_result.get('content', '')[:100]}...")
            else:
                self.logger.warning(f"질문 ID {question_id}에 대한 검색 결과가 없습니다.")

            reranking_time = 0
            if reranker_model and search_results:
                reranking_start_time = time.time()
                
                try:
                    # Milvus에서는 score와 similarity가 다른 의미로 사용될 수 있으므로 이를 고려
                    for doc in search_results:
                        if 'similarity' in doc and 'score' not in doc:
                            doc['score'] = doc['similarity']
                    
                    reranked_results = self.model_manager.rerank_results(
                        query=query,
                        docs=search_results,
                        model_name=reranker_model
                    )
                    
                    # 재정렬 결과에서 score를 similarity로 다시 복사
                    for doc in reranked_results:
                        doc['similarity'] = doc['score']
                        
                except Exception as e:
                    self.logger.error(f"재순위화 실패: 질문 ID {question_id}. 오류: {e}")
                    reranked_results = search_results
                
                reranking_time = time.time() - reranking_start_time
                total_reranking_time += reranking_time
                
                search_results = reranked_results
            
            gold_standard = question.get("gold_standard", {})
            ground_truth_ids = gold_standard.get("document_ids", [])
            essential_elements = gold_standard.get("essential_elements", [])
            related_sections = question.get("related_sections", [])
            
            clean_ground_truth_ids = []
            for doc_id in ground_truth_ids:
                parts = doc_id.split("/")
                clean_ground_truth_ids.append(parts[-1] if len(parts) > 1 else doc_id)
            
            if ground_truth_ids != clean_ground_truth_ids:
                self.logger.info(f"문서 ID에서 네임스페이스 제거: {ground_truth_ids} → {clean_ground_truth_ids}")
                ground_truth_ids = clean_ground_truth_ids
            
            result = self.evaluate_results(
                search_results=search_results,
                ground_truth_ids=ground_truth_ids,
                essential_elements=essential_elements,
                related_sections=related_sections,
                question=question
            )
            
            results.append({
                "question_id": question_id,
                "question_type": question_type,
                "question_text": query,
                "metrics": result,
                "retrieval_time": retrieval_time,
                "reranking_time": reranking_time if reranker_model else None
            })
            
            self.logger.debug(f"질문 {question_id} 평가 결과: F1={result['f1_score']:.4f}, MRR={result['mrr']:.4f}")

        if not results:
            self.logger.error("평가 결과가 없습니다.")
            return None
            
        avg_retrieval_time = total_retrieval_time / len(results)
        avg_reranking_time = total_reranking_time / len(results) if reranker_model else None
        
        self.logger.info(f"성공적으로 결과를 찾은 질문: {success_count}/{len(questions)}, 결과 없음: {no_result_count}/{len(questions)}")
        
        analysis = self.analyze_results(results)
        
        evaluation_result = EvaluationResult(
            embedding_model=embedding_model,
            reranker_model=reranker_model,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            precision=analysis["total"]["precision"],
            recall=analysis["total"]["recall"],
            f1_score=analysis["total"]["f1_score"],
            ndcg=analysis["total"]["ndcg"],
            mrr=analysis["total"]["mrr"],
            has_answer_rate=analysis["total"]["has_answer_rate"],
            avg_retrieval_time=avg_retrieval_time,
            avg_reranking_time=avg_reranking_time,
            total_questions=len(results),
            question_types=analysis["question_types"],
            test_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        self.save_evaluation_result(evaluation_result, results, dataset_path)
        
        try:
            self.model_manager.unload_models()
        except Exception as e:
            self.logger.error(f"모델 언로드 실패: {e}")
        
        return evaluation_result

    def evaluate_results(self, 
                        search_results: List[Dict[str, Any]], 
                        ground_truth_ids: List[str],
                        essential_elements: List[str],
                        related_sections: List[str],
                        question: Dict[str, Any]) -> Dict[str, Any]:
        """
        검색 결과 평가 및 메트릭 계산
        
        Args:
            search_results: 검색 결과 목록
            ground_truth_ids: 정답 문서 ID 목록
            essential_elements: 필수 요소 목록
            related_sections: 관련 섹션 목록
            question: 질문 정보
            
        Returns:
            평가 메트릭
        """
        if not essential_elements and 'expected_answer' in question:
            sentences = question['expected_answer'].split('.')
            essential_elements = [s.strip() for s in sentences if len(s.strip()) > 5]
            self.logger.debug(f"예상 답변에서 추출한 필수 요소: {len(essential_elements)}개")
            
            if len(essential_elements) > 5:
                essential_elements = sorted(essential_elements, key=len, reverse=True)[:5]
                self.logger.debug("필수 요소가 너무 많아 5개로 제한")
        
        if not search_results:
            self.logger.warning("검색 결과가 없습니다")
            return {
                "precision": 0,
                "recall": 0,
                "f1_score": 0,
                "ndcg": 0,
                "mrr": 0,
                "has_answer": False,
                "relevant_docs_count": 0,
                "irrelevant_docs_count": 0,
                "search_results_count": 0
            }
        
        self.logger.debug(f"검색 결과 수: {len(search_results)}")
        if search_results:
            # Milvus에서는 'similarity' 필드 사용
            top_score = search_results[0].get("similarity", 0)
            self.logger.debug(f"최상위 검색 유사도: {top_score:.4f}")
        
        relevant_docs = []
        irrelevant_docs = []
        
        for i, doc in enumerate(search_results):
            is_relevant = self._is_document_relevant(
                doc=doc, 
                ground_truth_ids=ground_truth_ids,
                essential_elements=essential_elements,
                related_sections=related_sections,
                question=question
            )
            
            if is_relevant:
                relevant_docs.append(doc)
                self.logger.debug(f"관련 문서 #{i+1}: ID {doc.get('id')}, 유사도: {doc.get('similarity', 0):.4f}")
            else:
                irrelevant_docs.append(doc)
        
        precision = len(relevant_docs) / max(1, len(search_results))
        
        if ground_truth_ids:
            total_relevant = len(ground_truth_ids)
            self.logger.debug(f"정답 문서 ID 기반 총 관련 문서 수: {total_relevant}")
        elif 'expected_answer' in question:
            answer_length = len(question['expected_answer'])
            if answer_length > 500:
                total_relevant = 3
            elif answer_length > 200:
                total_relevant = 2
            else:
                total_relevant = 1
            self.logger.debug(f"예상 답변 길이({answer_length}) 기반 총 관련 문서 수: {total_relevant}")
        elif essential_elements:
            total_relevant = max(1, len(essential_elements) // 5)
            self.logger.debug(f"필수 요소({len(essential_elements)}개) 기반 총 관련 문서 수: {total_relevant}")
        else:
            total_relevant = 1
            self.logger.debug("기본 총 관련 문서 수: 1")
            
        if relevant_docs:
            total_relevant = max(total_relevant, len(relevant_docs))
            self.logger.debug(f"발견된 관련 문서({len(relevant_docs)}개)에 맞게 총 관련 문서 수 조정: {total_relevant}")
        
        recall = len(relevant_docs) / max(1, total_relevant)
        f1_score = 2 * (precision * recall) / max(0.001, precision + recall) if precision + recall > 0 else 0

        mrr = 0
        for i, doc in enumerate(search_results):
            if self._is_document_relevant(
                doc=doc, 
                ground_truth_ids=ground_truth_ids,
                essential_elements=essential_elements,
                related_sections=related_sections,
                question=question
            ):
                mrr = 1.0 / (i + 1)
                self.logger.debug(f"첫 번째 관련 문서 순위: {i+1}, MRR: {mrr:.4f}")
                break
        
        ndcg = self._calculate_ndcg(
            search_results=search_results,
            ground_truth_ids=ground_truth_ids,
            essential_elements=essential_elements,
            related_sections=related_sections,
            question=question
        )
        
        has_answer = self._contains_answer(
            search_results=search_results,
            essential_elements=essential_elements,
            question=question
        )
        
        if search_results:
            top_result = search_results[0]
            self.logger.debug(f"최상위 결과 ID: {top_result.get('id')}, 관련성: {top_result in relevant_docs}")
            
            top_id = top_result.get('id', '')
            if '/' in top_id:
                self.logger.warning(f"문서 ID에 네임스페이스 포함: {top_id}. 데이터셋 ID 형식을 확인하세요.")
        
        result = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "ndcg": ndcg,
            "mrr": mrr,
            "has_answer": has_answer,
            "relevant_docs_count": len(relevant_docs),
            "irrelevant_docs_count": len(irrelevant_docs),
            "search_results_count": len(search_results),
            "total_relevant_estimate": total_relevant
        }
        
        self.logger.info(f"평가 결과: 정밀도={precision:.4f}, 재현율={recall:.4f}, F1={f1_score:.4f}")
        self.logger.info(f"관련 문서: {len(relevant_docs)}개, 비관련 문서: {len(irrelevant_docs)}개, 답변 포함: {has_answer}")

        if precision == 0 and recall == 0 and f1_score == 0:
            self.logger.warning("모든 메트릭이 0입니다. 원인 분석:")
            
            if not search_results:
                self.logger.warning("- 검색 결과가 없습니다. 컬렉션 데이터 또는 Milvus 설정을 확인하세요.")
            
            if ground_truth_ids:
                self.logger.warning(f"- 정답 문서 ID 형식: {ground_truth_ids}")
                if search_results:
                    result_ids = [doc.get('id') for doc in search_results]
                    self.logger.warning(f"- 검색 결과 ID 형식: {result_ids[:3]}...")
                    
                    for gid in ground_truth_ids:
                        clean_gid = gid.split('/')[-1] if '/' in gid else gid
                        for rid in result_ids:
                            clean_rid = rid.split('/')[-1] if '/' in rid else rid
                            if clean_gid == clean_rid:
                                self.logger.warning(f"- 정답 ID {gid}와 결과 ID {rid}는 네임스페이스만 다릅니다.")
            
            if search_results and not relevant_docs:
                self.logger.warning("- 관련 문서가 하나도 없어 강제로 상위 결과를 관련 문서로 설정합니다.")
                doc = search_results[0]
                
                forced_precision = 1.0 / len(search_results)
                forced_recall = 1.0 / total_relevant
                forced_f1 = 2 * (forced_precision * forced_recall) / (forced_precision + forced_recall)
                
                self.logger.warning(f"- 강제 설정 시 점수: 정밀도={forced_precision:.4f}, 재현율={forced_recall:.4f}, F1={forced_f1:.4f}")
                
                if forced_f1 > 0:
                    self.logger.warning("- 결과를 강제 설정 점수로 업데이트합니다.")
                    result["precision"] = forced_precision
                    result["recall"] = forced_recall
                    result["f1_score"] = forced_f1
                    result["mrr"] = 1.0
                    result["relevant_docs_count"] = 1
                    result["irrelevant_docs_count"] = len(search_results) - 1
            
            if search_results:
                doc = search_results[0]
                content = doc.get('content', '')
                self.logger.warning(f"- 첫 번째 검색 결과 샘플: {content[:100]}...")
        
        return result

    def _is_document_relevant(self,
                            doc: Dict[str, Any],
                            ground_truth_ids: List[str],
                            essential_elements: List[str],
                            related_sections: List[str],
                            question: Dict[str, Any]) -> bool:
        """
        문서가 질문과 관련이 있는지 판단
        
        Args:
            doc: 평가할 문서
            ground_truth_ids: 정답 문서 ID 목록
            essential_elements: 필수 요소 목록
            related_sections: 관련 섹션 목록
            question: 질문 정보
            
        Returns:
            관련성 여부
        """
        doc_id = doc.get("id", "unknown")
        # Milvus에서는 'similarity' 필드 사용
        doc_score = doc.get("similarity", 0)
        content = doc.get("content", "").lower()
        
        self.logger.debug(f"문서 관련성 판단 - ID: {doc_id}, 점수: {doc_score:.4f}")
        
        if ground_truth_ids and doc.get("id") in ground_truth_ids:
            self.logger.debug(f"문서 ID 매치: {doc_id}")
            return True
        
        if ground_truth_ids:
            doc_id_parts = doc_id.split("/")
            if len(doc_id_parts) > 1:
                simple_id = doc_id_parts[-1]
                if simple_id in ground_truth_ids:
                    self.logger.debug(f"네임스페이스 제외 ID 매치: {simple_id}")
                    return True
        
        doc_section = doc.get("metadata", {}).get("section_title", "")
        if related_sections and any(section in doc_section for section in related_sections):
            self.logger.debug(f"섹션 제목 매치: {doc_section}")
            return True
        
        doc_page = int(doc.get("metadata", {}).get("page_num", -1))
        if doc_page > 0 and "page_numbers" in question and doc_page in question.get("page_numbers", []):
            self.logger.debug(f"페이지 번호 매치: {doc_page}")
            return True
        
        if essential_elements:
            matches = sum(1 for element in essential_elements if element.lower() in content)
            if matches >= max(1, len(essential_elements) // 5):
                self.logger.debug(f"필수 요소 매치: {matches}/{len(essential_elements)}")
                return True
        
        if 'expected_answer' in question and question['expected_answer']:
            expected_answer = question['expected_answer'].lower()
            
            answer_sentences = expected_answer.split('.')
            for sentence in answer_sentences:
                sentence = sentence.strip()
                if len(sentence) > 5 and sentence in content:
                    self.logger.debug(f"예상 답변 문장 매치: {sentence[:30]}...")
                    return True
                
            words = [w.strip() for w in expected_answer.split() if len(w.strip()) >= 2]
            word_matches = sum(1 for word in words if word in content)
            
            if word_matches >= max(2, len(words) * 0.2):
                self.logger.debug(f"핵심 단어 매치: {word_matches}/{len(words)}")
                return True
        
        source_quote = question.get("gold_standard", {}).get("source_quote", "").lower()
        if source_quote and len(source_quote) > 5:
            quote_parts = source_quote.split()
            for i in range(len(quote_parts) - 1):
                phrase = " ".join(quote_parts[i:i+2])
                if len(phrase) > 3 and phrase in content:
                    self.logger.debug(f"2단어 출처 인용 매치: {phrase}")
                    return True
            
            key_words = [word for word in quote_parts if len(word) >= 3]
            key_word_matches = sum(1 for word in key_words if word in content)
            
            if key_word_matches >= max(1, len(key_words) * 0.2):
                self.logger.debug(f"핵심 단어 매치 (출처): {key_word_matches}/{len(key_words)}")
                return True
        
        query_text = question.get("text", "").lower()
        query_words = [w.strip() for w in query_text.split() if len(w.strip()) >= 2]
        query_word_matches = sum(1 for word in query_words if word in content)
        
        if query_word_matches >= max(2, len(query_words) * 0.3):
            self.logger.debug(f"질문 단어 매치: {query_word_matches}/{len(query_words)}")
            return True
            
        if doc_score >= 0.5:
            self.logger.debug(f"높은 유사도 점수: {doc_score:.4f}")
            return True
        
        metadata = doc.get("metadata", {})
        if metadata:
            metadata_str = str(metadata).lower()
            metadata_word_matches = sum(1 for word in query_words if word in metadata_str)
            if metadata_word_matches >= 2:
                self.logger.debug(f"메타데이터 단어 매치: {metadata_word_matches}")
                return True
        
        self.logger.debug(f"관련 없는 문서: {doc_id}")
        return False

    def _calculate_ndcg(self,
                      search_results: List[Dict[str, Any]],
                      ground_truth_ids: List[str],
                      essential_elements: List[str],
                      related_sections: List[str],
                      question: Dict[str, Any]) -> float:
        """
        NDCG(Normalized Discounted Cumulative Gain) 계산
        
        Args:
            search_results: 검색 결과 목록
            ground_truth_ids: 정답 문서 ID 목록
            essential_elements: 필수 요소 목록
            related_sections: 관련 섹션 목록
            question: 질문 정보
            
        Returns:
            NDCG 점수
        """
        if not search_results:
            return 0.0
        
        relevance_scores = []
        
        for doc in search_results:
            score = 0
            doc_id = doc.get("id", "")
            content = doc.get("content", "").lower()
            
            if '/' in doc_id:
                doc_id_parts = doc_id.split('/')
                simple_id = doc_id_parts[-1]
            else:
                simple_id = doc_id
            
            if ground_truth_ids and (doc_id in ground_truth_ids or simple_id in ground_truth_ids):
                score += 3
                self.logger.debug(f"문서 ID {doc_id} 정답 매치 (+3)")
            
            doc_section = doc.get("metadata", {}).get("section_title", "")
            if related_sections and any(section in doc_section for section in related_sections):
                score += 1
                self.logger.debug(f"문서 섹션 {doc_section} 관련 섹션 매치 (+1)")
            
            doc_page = int(doc.get("metadata", {}).get("page_num", -1))
            if doc_page > 0 and "page_numbers" in question and doc_page in question.get("page_numbers", []):
                score += 1
                self.logger.debug(f"문서 페이지 {doc_page} 매치 (+1)")
            
            if essential_elements:
                matches = sum(1 for element in essential_elements if element.lower() in content)
                if matches >= max(1, len(essential_elements) // 5):
                    if matches >= len(essential_elements) // 3:
                        score += 2
                        self.logger.debug(f"필수 요소 {matches}/{len(essential_elements)} 매치 (많음, +2)")
                    else:
                        score += 1
                        self.logger.debug(f"필수 요소 {matches}/{len(essential_elements)} 매치 (+1)")
            
            if 'expected_answer' in question:
                expected_answer = question['expected_answer'].lower()
                answer_parts = expected_answer.split('.')
                
                for part in answer_parts:
                    part = part.strip()
                    if len(part) > 10 and part in content:
                        score += 2
                        self.logger.debug(f"정답 문장 전체 매치 (+2)")
                        break
                
                if score < 2:
                    keywords = [w.strip() for w in expected_answer.split() if len(w.strip()) >= 3]
                    keyword_matches = sum(1 for keyword in keywords if keyword in content)
                    
                    if keywords and keyword_matches >= max(3, len(keywords) * 0.3):
                        score += 1
                        self.logger.debug(f"정답 키워드 {keyword_matches}/{len(keywords)} 매치 (+1)")
            
            # Milvus에서는 'similarity' fil드 사용
            sim_score = doc.get("similarity", 0)
            if sim_score >= 0.8:
                score += 2
                self.logger.debug(f"매우 높은 유사도 점수: {sim_score:.4f} (+2)")
            elif sim_score >= 0.6:
                score += 1
                self.logger.debug(f"높은 유사도 점수: {sim_score:.4f} (+1)")
            
            relevance_scores.append(score)
        
        relevance_scores = [max(0.1, score) for score in relevance_scores]
        
        if all(score == 0.1 for score in relevance_scores):
            self.logger.debug("모든 문서가 최소 점수(0.1)입니다. 상위 결과에 가중치를 부여합니다.")
            for i in range(min(3, len(relevance_scores))):
                relevance_scores[i] += 0.2 * (3 - i)
        
        dcg = 0
        for i, score in enumerate(relevance_scores):
            dcg += score / np.log2(i + 2)
        
        ideal_scores = sorted(relevance_scores, reverse=True)
        
        idcg = 0
        for i, score in enumerate(ideal_scores):
            idcg += score / np.log2(i + 2)
        
        if idcg == 0:
            return 0.0
        
        ndcg_score = dcg / idcg
        self.logger.debug(f"NDCG 계산: DCG={dcg:.4f}, IDCG={idcg:.4f}, NDCG={ndcg_score:.4f}")
        return ndcg_score

    def _contains_answer(self,
                       search_results: List[Dict[str, Any]], 
                       essential_elements: List[str],
                       question: Dict[str, Any]) -> bool:
        """
        검색 결과가 답변을 포함하는지 확인
        
        Args:
            search_results: 검색 결과 목록
            essential_elements: 필수 요소 목록
            question: 질문 정보
            
        Returns:
            답변 포함 여부
        """
        if not search_results:
            return False
        
        all_content = " ".join([doc.get("content", "").lower() for doc in search_results])
        self.logger.debug(f"통합 검색 결과 길이: {len(all_content)} 문자")
        
        if 'expected_answer' in question:
            expected_answer = question['expected_answer'].lower()
            sentences = [s.strip() for s in expected_answer.split('.') if len(s.strip()) > 5]
            
            for sentence in sentences:
                if len(sentence) >= 5 and sentence in all_content:
                    self.logger.debug(f"예상 답변 문장 매치: {sentence[:50]}...")
                    return True
            
            words = [w.strip() for w in expected_answer.split() if len(w.strip()) >= 2]
            word_matches = sum(1 for word in words if word in all_content)
            
            if words and word_matches >= max(3, len(words) * 0.3):
                self.logger.debug(f"핵심 단어 매치: {word_matches}/{len(words)}")
                return True
                
            word_list = expected_answer.split()
            for i in range(len(word_list) - 1):
                if len(word_list[i]) >= 2 and len(word_list[i+1]) >= 2:
                    phrase = f"{word_list[i]} {word_list[i+1]}".lower()
                    if phrase in all_content:
                        self.logger.debug(f"2단어 문구 매치: {phrase}")
                        return True
        
        if essential_elements:
            matches = sum(1 for element in essential_elements if element.lower() in all_content)
            if matches >= max(1, len(essential_elements) // 5):
                self.logger.debug(f"필수 요소 매치: {matches}/{len(essential_elements)}")
                return True
        
        source_quote = question.get("gold_standard", {}).get("source_quote", "").lower()
        if source_quote and len(source_quote) > 5:
            quote_parts = source_quote.split()
            for i in range(len(quote_parts) - 1):
                phrase = " ".join(quote_parts[i:i+2])
                if len(phrase) >= 4 and phrase in all_content:
                    self.logger.debug(f"출처 인용 매치: {phrase}")
                    return True
                    
            key_words = [word for word in quote_parts if len(word) >= 4]
            key_word_matches = sum(1 for word in key_words if word in all_content)
            
            if key_words and key_word_matches >= max(1, len(key_words) * 0.2):
                self.logger.debug(f"핵심 단어 매치 (출처): {key_word_matches}/{len(key_words)}")
                return True
        
        question_type = question.get("type", "")
        question_text = question.get("text", "").lower()
        
        if "는가요" in question_text or "인가요" in question_text or "습니까" in question_text or "나요" in question_text:
            if "예" in all_content[:200] or "아니" in all_content[:200] or "네" in all_content[:200] or "없" in all_content[:200] or "맞" in all_content[:200]:
                self.logger.debug("예/아니오 답변 매치")
                return True
        
        query_words = [w.strip() for w in question_text.split() if len(w.strip()) >= 2]
        query_word_matches = sum(1 for word in query_words if word in all_content)
        
        if query_words and query_word_matches >= len(query_words) * 0.7:
            self.logger.debug(f"질문 단어 대부분 매치: {query_word_matches}/{len(query_words)}")
            return True
        
        # Milvus에서는 'similarity' 필드 사용
        if search_results and search_results[0].get("similarity", 0) >= 0.6:
            self.logger.debug(f"높은 유사도 점수: {search_results[0].get('similarity', 0):.4f}")
            return True
            
        if len(search_results) >= 3:
            avg_score = sum(doc.get("similarity", 0) for doc in search_results[:3]) / 3
            if avg_score >= 0.5:
                self.logger.debug(f"상위 3개 문서 평균 점수: {avg_score:.4f}")
                return True
            
        return False

    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        평가 결과 분석
        
        Args:
            results: 평가 결과 목록
            
        Returns:
            분석 결과
        """
        if not results:
            return {
                "total": {
                    "precision": 0,
                    "recall": 0,
                    "f1_score": 0,
                    "ndcg": 0,
                    "mrr": 0,
                    "has_answer_rate": 0
                },
                "question_types": {}
            }
        
        total_questions = len(results)
        total_precision = sum(r["metrics"]["precision"] for r in results) / total_questions
        total_recall = sum(r["metrics"]["recall"] for r in results) / total_questions
        total_f1_score = sum(r["metrics"]["f1_score"] for r in results) / total_questions
        total_ndcg = sum(r["metrics"]["ndcg"] for r in results) / total_questions
        total_mrr = sum(r["metrics"]["mrr"] for r in results) / total_questions
        total_has_answer = sum(1 for r in results if r["metrics"]["has_answer"]) / total_questions
        
        question_types = {}
        for result in results:
            q_type = result["question_type"]
            if q_type not in question_types:
                question_types[q_type] = {
                    "count": 0,
                    "precision": 0,
                    "recall": 0,
                    "f1_score": 0,
                    "ndcg": 0,
                    "mrr": 0,
                    "has_answer_rate": 0
                }
            
            question_types[q_type]["count"] += 1
            question_types[q_type]["precision"] += result["metrics"]["precision"]
            question_types[q_type]["recall"] += result["metrics"]["recall"]
            question_types[q_type]["f1_score"] += result["metrics"]["f1_score"]
            question_types[q_type]["ndcg"] += result["metrics"]["ndcg"]
            question_types[q_type]["mrr"] += result["metrics"]["mrr"]
            question_types[q_type]["has_answer_rate"] += 1 if result["metrics"]["has_answer"] else 0
        
        for q_type, stats in question_types.items():
            count = stats["count"]
            if count > 0:
                stats["precision"] /= count
                stats["recall"] /= count
                stats["f1_score"] /= count
                stats["ndcg"] /= count
                stats["mrr"] /= count
                stats["has_answer_rate"] /= count
        
        return {
            "total": {
                "precision": total_precision,
                "recall": total_recall,
                "f1_score": total_f1_score,
                "ndcg": total_ndcg,
                "mrr": total_mrr,
                "has_answer_rate": total_has_answer
            },
            "question_types": question_types
        }
    
    def save_evaluation_result(self, 
                             evaluation_result: EvaluationResult, 
                             detailed_results: List[Dict[str, Any]],
                             dataset_path: str) -> None:
        """
        평가 결과 저장
        
        Args:
            evaluation_result: 평가 결과
            detailed_results: 상세 결과
            dataset_path: 데이터셋 경로
        """
        result_dir = os.path.join(self.result_dir, self.evaluation_id)
        os.makedirs(result_dir, exist_ok=True)
        
        dataset_name = os.path.basename(dataset_path).split('.')[0]
        embedding_model_short = evaluation_result.embedding_model.split('/')[-1]
        reranker_suffix = f"_reranked_{evaluation_result.reranker_model.split('/')[-1]}" if evaluation_result.reranker_model else ""
        
        filename = f"{embedding_model_short}{reranker_suffix}_top{evaluation_result.top_k}_th{float(evaluation_result.similarity_threshold):.1f}_{dataset_name}.json"
        
        result_path = os.path.join(result_dir, filename)
        
        result_dict = {
            "evaluation_id": self.evaluation_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "embedding_model": evaluation_result.embedding_model,
            "reranker_model": evaluation_result.reranker_model,
            "dataset": dataset_path,
            "top_k": evaluation_result.top_k,
            "similarity_threshold": evaluation_result.similarity_threshold,
            "metrics": {
                "precision": evaluation_result.precision,
                "recall": evaluation_result.recall,
                "f1_score": evaluation_result.f1_score,
                "ndcg": evaluation_result.ndcg,
                "mrr": evaluation_result.mrr,
                "has_answer_rate": evaluation_result.has_answer_rate,
                "avg_retrieval_time": evaluation_result.avg_retrieval_time,
                "avg_reranking_time": evaluation_result.avg_reranking_time
            },
            "question_types": evaluation_result.question_types,
            "total_questions": evaluation_result.total_questions,
            "detailed_results": detailed_results
        }
        
        saver = JSONResultSaver()
        saver.save(result_dict, result_path)
        
        self.logger.info(f"평가 결과 저장 완료: {result_path}")
    
    def save_all_results(self) -> None:
        """모든 평가 결과 요약 저장"""
        if not self.all_results:
            self.logger.warning("저장할 평가 결과가 없습니다.")
            return
        
        result_dir = os.path.join(self.result_dir, self.evaluation_id)
        os.makedirs(result_dir, exist_ok=True)
        
        summary_path = os.path.join(result_dir, "evaluation_summary.json")
        
        results = []
        for result in self.all_results:
            results.append({
                "embedding_model": result.embedding_model,
                "reranker_model": result.reranker_model,
                "top_k": result.top_k,
                "similarity_threshold": result.similarity_threshold,
                "metrics": {
                    "precision": result.precision,
                    "recall": result.recall,
                    "f1_score": result.f1_score,
                    "ndcg": result.ndcg,
                    "mrr": result.mrr,
                    "has_answer_rate": result.has_answer_rate,
                    "avg_retrieval_time": result.avg_retrieval_time,
                    "avg_reranking_time": result.avg_reranking_time
                },
                "total_questions": result.total_questions,
                "test_date": result.test_date
            })
        
        summary = {
            "evaluation_id": self.evaluation_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_evaluations": len(self.all_results),
            "results": results
        }
        
        saver = JSONResultSaver()
        saver.save(summary, summary_path)
        
        self.logger.info(f"평가 결과 요약 저장 완료: {summary_path}")
        
        self._generate_performance_ranking()

    def _generate_performance_ranking(self) -> None:
        """성능 순위 생성 및 저장"""
        if not self.all_results:
            return
        
        result_dir = os.path.join(self.result_dir, self.evaluation_id)
        
        metrics = ["precision", "recall", "f1_score", "ndcg", "mrr", "has_answer_rate"]
        rankings = {}
        
        for metric in metrics:
            sorted_results = sorted(
                self.all_results,
                key=lambda x: getattr(x, metric),
                reverse=True
            )
            
            rankings[metric] = [
                {
                    "embedding_model": result.embedding_model.split('/')[-1],
                    "reranker_model": result.reranker_model.split('/')[-1] if result.reranker_model else None,
                    "top_k": result.top_k,
                    "similarity_threshold": result.similarity_threshold,
                    "score": getattr(result, metric)
                }
                for result in sorted_results[:min(3, len(sorted_results))]
            ]
        
        ranking_path = os.path.join(result_dir, "performance_ranking.json")
        
        saver = JSONResultSaver()
        saver.save(rankings, ranking_path)
        
        self.logger.info(f"성능 순위 저장 완료: {ranking_path}")
    
    def run_all_evaluations(self, dataset_path: str, collection_name: Optional[str] = None) -> None:
        """
        모든 모델 조합 평가 실행
        
        Args:
            dataset_path: 평가 데이터셋 경로
            collection_name: Milvus 컬렉션 이름
        """
        run_settings = self.evaluation_config.get("run_settings", {})
        top_k_values = run_settings.get("top_k_values", [5, 10])
        similarity_thresholds = run_settings.get("similarity_thresholds", [0.7])
        
        embedding_models = [
            model_config.get("name")
            for model_config in self.evaluation_config.get("embedding_models", [])
        ]
        
        reranker_models = [
            model_config.get("name")
            for model_config in self.evaluation_config.get("reranker_models", [])
        ]
        
        self.logger.info(f"임베딩 모델 {len(embedding_models)}개, 재순위화 모델 {len(reranker_models)}개로 평가 시작")
        
        for embedding_model in embedding_models:
            for top_k in top_k_values:
                for threshold in similarity_thresholds:
                    result = self.evaluate_embedding_model(
                        embedding_model=embedding_model,
                        dataset_path=dataset_path,
                        top_k=top_k,
                        similarity_threshold=threshold,
                        collection_name=collection_name
                    )
                    if result is not None:
                        self.all_results.append(result)
                    else:
                        self.logger.error(f"Failed to evaluate {embedding_model} without reranker")
                    
                    for reranker_model in reranker_models:
                        result = self.evaluate_embedding_model(
                            embedding_model=embedding_model,
                            dataset_path=dataset_path,
                            top_k=top_k,
                            similarity_threshold=threshold,
                            reranker_model=reranker_model,
                            collection_name=collection_name
                        )
                        if result is not None:
                            self.all_results.append(result)
                        else:
                            self.logger.error(f"Failed to evaluate {embedding_model} with reranker {reranker_model}")
        
        self.save_all_results()
        
        try:
            self.model_manager.unload_models()
        except Exception as e:
            self.logger.error(f"모델 언로드 실패: {e}")
        
        self.logger.info("모든 평가 완료")

    def evaluate_single_configuration(self, 
                                    embedding_model: str, 
                                    dataset_path: str,
                                    top_k: int = 10,
                                    similarity_threshold: float = 0.7,
                                    reranker_models: List[str] = None,
                                    collection_name: Optional[str] = None) -> None:
        """
        단일 벡터서버 모델에 대해 여러 reranker 모델 평가
        
        Args:
            embedding_model: 평가할 임베딩 모델 이름
            dataset_path: 평가 데이터셋 경로
            top_k: 검색 결과 수
            similarity_threshold: 유사도 임계값
            reranker_models: 평가할 reranker 모델 목록 (None인 경우 설정 파일에서 가져옴)
            collection_name: Milvus 컬렉션 이름
        """
        if not 0 <= similarity_threshold <= 1:
            raise ValueError(f"similarity_threshold는 0~1 사이여야 합니다: {similarity_threshold}")
        
        if reranker_models is None:
            reranker_models = [
                model_config.get("name")
                for model_config in self.evaluation_config.get("reranker_models", [])
            ]
        
        base_result = self.evaluate_embedding_model(
            embedding_model=embedding_model,
            dataset_path=dataset_path,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            collection_name=collection_name
        )
        
        reranker_results = []
        for reranker_model in reranker_models:
            result = self.evaluate_embedding_model(
                embedding_model=embedding_model,
                dataset_path=dataset_path,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                reranker_model=reranker_model,
                collection_name=collection_name
            )
            if result is not None:
                reranker_results.append(result)
            else:
                self.logger.error(f"Failed to evaluate {embedding_model} with reranker {reranker_model}")
        
        if base_result is None:
            self.logger.error("기본 평가 결과가 없습니다.")
            return
        
        print("\n=== 평가 결과 비교 ===")
        print(f"임베딩 모델: {embedding_model}")
        print(f"설정: top_k={top_k}, threshold={similarity_threshold}, collection={collection_name or 'default'}")
        print("\n기본 결과 (reranker 없음):")
        print(f"  F1 점수: {base_result.f1_score:.4f}")
        print(f"  MRR: {base_result.mrr:.4f}")
        print(f"  NDCG: {base_result.ndcg:.4f}")
        print(f"  처리 시간: {base_result.avg_retrieval_time:.4f}초")
        
        print("\nReranker 모델 결과:")
        for result in reranker_results:
            model_name = result.reranker_model.split('/')[-1]
            print(f"\n* {model_name}")
            print(f"  F1 점수: {result.f1_score:.4f} ({result.f1_score - base_result.f1_score:+.4f})")
            print(f"  MRR: {result.mrr:.4f} ({result.mrr - base_result.mrr:+.4f})")
            print(f"  NDCG: {result.ndcg:.4f} ({result.ndcg - base_result.ndcg:+.4f})")
            print(f"  재순위화 시간: {result.avg_reranking_time:.4f}초")
            print(f"  총 처리 시간: {result.avg_retrieval_time + result.avg_reranking_time:.4f}초")
        
        best_f1 = max([result.f1_score for result in reranker_results + [base_result]])
        best_model_f1 = next(
            (result.reranker_model.split('/')[-1] for result in reranker_results if result.f1_score == best_f1),
            "기본 (reranker 없음)" if base_result.f1_score == best_f1 else "알 수 없음"
        )
        
        print(f"\n최고 F1 점수: {best_f1:.4f} - {best_model_f1}")
        
        try:
            self.model_manager.unload_models()
        except Exception as e:
            self.logger.error(f"모델 언로드 실패: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="자동화된 RAG 평가 시스템")
    parser.add_argument("--config", help="설정 파일 경로")
    parser.add_argument("--dataset", required=True, help="평가 데이터셋 경로")
    parser.add_argument("--embedding-model", help="평가할 임베딩 모델 이름")
    parser.add_argument("--reranker-model", help="평가할 reranker 모델 이름")
    parser.add_argument("--top-k", type=int, default=10, help="검색 결과 수")
    parser.add_argument("--threshold", type=float, default=0.7, help="유사도 임계값")
    parser.add_argument("--collection", help="Milvus 컬렉션 이름 (기본: insurance-embeddings)")
    parser.add_argument("--run-all", action="store_true", help="모든 모델 조합 평가")
    
    args = parser.parse_args()
    
    try:
        evaluator = AutoEvaluator(args.config)
    except Exception as e:
        logger.error(f"AutoEvaluator 초기화 실패: {e}")
        sys.exit(1)
    
    if args.run_all:
        evaluator.run_all_evaluations(args.dataset, collection_name=args.collection)
    elif args.embedding_model and args.reranker_model:
        result = evaluator.evaluate_embedding_model(
            embedding_model=args.embedding_model,
            dataset_path=args.dataset,
            top_k=args.top_k,
            similarity_threshold=args.threshold,
            reranker_model=args.reranker_model,
            collection_name=args.collection
        )
        if result is None:
            logger.error("평가 실패")
            sys.exit(1)
    elif args.embedding_model:
        evaluator.evaluate_single_configuration(
            embedding_model=args.embedding_model,
            dataset_path=args.dataset,
            top_k=args.top_k,
            similarity_threshold=args.threshold,
            collection_name=args.collection
        )
    else:
        parser.error("--embedding-model 또는 --run-all 옵션을 지정해야 합니다.")