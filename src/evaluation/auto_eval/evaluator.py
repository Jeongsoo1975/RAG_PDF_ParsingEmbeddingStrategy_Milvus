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
try:
    import yaml  # PyYAML 설치 필요: pip install PyYAML
except ImportError:
    print("오류: PyYAML 모듈이 설치되지 않았습니다. 'pip install PyYAML'을 실행하여 설치하세요.")
    import json as yaml  # yaml 깜수만 대체하는 간단한 대체 해결책
from typing import Dict, Any, List, Optional, Union, Tuple, Callable  # Callable 추가
from abc import ABC, abstractmethod
from datetime import datetime
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict, field
try:
    from tqdm import tqdm
except ImportError:
    print("오류: tqdm 모듈이 설치되지 않았습니다. 'pip install tqdm'을 실행하여 설치하세요.")
    # tqdm에 대한 대체 구현
    def tqdm(iterable, *args, **kwargs):
        print(f"... 실행 중 (tqdm 없이 진행률 표시 불가)") 
        return iterable
import argparse  # argparse 임포트 추가
import re
import sys
import uuid

# 프로젝트 루트 설정 (이 파일의 위치에 따라 조정 필요)
try:
    CURRENT_FILE_PATH = Path(__file__).resolve()
    PROJECT_ROOT = CURRENT_FILE_PATH.parent.parent.parent.parent
    if not (PROJECT_ROOT / "src").is_dir() or not (PROJECT_ROOT / "configs").is_dir():
        PROJECT_ROOT = Path.cwd()
        if not (PROJECT_ROOT / "src").is_dir() or not (PROJECT_ROOT / "configs").is_dir():
            logging.warning(f"자동으로 감지된 프로젝트 루트({PROJECT_ROOT})가 올바르지 않을 수 있습니다. src 및 configs 폴더를 확인하세요.")

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    logging.info(f"Project root set to: {PROJECT_ROOT}")
    logging.info(f"Sys.path modified with: {PROJECT_ROOT}")

except Exception as e:
    logging.error(f"Error setting up project root and sys.path: {e}", exc_info=True)

# 필요한 모듈 임포트
try:
    from src.utils.config import Config
    from src.utils.logger import get_logger
    from src.evaluation.auto_eval.model_manager import ModelManager
    from src.rag.retriever import DocumentRetriever
    from src.rag.embedder import DocumentEmbedder
except ImportError as e:
    logging.critical(f"필수 모듈 임포트 실패: {e}. PYTHONPATH 또는 파일 위치를 확인하세요.", exc_info=True)
    sys.exit(f"필수 모듈 로드 실패 ({e}). 프로그램을 종료합니다.")

# .env 파일 로드 (프로젝트 루트에서)
try:
    from dotenv import load_dotenv

    dotenv_path = PROJECT_ROOT / '.env'
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)
        logging.info(f"Loaded .env file from: {dotenv_path}")
    else:
        logging.info(f".env file not found at {dotenv_path}, relying on environment variables if set.")
except ImportError:
    logging.warning("dotenv library not found, cannot load .env file. 'pip install python-dotenv' to use it.")
except Exception as e:
    logging.error(f"Error loading .env file: {e}", exc_info=True)

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
    test_date: Optional[str] = None


class ResultSaver(ABC):
    """결과 저장 추상 클래스"""

    @abstractmethod
    def save(self, summary_result: EvaluationResult, individual_results: List[Dict[str, Any]], filepath: str) -> None:
        pass


class JSONResultSaver(ResultSaver):
    """JSON 형식으로 결과 저장"""

    def save(self, summary_result: EvaluationResult, individual_results: List[Dict[str, Any]], filepath: str) -> None:
        summary_data = asdict(summary_result) if isinstance(summary_result, EvaluationResult) else summary_result

        data_to_save = {
            "summary": summary_data,
            "individual_question_results": individual_results
        }
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, ensure_ascii=False, indent=2)
            logger.info(f"평가 결과 저장 완료: {filepath}")
        except Exception as e:
            logger.error(f"결과 저장 실패 {filepath}: {e}", exc_info=True)


class AutoEvaluator:
    """
    자동화된 RAG 평가 시스템 클래스
    벡터서버 모델과 reranker 모델을 교체하며 성능을 자동으로 평가합니다.
    """

    def __init__(self, config_path_str: Optional[str] = None):
        self.logger = logger
        self.logger.info("AutoEvaluator 초기화 중...")

        if config_path_str is None:
            self.config_path = PROJECT_ROOT / "configs" / "evaluation_config.yaml"
        else:
            self.config_path = Path(config_path_str)

        if not self.config_path.exists():
            self.logger.error(f"설정 파일을 찾을 수 없습니다: {self.config_path}")
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {self.config_path}")

        try:
            # ModelManager 초기화 시 Config 객체 대신 설정 파일 경로(str)를 전달해야 할 수 있음
            # ModelManager가 Config 객체를 직접 받는다면 Config(str(self.config_path)) 와 같이 전달
            self.model_manager = ModelManager(str(self.config_path))
            self.logger.info("ModelManager 초기화 완료.")
        except Exception as e:
            self.logger.error(f"ModelManager 초기화 실패: {e}", exc_info=True)
            raise

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.main_config_data = yaml.safe_load(f)
            self.logger.info(f"설정 파일 로드 완료: {self.config_path}")
        except Exception as e:
            self.logger.error(f"설정 파일 로드 실패: {e}", exc_info=True)
            raise

        self.validate_config()

        self.evaluation_specific_config = self.main_config_data.get("evaluation", {})

        run_settings = self.evaluation_specific_config.get("run_settings", {})
        # 결과 디렉토리 경로를 프로젝트 루트 기준으로 설정
        self.result_dir = PROJECT_ROOT / Path(run_settings.get("result_dir", "evaluation_results"))
        self.result_dir.mkdir(parents=True, exist_ok=True)  # Path 객체의 mkdir 사용

        self.evaluation_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.all_runs_summary_list = []

    def validate_config(self) -> None:
        required_keys = ["evaluation", "evaluation.run_settings", "evaluation.embedding_models"]
        config_to_validate = self.main_config_data
        for key_path_str in required_keys:
            keys = key_path_str.split(".")
            current_level = config_to_validate
            for k_part in keys:
                if not isinstance(current_level, dict) or k_part not in current_level:
                    self.logger.critical(f"설정 파일에 필수 키 '{key_path_str}' (세부 키: '{k_part}')가 없습니다.")
                    raise ValueError(f"설정 파일에 필수 키 '{key_path_str}' (세부 키: '{k_part}')가 없습니다.")
                current_level = current_level[k_part]
        self.logger.info("설정 파일의 주요 구조 검증 완료.")

    def load_dataset(self, dataset_path_str: str) -> Dict[str, Any]:
        dataset_path = Path(dataset_path_str)
        if not dataset_path.is_absolute():
            dataset_path = PROJECT_ROOT / dataset_path

        if not dataset_path.exists():
            self.logger.error(f"데이터셋 파일을 찾을 수 없습니다: {dataset_path}")
            return {"questions": []}

        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            self.logger.info(f"데이터셋 로드 완료: {dataset_path} (질문 수: {len(dataset.get('questions', []))})")
            return dataset
        except json.JSONDecodeError as e:
            self.logger.error(f"데이터셋 JSON 디코딩 실패 ({dataset_path}): {e}")
            return {"questions": []}
        except Exception as e:
            self.logger.error(f"데이터셋 로드 중 예외 발생 ({dataset_path}): {e}", exc_info=True)
            return {"questions": []}

    def _is_document_relevant(self,
                              doc: Dict[str, Any],
                              ground_truth_ids: List[str],
                              essential_elements: List[str],
                              related_sections: List[str],
                              question: Dict[str, Any]
                              ) -> bool:
        doc_id = doc.get("id", "unknown")
        doc_score = doc.get("similarity", 0.0)
        content_lower = doc.get("content", "").lower()  # 여기서 lower() 한번만

        # self.logger.debug(f"문서 관련성 판단 시작 - 청크 ID: {doc_id}, 초기 유사도: {doc_score:.4f}") # 너무 빈번한 로그 줄이기

        if ground_truth_ids and doc_id:
            # GT ID가 이미 정제되었다고 가정하거나, 여기서 doc_id도 동일하게 정제
            clean_doc_id = str(doc_id).split('/')[-1]
            # ground_truth_ids 내부의 각 id도 동일한 방식으로 정제된 상태여야 함
            if clean_doc_id in [str(gid).split('/')[-1] for gid in ground_truth_ids]:
                self.logger.debug(f"  결과: 관련 (ID 일치: '{clean_doc_id}')")
                return True

        doc_metadata = doc.get("metadata", {})
        doc_section = doc_metadata.get("article_title", doc_metadata.get("section_title", "")).lower()
        if related_sections and doc_section:
            if any(section.lower() in doc_section for section in related_sections):
                self.logger.debug(f"  결과: 관련 (섹션 제목 매치: '{doc_section}')")
                return True

        doc_page_str = str(doc_metadata.get("page_num", -1))
        gt_page_numbers_str = [str(p) for p in question.get("page_numbers", [])]
        if doc_page_str != "-1" and gt_page_numbers_str and doc_page_str in gt_page_numbers_str:
            self.logger.debug(f"  결과: 관련 (페이지 번호 매치: {doc_page_str})")
            return True

        if essential_elements and content_lower:
            matches = sum(1 for element in essential_elements if element.lower() in content_lower)
            if matches >= max(1, len(essential_elements) * 0.2):
                self.logger.debug(f"  결과: 관련 (필수 요소 {matches}/{len(essential_elements)}개 매치)")
                return True

        expected_answer = question.get('expected_answer', "").lower()
        if expected_answer and content_lower:
            answer_sentences = [s.strip() for s in expected_answer.split('.') if len(s.strip()) > 10]
            if any(sentence in content_lower for sentence in answer_sentences):
                self.logger.debug(f"  결과: 관련 (예상 답변 주요 문장 일치)")
                return True

            answer_words = [w.strip() for w in expected_answer.split() if len(w.strip()) >= 2]
            if answer_words:
                word_matches = sum(1 for word in answer_words if word in content_lower)
                if word_matches >= max(2, int(len(answer_words) * 0.3)):  # 정수형으로 변환
                    self.logger.debug(f"  결과: 관련 (예상 답변 핵심 단어 {word_matches}/{len(answer_words)}개 매치)")
                    return True

            word_list_expected = expected_answer.split()
            for i in range(len(word_list_expected) - 1):
                phrase = f"{word_list_expected[i]} {word_list_expected[i + 1]}".lower()  # 루프 내에서 lower
                if len(word_list_expected[i]) >= 2 and len(word_list_expected[i + 1]) >= 2 and phrase in content_lower:
                    self.logger.debug(f"  결과: 관련 (예상 답변 2단어 구문 매치: '{phrase}')")
                    return True

        source_quote = question.get("gold_standard", {}).get("source_quote", "").lower()
        if source_quote and len(source_quote) > 5 and content_lower:
            quote_parts = source_quote.split()
            for i in range(len(quote_parts) - 1):
                phrase = " ".join(quote_parts[i:i + 2])
                if len(phrase) > 3 and phrase in content_lower:
                    self.logger.debug(f"  결과: 관련 (출처 인용 2단어 구문 매치: '{phrase}')")
                    return True
            key_words_quote = [word for word in quote_parts if len(word) >= 3]
            if key_words_quote:
                key_word_matches_quote = sum(1 for word in key_words_quote if word in content_lower)
                if key_word_matches_quote >= max(1, int(len(key_words_quote) * 0.2)):  # 정수형으로 변환
                    self.logger.debug(f"  결과: 관련 (출처 인용 핵심 단어 {key_word_matches_quote}/{len(key_words_quote)}개 매치)")
                    return True

        query_text_lower = question.get("text", "").lower()
        if query_text_lower and content_lower:
            query_words = [w.strip() for w in query_text_lower.split() if len(w.strip()) >= 2]
            if query_words:
                query_word_matches_in_content = sum(1 for word in query_words if word in content_lower)
                if query_word_matches_in_content >= max(2, int(len(query_words) * 0.3)):  # 정수형으로 변환
                    self.logger.debug(f"  결과: 관련 (질문 단어 {query_word_matches_in_content}/{len(query_words)}개 문서 내 매치)")
                    return True

        if doc_metadata and query_text_lower:
            metadata_str = str(doc_metadata).lower()
            query_words_for_meta = [w.strip() for w in query_text_lower.split() if len(w.strip()) >= 2]
            if query_words_for_meta:
                metadata_word_matches = sum(1 for word in query_words_for_meta if word in metadata_str)
                if metadata_word_matches >= 2:
                    self.logger.debug(f"  결과: 관련 (메타데이터 내 질문 단어 {metadata_word_matches}개 매치)")
                    return True

        # self.logger.debug(f"  결과: 관련 없음 (청크 ID: {doc_id})") # 너무 빈번한 로그 줄이기
        return False

    def _calculate_ndcg(self,
                        search_results: List[Dict[str, Any]],
                        ground_truth_ids: List[str],
                        essential_elements: List[str],
                        related_sections: List[str],
                        question: Dict[str, Any],
                        k_for_ndcg: Optional[int] = None) -> float:
        if k_for_ndcg is None: k_for_ndcg = len(search_results)
        results_to_consider = search_results[:k_for_ndcg]
        if not results_to_consider: return 0.0

        relevance_scores = []
        for doc_ndcg in results_to_consider:
            current_relevance_score = 0.0
            if self._is_document_relevant(doc_ndcg, ground_truth_ids, essential_elements, related_sections, question):
                current_relevance_score = 1.0

            sim_score_ndcg = doc_ndcg.get("similarity", 0.0)
            if sim_score_ndcg >= 0.8:
                current_relevance_score += 2.0
            elif sim_score_ndcg >= 0.6:
                current_relevance_score += 1.0
            relevance_scores.append(max(0.0, current_relevance_score))

        dcg = 0.0
        for i, rel_score_val in enumerate(relevance_scores):
            dcg += rel_score_val / np.log2(i + 2)

        ideal_scores_ndcg = sorted(relevance_scores, reverse=True)
        idcg = 0.0
        for i, ideal_score_val in enumerate(ideal_scores_ndcg):
            idcg += ideal_score_val / np.log2(i + 2)

        ndcg_final = dcg / idcg if idcg > 0 else 0.0
        # self.logger.debug(f"NDCG@{k_for_ndcg}: Scores={relevance_scores}, DCG={dcg:.4f}, IDCG={idcg:.4f}, Final NDCG={ndcg_final:.4f}") # 너무 상세한 로그 줄이기
        return ndcg_final

    def _contains_answer(self,
                         search_results: List[Dict[str, Any]],
                         essential_elements: List[str],
                         question: Dict[str, Any]) -> bool:
        if not search_results: return False

        all_content_lower = " ".join([doc.get("content", "").lower() for doc in search_results if doc.get("content")])
        if not all_content_lower.strip(): return False

        expected_answer = question.get('expected_answer', "").lower()
        if expected_answer:
            expected_sentences = [s.strip() for s in expected_answer.split('.') if len(s.strip()) > 10]
            if any(sentence in all_content_lower for sentence in expected_sentences): return True

            expected_words = [w.strip() for w in expected_answer.split() if len(w.strip()) >= 2]
            if expected_words:
                word_matches_exp = sum(1 for word in expected_words if word in all_content_lower)
                if word_matches_exp >= max(3, int(len(expected_words) * 0.3)): return True  # 정수 변환

            word_list_e = expected_answer.split()
            for i in range(len(word_list_e) - 1):
                phrase = f"{word_list_e[i]} {word_list_e[i + 1]}".lower()
                if len(word_list_e[i]) >= 2 and len(
                    word_list_e[i + 1]) >= 2 and phrase in all_content_lower: return True

        if essential_elements:
            essential_matches = sum(1 for element in essential_elements if element.lower() in all_content_lower)
            if essential_matches >= max(1, int(len(essential_elements) * 0.2)): return True  # 정수 변환

        source_quote = question.get("gold_standard", {}).get("source_quote", "").lower()
        if source_quote and len(source_quote) > 5:
            quote_parts = source_quote.split()
            for i in range(len(quote_parts) - 1):
                phrase = " ".join(quote_parts[i:i + 2])
                if len(phrase) >= 4 and phrase in all_content_lower: return True
            key_words_sq = [word for word in quote_parts if len(word) >= 4]
            if key_words_sq:
                key_word_matches_sq = sum(1 for word in key_words_sq if word in all_content_lower)
                if key_word_matches_sq >= max(1, int(len(key_words_sq) * 0.2)): return True  # 정수 변환

        question_text_lower = question.get("text", "").lower()
        if any(suffix in question_text_lower for suffix in ["는가요", "인가요", "습니까", "나요"]):
            if any(ans_pattern in all_content_lower[:300] for ans_pattern in ["예", "아니", "네", "없", "맞"]): return True

        query_words = [w.strip() for w in question_text_lower.split() if len(w.strip()) >= 2]
        if query_words:
            query_word_matches = sum(1 for word in query_words if word in all_content_lower)
            if query_word_matches >= int(len(query_words) * 0.7): return True  # 정수 변환

        if search_results and search_results[0].get("similarity", 0.0) >= 0.6: return True
        if len(search_results) >= 3:
            avg_top3_similarity = sum(doc.get("similarity", 0.0) for doc in search_results[:3]) / 3
            if avg_top3_similarity >= 0.5: return True

        return False

    def evaluate_results(self,
                         search_results: List[Dict[str, Any]],
                         ground_truth_ids: List[str],
                         essential_elements: List[str],
                         related_sections: List[str],
                         question: Dict[str, Any]) -> Dict[str, Any]:

        if not essential_elements and 'expected_answer' in question and question['expected_answer']:
            sentences = re.split(r'(?<=[다나까여요죠구려했죠습니다]\.)\s+', question['expected_answer'])
            potential_elements = [s.strip() for s in sentences if len(s.strip()) > 10]
            essential_elements = sorted(potential_elements, key=len, reverse=True)[:5]

        if not search_results:
            default_total_relevant = len(ground_truth_ids) if ground_truth_ids else (
                len(essential_elements) if essential_elements else 1)
            return {
                "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "ndcg": 0.0, "mrr": 0.0,
                "has_answer": False, "relevant_docs_count": 0, "irrelevant_docs_count": 0,
                "search_results_count": 0, "total_relevant_estimate": default_total_relevant
            }

        retrieved_relevant_docs = [
            doc for doc in search_results
            if self._is_document_relevant(doc, ground_truth_ids, essential_elements, related_sections, question)
        ]
        num_retrieved_relevant = len(retrieved_relevant_docs)
        num_search_results = len(search_results)

        precision = num_retrieved_relevant / num_search_results if num_search_results > 0 else 0.0

        total_relevant_estimate = len(ground_truth_ids) if ground_truth_ids else (
            len(essential_elements) if essential_elements else 1)
        total_relevant_for_recall = max(total_relevant_estimate, num_retrieved_relevant)

        recall = num_retrieved_relevant / total_relevant_for_recall if total_relevant_for_recall > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        mrr = 0.0
        for i, doc_mrr in enumerate(search_results):
            if self._is_document_relevant(doc_mrr, ground_truth_ids, essential_elements, related_sections, question):
                mrr = 1.0 / (i + 1)
                break

        ndcg = self._calculate_ndcg(search_results, ground_truth_ids, essential_elements, related_sections, question)
        has_answer_val_final = self._contains_answer(search_results, essential_elements, question)

        metrics_output = {
            "precision": precision, "recall": recall, "f1_score": f1_score,
            "ndcg": ndcg, "mrr": mrr, "has_answer": has_answer_val_final,
            "relevant_docs_count": num_retrieved_relevant,
            "irrelevant_docs_count": num_search_results - num_retrieved_relevant,
            "search_results_count": num_search_results,
            "total_relevant_estimate": total_relevant_estimate
        }
        # self.logger.info(f"QID '{question.get('id')}': P={precision:.4f}, R={recall:.4f}, F1={f1_score:.4f}, MRR={mrr:.4f}, NDCG={ndcg:.4f}, HasAns={has_answer_val_final}") # 로그 줄이기

        is_all_numeric_metrics_zero = all(
            metrics_output[key] == 0.0 for key in ["precision", "recall", "f1_score", "ndcg", "mrr"]
        )
        if is_all_numeric_metrics_zero and num_search_results > 0 and not num_retrieved_relevant:
            self.logger.warning(f"QID '{question.get('id')}': 주요 수치 메트릭이 0입니다. (검색결과 {num_search_results}개 중 관련문서 0개)")
        return metrics_output

    def _build_eval_run_config(self, embedding_model_name: str, collection_name: str,
                               top_k: int, similarity_threshold: float,
                               is_reranking_enabled: bool) -> Dict[str, Any]:
        run_s_cfg = self.evaluation_specific_config.get("run_settings", {})
        embedding_dim_cfg = self.model_manager.get_embedding_dimension(embedding_model_name) or 768

        # Create a deep copy of main_config_data to avoid modifying the original
        temp_run_config_data = json.loads(json.dumps(self.main_config_data))  # 간단한 deepcopy 방식

        # Override specific sections for this evaluation run
        temp_run_config_data['embedding'] = {
            "model": embedding_model_name, "dimension": embedding_dim_cfg,
            "normalize": self.model_manager.embedding_model_configs.get(embedding_model_name, {}).get("normalize",
                                                                                                      True),
            "batch_size": self.model_manager.embedding_model_configs.get(embedding_model_name, {}).get("batch_size", 32)
        }
        temp_run_config_data['retrieval'] = {
            "vector_db_type": "milvus", "top_k": top_k, "similarity_threshold": similarity_threshold,
            "similarity_metric": self.evaluation_specific_config.get("similarity_metric", "COSINE"),
            "hybrid_search": self.evaluation_specific_config.get("hybrid_search_in_eval", False),
            "reranking": is_reranking_enabled, "collections": [collection_name],  # retriever가 리스트를 기대
            "small_chunk_types": self.evaluation_specific_config.get("small_chunk_types",
                                                                     ["item", "item_sub_chunk", "csv_row",
                                                                      "text_block"]),
            "parent_chunk_data_dir": run_s_cfg.get("parent_chunk_data_dir",
                                                   str(PROJECT_ROOT / "data" / "parsed_output")),
            "milvus_output_fields": self.evaluation_specific_config.get("milvus_output_fields",
                                                                        ["id", "text", "doc_id", "source", "page_num",
                                                                         "chunk_id", "parent_chunk_id", "chunk_type",
                                                                         "article_title", "item_marker"]
                                                                        )
        }
        
        # 리랭커 모델 설정 추가
        if is_reranking_enabled:
            # 리랭커 모델 이름 처리
            reranker_model_name = ""
            if isinstance(self.evaluation_specific_config.get("reranker_model", ""), dict):
                reranker_model_name = self.evaluation_specific_config.get("reranker_model", {}).get("name", "")
            else:
                reranker_model_name = self.evaluation_specific_config.get("reranker_model", "")
            temp_run_config_data['retrieval']['reranker_model'] = reranker_model_name
        temp_run_config_data['milvus'] = {
            "host": os.getenv("MILVUS_HOST", "localhost"), "port": int(os.getenv("MILVUS_PORT", "19530")),
            "user": os.getenv("MILVUS_USER", ""), "password": os.getenv("MILVUS_PASSWORD", ""),
            "index_type": os.getenv("MILVUS_INDEX_TYPE",
                                    self.evaluation_specific_config.get("milvus_index_type", "HNSW")),
            "metric_type": self.evaluation_specific_config.get("similarity_metric", "COSINE"),
            "index_params": self.evaluation_specific_config.get("milvus_index_params_hnsw",
                                                                {"M": 16, "efConstruction": 256}),
            "search_params": self.evaluation_specific_config.get("milvus_search_params_hnsw", {"ef": 64})
        }
        if 'general' not in temp_run_config_data: temp_run_config_data['general'] = {}
        temp_run_config_data['general']['log_level'] = logging.getLevelName(logger.getEffectiveLevel())

        return temp_run_config_data

    def _get_model_name(self, model_name_or_config: Union[str, Dict[str, Any]]) -> str:
        """
        모델 이름 또는 모델 설정 딕셔너리에서 실제 모델 이름을 추출하는 유틸리티 함수
        
        Args:
            model_name_or_config: 모델 이름(문자열) 또는 모델 설정 딕셔너리
            
        Returns:
            모델 이름 문자열
        """
        if isinstance(model_name_or_config, dict) and 'name' in model_name_or_config:
            return model_name_or_config['name']
        return str(model_name_or_config) if model_name_or_config is not None else ""

    def evaluate_embedding_model(self,
                                 embedding_model_name: str,
                                 dataset_path: str,
                                 top_k: int = 10,
                                 similarity_threshold: float = 0.7,
                                 reranker_model_name: Optional[str] = None,
                                 collection_name_override: Optional[str] = None) -> Optional[EvaluationResult]:

        if not (0.0 <= similarity_threshold <= 1.0):
            self.logger.error(f"유사도 임계값(입력: {similarity_threshold})은 0.0과 1.0 사이여야 합니다.")
            return None

        self.logger.info(f"===== 임베딩 모델 평가 시작: {embedding_model_name} =====")
        self.logger.info(f"  데이터셋: {dataset_path}")
        self.logger.info(
            f"  설정: TopK={top_k}, Threshold={similarity_threshold}, Reranker='{reranker_model_name or '없음'}'")
        self.logger.info(f"  컬렉션 Override: '{collection_name_override or '기본값 사용'}'")

        run_s_cfg = self.evaluation_specific_config.get("run_settings", {})

        dataset = self.load_dataset(dataset_path)
        questions = dataset.get("questions", [])
        if not questions:
            self.logger.error(f"데이터셋 '{dataset_path}'에 평가할 질문이 없어 이 조합 건너뜁니다.")
            return None

        try:
            self.model_manager.load_embedding_model(embedding_model_name)
            
            # 리랭커 모델 처리 수정: 딕셔너리 객체인 경우 실제 이름 추출
            if reranker_model_name:
                # 이미 get_model_name 함수가 있으니 활용
                actual_reranker_name = self._get_model_name(reranker_model_name)
                self.logger.info(f"리랭커 모델 로드: {actual_reranker_name}")
                self.model_manager.load_reranker_model(actual_reranker_name)
        except Exception as e_model_load:
            self.logger.error(
                f"모델 로드 실패 (Emb='{embedding_model_name}', Reranker='{reranker_model_name}'): {e_model_load}",
                exc_info=True)
            return None

        config_coll_name = run_s_cfg.get("collection_name",
                                         f"eval_coll_{embedding_model_name.split('/')[-1].replace('-', '_')}")
        final_collection_name = os.getenv("MILVUS_COLLECTION_EVAL") or collection_name_override or config_coll_name

        eval_run_config_dict_built = self._build_eval_run_config(embedding_model_name, final_collection_name, top_k,
                                                                 similarity_threshold, reranker_model_name is not None)

        # 이제 Config 클래스가 config_dict 매개변수를 통해 딕셔너리를 직접 받을 수 있음
        temp_config_for_eval_run = Config(config_dict=eval_run_config_dict_built)

        try:
            retriever = DocumentRetriever(temp_config_for_eval_run)  # 임시 config 전달
        except Exception as e_retriever_init:
            self.logger.error(f"DocumentRetriever 초기화 실패 (Collection: {final_collection_name}): {e_retriever_init}",
                              exc_info=True)
            return None

        individual_q_results = []
        total_retrieval_time_ns_val = 0
        total_reranking_time_ns_val = 0
        q_with_results_count = 0
        q_fallback_count = 0

        for q_data in tqdm(questions, desc=f"질문 평가 ({embedding_model_name} | {reranker_model_name or 'NoReranker'})",
                           unit="q"):
            query_txt = q_data.get("text", "")
            q_id = q_data.get("id", str(uuid.uuid4()))
            q_type_val = q_data.get("type", "unknown")

            if not query_txt.strip(): continue

            search_res_current = []
            retrieval_start_ns_val = time.perf_counter_ns()
            try:
                use_s2b_flag_eval = run_s_cfg.get("use_small_to_big_in_eval", True)
                enable_opt_flag_eval = run_s_cfg.get("enable_query_optimization_in_eval", False)

                raw_s_results = retriever.retrieve(
                    query=query_txt, top_k=top_k * 2,
                    threshold=max(0.05, similarity_threshold * 0.8),
                    target_collections=[final_collection_name],
                    use_parent_chunks=use_s2b_flag_eval,
                    enable_query_optimization=enable_opt_flag_eval,
                    force_filter_expr='default'
                )
                search_res_current = [doc for doc in raw_s_results if
                                      doc.get("similarity", 0.0) >= similarity_threshold][:top_k]

                if search_res_current:
                    q_with_results_count += 1
                else:
                    fallback_s_results = retriever.retrieve(
                        query=query_txt, top_k=top_k, threshold=0.0, target_collections=[final_collection_name],
                        use_parent_chunks=use_s2b_flag_eval, enable_query_optimization=enable_opt_flag_eval,
                        force_filter_expr='default'
                    )
                    if fallback_s_results:
                        search_res_current = fallback_s_results
                        q_fallback_count += 1
            except Exception as e_search:
                self.logger.error(f"검색 중 오류 (QID: '{q_id}'): {e_search}", exc_info=True)

            retrieval_time_ns_val = time.perf_counter_ns() - retrieval_start_ns_val
            total_retrieval_time_ns_val += retrieval_time_ns_val

            current_rerank_time_ns_val = 0.0
            if reranker_model_name and search_res_current:
                reranking_start_ns_val = time.perf_counter_ns()
                try:
                    docs_for_rr = [doc.copy() for doc in search_res_current]
                    for doc_to_rr in docs_for_rr: doc_to_rr['score'] = doc_to_rr.get('similarity', 0.0)

                    reranked_list = self.model_manager.rerank_results(
                        query=query_txt, docs=docs_for_rr,
                        model_name=self._get_model_name(reranker_model_name),
                        top_n=top_k
                    )

                    temp_reranked_list = []
                    reranked_map_ids = {r_doc.get("id"): r_doc for r_doc in reranked_list}
                    for orig_doc_sr in search_res_current:
                        doc_id_orig = orig_doc_sr.get("id")
                        if doc_id_orig in reranked_map_ids:
                            reranked_v_doc = reranked_map_ids[doc_id_orig]
                            merged_doc_item = orig_doc_sr.copy()
                            merged_doc_item["similarity"] = reranked_v_doc.get("score",
                                                                               orig_doc_sr.get("similarity", 0.0))
                            merged_doc_item.setdefault("metadata", {})["reranker_score"] = reranked_v_doc.get("score",
                                                                                                              0.0)
                            temp_reranked_list.append(merged_doc_item)
                    search_res_current = sorted(temp_reranked_list, key=lambda x_doc: x_doc.get("similarity", 0.0),
                                                reverse=True)
                except Exception as e_rerank:
                    self.logger.error(f"재순위화 실패 (QID: '{q_id}'): {e_rerank}", exc_info=True)
                current_rerank_time_ns_val = time.perf_counter_ns() - reranking_start_ns_val
                total_reranking_time_ns_val += current_rerank_time_ns_val

            gold_std_data = q_data.get("gold_standard", {})
            gt_ids_list = gold_std_data.get("document_ids", [])
            essential_phrases_list = gold_std_data.get("essential_elements", [])

            metrics_q = self.evaluate_results(
                search_results=search_res_current, ground_truth_ids=gt_ids_list,
                essential_elements=essential_phrases_list, related_sections=[], question=q_data
            )

            individual_q_results.append({
                "question_id": q_id, "question_type": q_type_val, "question_text": query_txt,
                "retrieved_chunk_ids": [res.get("id") for res in search_res_current],
                "retrieved_similarities": [res.get("similarity") for res in search_res_current],
                "metrics": metrics_q,
                "retrieval_time_seconds": retrieval_time_ns_val / 1e9,
                "reranking_time_seconds": current_rerank_time_ns_val / 1e9 if reranker_model_name and search_res_current else None
            })

        if not individual_q_results:
            self.logger.error("개별 질문 결과가 없습니다. 평가 조합을 건너뜁니다.")
            return None

        avg_ret_time_s = (total_retrieval_time_ns_val / 1e9) / len(individual_q_results)
        avg_rerank_time_s = None
        if reranker_model_name:
            num_reranked_actual = sum(1 for r_item in individual_q_results if
                                      r_item.get("reranking_time_seconds") is not None and r_item.get(
                                          "reranking_time_seconds") > 0)
            if num_reranked_actual > 0:
                avg_rerank_time_s = (total_reranking_time_ns_val / 1e9) / num_reranked_actual

        self.logger.info(
            f"임계값 만족 질문: {q_with_results_count}/{len(questions)}, Fallback 사용: {q_fallback_count}/{len(questions)}")

        analysis_sum_metrics = self.analyze_results(individual_q_results)
        
        # 리랭커 모델이 딕셔너리인 경우 문자열로 변환
        actual_reranker_model = reranker_model_name
        if isinstance(reranker_model_name, dict) and 'name' in reranker_model_name:
            actual_reranker_model = reranker_model_name['name']
        elif reranker_model_name is not None:
            actual_reranker_model = str(reranker_model_name)

        eval_result_output = EvaluationResult(
            embedding_model=embedding_model_name, reranker_model=actual_reranker_model,
            top_k=top_k, similarity_threshold=similarity_threshold,
            precision=analysis_sum_metrics["overall_metrics"]["precision"],
            recall=analysis_sum_metrics["overall_metrics"]["recall"],
            f1_score=analysis_sum_metrics["overall_metrics"]["f1_score"],
            ndcg=analysis_sum_metrics["overall_metrics"]["ndcg"],
            mrr=analysis_sum_metrics["overall_metrics"]["mrr"],
            has_answer_rate=analysis_sum_metrics["overall_metrics"]["has_answer_rate"],
            avg_retrieval_time=avg_ret_time_s,
            avg_reranking_time=avg_rerank_time_s,
            total_questions=len(individual_q_results),
            question_types=analysis_sum_metrics["by_question_type"],
            test_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        self.save_evaluation_output(eval_result_output, individual_q_results, dataset_path)
        self.all_runs_summary_list.append(asdict(eval_result_output))

        try:
            self.model_manager.unload_models()
        except Exception as e_unload:
            self.logger.error(f"모델 언로드 실패: {e_unload}", exc_info=True)

        return eval_result_output

    def analyze_results(self, individual_question_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not individual_question_results:
            self.logger.warning("분석할 개별 질문 결과가 없습니다.")
            return {
                "overall_metrics": {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "ndcg": 0.0, "mrr": 0.0,
                                    "has_answer_rate": 0.0,
                                    "avg_relevant_docs_retrieved": 0.0, "avg_search_results_returned": 0.0,
                                    "avg_total_relevant_estimated": 0.0},
                "by_question_type": {}
            }

        overall_metrics_sum = {k: 0.0 for k in ["precision", "recall", "f1_score", "ndcg", "mrr", "has_answer_count",
                                                "relevant_docs_total", "search_results_total",
                                                "ground_truth_total_estimate"]}
        by_question_type_agg: Dict[str, Dict[str, Any]] = {}

        for res_item in individual_question_results:
            metrics = res_item.get("metrics", {})
            q_type = res_item.get("question_type", "unknown_type")

            for key in ["precision", "recall", "f1_score", "ndcg", "mrr"]:
                overall_metrics_sum[key] += metrics.get(key, 0.0)
            if metrics.get("has_answer", False): overall_metrics_sum["has_answer_count"] += 1
            overall_metrics_sum["relevant_docs_total"] += metrics.get("relevant_docs_count", 0)
            overall_metrics_sum["search_results_total"] += metrics.get("search_results_count", 0)
            overall_metrics_sum["ground_truth_total_estimate"] += metrics.get("total_relevant_estimate", 0)

            if q_type not in by_question_type_agg:
                by_question_type_agg[q_type] = {
                    "count": 0, 
                    "precision_sum": 0.0, 
                    "recall_sum": 0.0, 
                    "f1_score_sum": 0.0,  # f1_sum 대신 f1_score_sum 사용
                    "mrr_sum": 0.0, 
                    "ndcg_sum": 0.0, 
                    "has_answer_sum": 0
                }

            by_question_type_agg[q_type]["count"] += 1
            for key in ["precision", "recall", "f1_score", "mrr", "ndcg"]:
                # 모든 키에 대해 일관된 '_sum' 형식 사용
                sum_key = f"{key}_sum"
                by_question_type_agg[q_type][sum_key] += metrics.get(key, 0.0)
            if metrics.get("has_answer", False): by_question_type_agg[q_type]["has_answer_sum"] += 1

        num_total_q = len(individual_question_results)
        avg_overall_metrics = {
            metric_name: overall_metrics_sum[metric_name] / num_total_q if num_total_q > 0 else 0.0
            for metric_name in ["precision", "recall", "f1_score", "ndcg", "mrr"]
        }
        avg_overall_metrics["has_answer_rate"] = overall_metrics_sum[
                                                     "has_answer_count"] / num_total_q if num_total_q > 0 else 0.0
        avg_overall_metrics["avg_relevant_docs_retrieved"] = overall_metrics_sum[
                                                                 "relevant_docs_total"] / num_total_q if num_total_q > 0 else 0.0
        avg_overall_metrics["avg_search_results_returned"] = overall_metrics_sum[
                                                                 "search_results_total"] / num_total_q if num_total_q > 0 else 0.0
        avg_overall_metrics["avg_total_relevant_estimated"] = overall_metrics_sum[
                                                                  "ground_truth_total_estimate"] / num_total_q if num_total_q > 0 else 0.0

        # self.logger.info(f"전체 평균 메트릭 ({num_total_q}개 질문): " + ", ".join([f"{k.replace('_',' ').title()}={v:.4f}" for k,v in avg_overall_metrics.items()])) # 로그 줄이기

        final_q_type_analysis = {}
        for q_type, data_sum in by_question_type_agg.items():
            count = data_sum["count"]
            final_q_type_analysis[q_type] = {"count": count}
            
            # f1_score_sum 키가 있는지 확인하고 없으면 f1_sum 키를 사용
            if "f1_score_sum" in data_sum:
                final_q_type_analysis[q_type]["avg_f1_score"] = data_sum["f1_score_sum"] / count if count > 0 else 0.0
            elif "f1_sum" in data_sum:
                final_q_type_analysis[q_type]["avg_f1_score"] = data_sum["f1_sum"] / count if count > 0 else 0.0
            else:
                final_q_type_analysis[q_type]["avg_f1_score"] = 0.0
                
            # precision, recall, mrr, ndcg 처리            
            for key in ["precision", "recall", "mrr", "ndcg"]:
                final_q_type_analysis[q_type][f"avg_{key}_score"] = data_sum[f"{key}_sum"] / count if count > 0 else 0.0
                
            final_q_type_analysis[q_type]["has_answer_rate"] = data_sum["has_answer_sum"] / count if count > 0 else 0.0
            # self.logger.info(f"  유형 '{q_type}' ({count}개): F1={final_q_type_analysis[q_type]['avg_f1_score']:.4f}, HasAnsRate={final_q_type_analysis[q_type]['has_answer_rate']:.2%}") # 로그 줄이기

        return {"overall_metrics": avg_overall_metrics, "by_question_type": final_q_type_analysis}

    def save_evaluation_output(self,
                               summary_result_obj: EvaluationResult,
                               individual_q_results: List[Dict[str, Any]],
                               dataset_path_str: str):
        emb_model_safe = re.sub(r'[\\/*?:"<>|]', "_", summary_result_obj.embedding_model)
        reranker_safe = ""
        if summary_result_obj.reranker_model:
            # reranker_model이 딕셔너리인 경우 name 필드 추출
            if isinstance(summary_result_obj.reranker_model, dict) and 'name' in summary_result_obj.reranker_model:
                reranker_name = summary_result_obj.reranker_model['name']
            else:
                reranker_name = str(summary_result_obj.reranker_model)
            reranker_safe = "_reranker_" + re.sub(r'[\\/*?:"<>|]', "_", reranker_name)

        dataset_file_stem = Path(dataset_path_str).stem
        filename = (
            f"eval_{self.evaluation_run_id}_{dataset_file_stem}_emb_{emb_model_safe}"
            f"{reranker_safe}_top{summary_result_obj.top_k}_thr{summary_result_obj.similarity_threshold:.2f}.json"
        )
        output_filepath = self.result_dir / filename

        saver = JSONResultSaver()
        saver.save(summary_result_obj, individual_q_results, str(output_filepath))

    def save_all_runs_summary(self) -> None:
        if not self.all_runs_summary_list:
            self.logger.warning("저장할 전체 실행 요약 결과가 없습니다.")
            return

        summary_filename = f"ALL_EVALUATION_RUNS_SUMMARY_{self.evaluation_run_id}.json"
        summary_filepath = self.result_dir / summary_filename

        summary_data_to_save = {
            "overall_evaluation_run_id": self.evaluation_run_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_evaluation_configurations_run": len(self.all_runs_summary_list),
            "evaluation_run_summaries": self.all_runs_summary_list
        }

        try:
            with open(summary_filepath, 'w', encoding='utf-8') as f:
                json.dump(summary_data_to_save, f, ensure_ascii=False, indent=2)
            self.logger.info(f"모든 평가 실행 요약 결과 저장 완료: {summary_filepath}")
        except Exception as e:
            self.logger.error(f"모든 평가 실행 요약 결과 저장 실패: {summary_filepath}: {e}", exc_info=True)

        self._generate_performance_ranking_report()

    def _generate_performance_ranking_report(self) -> None:
        if not self.all_runs_summary_list:
            self.logger.warning("성능 순위 생성 위한 결과 데이터 없음.")
            return

        metrics_to_rank = ["precision", "recall", "f1_score", "ndcg", "mrr", "has_answer_rate"]
        performance_rankings = {}

        for metric_key in metrics_to_rank:
            sorted_run_results = sorted(
                self.all_runs_summary_list,
                key=lambda r_dict: r_dict.get(metric_key, 0.0),
                reverse=True
            )

            performance_rankings[metric_key] = [
                {
                    "embedding_model": res_dict.get("embedding_model", "").split('/')[-1],
                    "reranker_model": res_dict.get("reranker_model", "").split('/')[-1] if res_dict.get(
                        "reranker_model") else "None",
                    "top_k": res_dict.get("top_k"),
                    "similarity_threshold": res_dict.get("similarity_threshold"),
                    "score": res_dict.get(metric_key)
                }
                for res_dict in sorted_run_results[:min(5, len(sorted_run_results))]
            ]

        ranking_filename = f"PERFORMANCE_RANKING_{self.evaluation_run_id}.json"
        ranking_filepath = self.result_dir / ranking_filename

        try:
            with open(ranking_filepath, 'w', encoding='utf-8') as f:
                json.dump(performance_rankings, f, ensure_ascii=False, indent=2)
            self.logger.info(f"성능 순위 리포트 저장 완료: {ranking_filepath}")
        except Exception as e:
            self.logger.error(f"성능 순위 리포트 저장 실패: {ranking_filepath}: {e}", exc_info=True)

    def run_all_evaluations(self) -> None:
        self.logger.info("===== 모든 설정 조합에 대한 자동 평가 시작 =====")
        self.all_runs_summary_list = []

        run_s = self.evaluation_specific_config.get("run_settings", {})
        dataset_paths_str_list = run_s.get("dataset_paths", [])  # 변수명 변경
        top_k_val_list = run_s.get("top_k_values", [5, 10])  # 변수명 변경
        sim_thresh_list = run_s.get("similarity_thresholds", [0.7, 0.75, 0.8])  # 변수명 변경
        collections_override_list = run_s.get("collections_to_evaluate_override", [None])  # 변수명 변경

        emb_model_names_list = self.evaluation_specific_config.get("embedding_models", [])  # 변수명 변경
        reranker_names_list = self.evaluation_specific_config.get("reranker_models", [])  # 변수명 변경
        if None not in reranker_names_list:
            reranker_names_list = [None] + reranker_names_list
        elif not reranker_names_list:
            reranker_names_list = [None]

        if not dataset_paths_str_list: self.logger.critical("평가 데이터셋 경로가 없습니다."); return
        if not emb_model_names_list: self.logger.critical("평가할 임베딩 모델이 없습니다."); return

        total_combinations_val = len(dataset_paths_str_list) * len(emb_model_names_list) * \
                                 len(reranker_names_list) * len(top_k_val_list) * \
                                 len(sim_thresh_list) * len(collections_override_list)  # 변수명 변경
        self.logger.info(f"예상되는 총 평가 실행 횟수: {total_combinations_val}")
        current_combination_num = 0  # 변수명 변경

        for dataset_str_path in dataset_paths_str_list:  # 변수명 변경
            dataset_p_obj = Path(dataset_str_path)  # 변수명 변경
            if not dataset_p_obj.is_absolute(): dataset_p_obj = PROJECT_ROOT / dataset_p_obj
            if not dataset_p_obj.exists():
                self.logger.error(f"데이터셋 파일 없음: {dataset_p_obj}. 건너뜀.")
                continue
            for emb_m_n_loop in emb_model_names_list:  # 변수명 변경
                for reranker_m_n_loop in reranker_names_list:  # 변수명 변경
                    for tk_val_l in top_k_val_list:  # 변수명 변경
                        for sim_thresh_l in sim_thresh_list:  # 변수명 변경
                            for coll_name_ov_l in collections_override_list:  # 변수명 변경
                                current_combination_num += 1
                                self.logger.info(f"--- 평가 실행 ({current_combination_num}/{total_combinations_val}) ---")
                                # 이하 로깅 및 evaluate_embedding_model 호출은 이전 버전과 동일하게 유지
                                self.logger.info(
                                    f"  Dataset: {dataset_str_path}, Embedding: {emb_m_n_loop}, Reranker: {reranker_m_n_loop or 'None'}")
                                self.logger.info(
                                    f"  TopK: {tk_val_l}, Threshold: {sim_thresh_l}, CollectionOverride: {coll_name_ov_l or 'DefaultInternal'}")

                                try:
                                    eval_summary_item = self.evaluate_embedding_model(  # 변수명 변경
                                        embedding_model_name=emb_m_n_loop,
                                        dataset_path=str(dataset_p_obj),
                                        top_k=tk_val_l,
                                        similarity_threshold=sim_thresh_l,
                                        reranker_model_name=reranker_m_n_loop,
                                        collection_name_override=coll_name_ov_l
                                    )
                                    if eval_summary_item:
                                        self.logger.info(
                                            f"  조합 평가 완료: F1={eval_summary_item.f1_score:.4f}, MRR={eval_summary_item.mrr:.4f}, HasAns={eval_summary_item.has_answer_rate:.2%}")
                                    else:
                                        self.logger.warning("  이 조합에 대한 평가 결과가 생성되지 않았습니다 (None 반환).")
                                except Exception as e_outer_loop:  # 변수명 변경
                                    self.logger.error(f"  평가 조합 실행 중 심각한 오류 (조합 건너뜀): {e_outer_loop}", exc_info=True)
                                self.logger.info(
                                    f"--- 평가 종료 ({current_combination_num}/{total_combinations_val}) ---\n")

        self.logger.info("모든 평가 조합에 대한 실행 완료.")
        self.save_all_runs_summary()

        try:
            self.model_manager.unload_models(unload_all=True)
            self.logger.info("모든 평가 완료 후 사용된 모든 모델 언로드 시도.")
        except Exception as e_final_unload:  # 변수명 변경
            self.logger.error(f"최종 모델 언로드 중 오류: {e_final_unload}", exc_info=True)

        self.logger.info("===== 자동 평가 전체 종료 =====")

    def evaluate_single_configuration(self,
                                      embedding_model: str,
                                      dataset_path: str,
                                      top_k: int = 10,
                                      similarity_threshold: float = 0.7,
                                      reranker_models_list: Optional[List[Optional[str]]] = None,
                                      collection_name: Optional[str] = None) -> None:
        self.logger.info(f"단일 임베딩 모델 '{embedding_model}'에 대한 구성 평가 시작...")
        # 이하 evaluator_part22의 로직과 이전 버전의 통합된 로직 사용
        if not (0.0 <= similarity_threshold <= 1.0):
            self.logger.error(f"유사도 임계값(입력: {similarity_threshold})은 0.0과 1.0 사이여야 합니다.")
            return

        rerankers_to_eval = []
        if reranker_models_list is None:
            rerankers_to_eval = self.evaluation_specific_config.get("reranker_models", [])
            # 설정 파일에 reranker_models가 아예 없거나 빈 리스트일 경우를 대비
            if not isinstance(rerankers_to_eval, list): rerankers_to_eval = []
        else:
            rerankers_to_eval = reranker_models_list

        if None not in rerankers_to_eval:  # 리랭커 없음 케이스 명시적 포함 (중복 방지하며)
            rerankers_to_eval = [None] + [r for r in rerankers_to_eval if r is not None]

        self.logger.info(
            f"  Dataset: {dataset_path}, TopK: {top_k}, Threshold: {similarity_threshold}, Collection: {collection_name or 'DefaultInternal'}")
        self.logger.info(f"  테스트할 리랭커 모델 조합: {rerankers_to_eval}")

        base_run_result_obj = None

        for reranker_model_single_eval in rerankers_to_eval:
            # 리랭커 모델 정보 표시 개선
            reranker_display_name = "None"
            if reranker_model_single_eval:
                if isinstance(reranker_model_single_eval, dict) and 'name' in reranker_model_single_eval:
                    reranker_display_name = reranker_model_single_eval['name']
                else:
                    reranker_display_name = str(reranker_model_single_eval)
                    
            self.logger.info(f"  Reranker 조합 테스트 시작: '{reranker_display_name}'")
            try:
                current_eval_result_obj = self.evaluate_embedding_model(
                    embedding_model_name=embedding_model,
                    dataset_path=dataset_path,
                    top_k=top_k,
                    similarity_threshold=similarity_threshold,
                    reranker_model_name=reranker_model_single_eval,
                    collection_name_override=collection_name
                )
                if current_eval_result_obj:
                    self.logger.info(
                        f"    단일 구성 평가 완료 (Reranker: {reranker_display_name}): F1={current_eval_result_obj.f1_score:.4f}, MRR={current_eval_result_obj.mrr:.4f}")
                    if reranker_model_single_eval is None:
                        base_run_result_obj = current_eval_result_obj
                else:
                    self.logger.warning(f"    단일 구성 평가 결과 없음 (Reranker: {reranker_display_name}).")
            except Exception as e_single_cfg_run:
                self.logger.error(
                    f"    단일 구성 평가 중 오류 (Reranker: {reranker_display_name}): {e_single_cfg_run}",
                    exc_info=True)

        self.logger.info(f"단일 임베딩 모델 '{embedding_model}'에 대한 모든 리랭커 조합 평가 완료.")

        if base_run_result_obj and self.all_runs_summary_list:  # all_runs_summary_list에 결과가 쌓였을 것
            print("\n=== 최종 평가 결과 비교 (단일 임베딩 모델에 대한 리랭커 효과) ===")
            # (이하 evaluator_part22의 비교 출력 로직 동일하게 적용)
            print(f"임베딩 모델: {embedding_model}")
            # ... (나머지 비교 출력 코드) ...
            current_reranker_results_objs = [  # EvaluationResult 객체로 다시 변환
                EvaluationResult(**r_dict) for r_dict in self.all_runs_summary_list
                if r_dict['embedding_model'] == embedding_model and \
                   r_dict['reranker_model'] is not None and \
                   r_dict['top_k'] == top_k and \
                   abs(r_dict['similarity_threshold'] - similarity_threshold) < 0.001
            ]
            if current_reranker_results_objs:
                print("\nReranker 모델별 결과:")
                for res_rerank_obj in current_reranker_results_objs:
                    reranker_short_name_disp = res_rerank_obj.reranker_model.split('/')[-1]
                    print(f"\n* Reranker: {reranker_short_name_disp}")
                    print(
                        f"  F1 Score: {res_rerank_obj.f1_score:.4f} (Diff: {res_rerank_obj.f1_score - base_run_result_obj.f1_score:+.4f})")
                    # ... (MRR, NDCG 등 다른 메트릭도 유사하게 출력) ...
                    avg_rerank_t_disp = res_rerank_obj.avg_reranking_time if res_rerank_obj.avg_reranking_time is not None else 0.0
                    print(f"  Avg Reranking Time: {avg_rerank_t_disp:.4f}s")
                    print(f"  Total Effective Time: {res_rerank_obj.avg_retrieval_time + avg_rerank_t_disp:.4f}s")

        # save_all_runs_summary는 evaluate_single_configuration의 호출자가 필요에 따라 호출.
        # 또는 이 메서드 마지막에 호출하여 현재까지의 요약 저장. 여기서는 run_all_evaluations에서 하도록 둠.


# __main__ 블록 (evaluator_part22의 내용과 이전 __main__ 블록 통합)
if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(description="RAG 자동 평가 시스템 CLI")  # 변수명 변경
    cli_parser.add_argument(
        "--config", "-c", type=str,
        help="평가 설정 YAML 파일 경로. 예: configs/evaluation_config.yaml",
        default=None
    )
    cli_parser.add_argument(
        "--dataset", "-d", type=str,
        help="평가에 사용할 특정 데이터셋 파일 경로. 설정 파일의 dataset_paths를 오버라이드 (run_mode 'all'일 때). 'single_embedding_config' 모드에서는 필수.",
        default=None
    )
    cli_parser.add_argument(
        "--embedding_model", "-em", type=str,
        help="평가할 특정 임베딩 모델 이름. 'single_embedding_config' 모드에서 필수.",
        default=None
    )
    cli_parser.add_argument(
        "--reranker_models", "-rm", type=str,
        help="평가할 특정 리랭커 모델 이름들의 콤마(,) 구분 리스트. 'None' 문자열로 리랭커 없음을 명시. 예: 'model1,None,model2'.",
        default=None
    )
    cli_parser.add_argument(
        "--collection", type=str,
        help="테스트에 사용할 특정 Milvus 컬렉션 이름. 설정 파일 및 환경 변수를 오버라이드.",
        default=None
    )
    cli_parser.add_argument(
        "--top_k", "-k", type=int, default=None,
        help="검색 결과 수 (설정 파일 오버라이드)"
    )
    cli_parser.add_argument(
        "--threshold", "-t", type=float, default=None,
        help="유사도 임계값 (설정 파일 오버라이드)"
    )
    cli_parser.add_argument(
        "--run_mode", choices=['all', 'single_embedding_config'], default='all',
        help="'all': 모든 조합 평가, 'single_embedding_config': 지정된 임베딩 모델에 대해 리랭커 조합 평가. 기본값: 'all'."
    )

    cli_args = cli_parser.parse_args()  # 변수명 변경

    # --- 로깅 설정 (스크립트 실행 시 한 번만) ---
    if not logging.getLogger().hasHandlers():  # 핸들러가 이미 설정되지 않았을 경우에만 기본 설정
        log_dir_cli_run = PROJECT_ROOT / "logs"  # 변수명 변경
        log_dir_cli_run.mkdir(parents=True, exist_ok=True)
        log_file_cli_run = log_dir_cli_run / f"evaluator_cli_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"  # 변수명 변경

        root_logger_cli_run = logging.getLogger()  # 변수명 변경
        log_level_cli_env_run = os.getenv("LOG_LEVEL", "INFO").upper()  # 변수명 변경
        initial_log_level_cli_run = getattr(logging, log_level_cli_env_run, logging.INFO)  # 변수명 변경
        root_logger_cli_run.setLevel(initial_log_level_cli_run)

        for handler_cli_run in root_logger_cli_run.handlers[:]:  # 변수명 변경
            root_logger_cli_run.removeHandler(handler_cli_run)
            handler_cli_run.close()

        formatter_cli_run = logging.Formatter(
            '%(asctime)s - %(name)s - [%(levelname)s] - %(message)s (%(filename)s:%(lineno)d)')  # 변수명 변경

        console_handler_cli_run = logging.StreamHandler(sys.stdout)  # 변수명 변경
        console_handler_cli_run.setFormatter(formatter_cli_run)
        root_logger_cli_run.addHandler(console_handler_cli_run)

        file_handler_cli_run = logging.FileHandler(log_file_cli_run, encoding='utf-8')  # 변수명 변경
        file_handler_cli_run.setFormatter(formatter_cli_run)
        root_logger_cli_run.addHandler(file_handler_cli_run)
        logger.info(f"CLI 실행 로거 설정 완료. 로그 파일: {log_file_cli_run}")
    # --- 로깅 설정 끝 ---

    logger.info(f"AutoEvaluator CLI 실행 시작 (PID: {os.getpid()}).")
    logger.info(f"  명령줄 인자: {cli_args}")

    try:
        evaluator_instance_main = AutoEvaluator(config_path_str=cli_args.config)  # 변수명 변경

        if cli_args.run_mode == 'all':
            logger.info("CLI: 전체 평가 모드 실행 (설정 파일 기반).")
            # CLI 인자로 받은 dataset, collection을 run_all_evaluations 내부 로직에서 사용하도록
            # evaluation_specific_config를 업데이트 (선택적)
            if cli_args.dataset:
                logger.info(f"CLI 오버라이드: 데이터셋을 '{cli_args.dataset}'(으)로 제한합니다.")
                evaluator_instance_main.evaluation_specific_config.setdefault("run_settings", {})["dataset_paths"] = [
                    cli_args.dataset]
            if cli_args.collection:
                logger.info(f"CLI 오버라이드: 컬렉션을 '{cli_args.collection}'(으)로 제한합니다.")
                evaluator_instance_main.evaluation_specific_config.setdefault("run_settings", {})[
                    "collections_to_evaluate_override"] = [cli_args.collection]

            evaluator_instance_main.run_all_evaluations()

        elif cli_args.run_mode == 'single_embedding_config':
            if not cli_args.embedding_model or not cli_args.dataset:
                cli_parser.error("--run_mode 'single_embedding_config'를 사용하려면 --embedding_model과 --dataset 인자가 필수입니다.")

            logger.info(f"CLI: 단일 임베딩 구성 평가 모드 실행 ('{cli_args.embedding_model}' on '{cli_args.dataset}').")

            rerankers_list_cli = None  # 변수명 변경
            if cli_args.reranker_models:
                rerankers_list_cli = [None if r.strip().lower() == 'none' else r.strip() for r in
                                      cli_args.reranker_models.split(',')]

            run_s_cli_cfg = evaluator_instance_main.evaluation_specific_config.get("run_settings", {})  # 변수명 변경
            top_k_val_cli = cli_args.top_k if cli_args.top_k is not None else run_s_cli_cfg.get("top_k_values", [10])[
                0]  # 변수명 변경
            threshold_val_cli = cli_args.threshold if cli_args.threshold is not None else \
            run_s_cli_cfg.get("similarity_thresholds", [0.7])[0]  # 변수명 변경

            evaluator_instance_main.evaluate_single_configuration(
                embedding_model=cli_args.embedding_model,
                dataset_path=cli_args.dataset,
                top_k=top_k_val_cli,
                similarity_threshold=threshold_val_cli,
                reranker_models_list=rerankers_list_cli,
                collection_name=cli_args.collection
            )
            # 단일 구성 실행 후에도 전체 요약 저장 (all_runs_summary_list에 결과가 쌓이므로)
            evaluator_instance_main.save_all_runs_summary()

    except FileNotFoundError as fnf_e_main:  # 변수명 변경
        logger.critical(f"실행 오류 (파일 찾을 수 없음): {fnf_e_main}.", exc_info=True)
    except ValueError as val_e_main:  # 변수명 변경
        logger.critical(f"실행 오류 (값 또는 설정 문제): {val_e_main}.", exc_info=True)
    except Exception as general_e_main:  # 변수명 변경
        logger.critical(f"AutoEvaluator CLI 실행 중 예상치 못한 최상위 오류 발생: {general_e_main}", exc_info=True)

    logger.info("AutoEvaluator CLI 실행 종료.")