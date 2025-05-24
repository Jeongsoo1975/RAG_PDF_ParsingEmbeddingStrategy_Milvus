#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RAG 파이프라인과 RAGAS 평가 연동 어댑터
- 기존 RAG 시스템(retriever, generator)과 RAGAS 평가 시스템 연동
- 질문에 대해 실제 RAG 답변을 생성하고 RAGAS 형식으로 변환
- 다양한 모델 조합에 대한 배치 처리 지원
"""

import os
import sys
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import traceback

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.abspath('.'))

# 기존 RAG 시스템 임포트
from src.rag.retriever import DocumentRetriever
from src.rag.generator import ResponseGenerator
from src.utils.config import Config

# RAGAS 관련 임포트
from src.evaluation.data.ragas_dataset_converter import RAGASDatasetItem

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/rag_pipeline_adapter.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("rag_pipeline_adapter")

@dataclass
class RAGPipelineConfig:
    """RAG 파이프라인 실행 설정"""
    # 검색 관련 설정
    top_k: int = 5
    similarity_threshold: float = 0.5
    use_parent_chunks: bool = True
    enable_query_optimization: bool = True
    filter_expr: str = "default"
    
    # 생성 관련 설정
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.3
    max_tokens: int = 1024
    response_language: str = "ko"
    
    # 대상 컬렉션
    target_collections: Optional[List[str]] = None
    
    # 배치 처리 설정
    batch_size: int = 10
    delay_between_requests: float = 1.0

@dataclass
class RAGEvaluationResult:
    """RAG 평가 결과"""
    question_id: str
    question: str
    retrieved_contexts: List[str]
    generated_answer: str
    ground_truth: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    error: Optional[str] = None

class RAGPipelineAdapter:
    """RAG 파이프라인과 RAGAS 평가 연동 어댑터"""
    
    def __init__(self, config: Optional[Config] = None):
        """
        RAG 파이프라인 어댑터 초기화
        
        Args:
            config: 설정 객체
        """
        self.config = config or Config()
        self.logger = logger
        
        # RAG 컴포넌트 초기화
        try:
            self.retriever = DocumentRetriever(self.config)
            self.generator = ResponseGenerator(self.config)
            self.logger.info("RAG 컴포넌트 초기화 완료")
        except Exception as e:
            self.logger.error(f"RAG 컴포넌트 초기화 실패: {e}")
            self.retriever = None
            self.generator = None
        
        # 결과 저장 경로
        self.output_dir = "evaluation_results/rag_pipeline_results"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def execute_rag_pipeline(self, 
                           question: str, 
                           ground_truth: str,
                           question_id: Optional[str] = None,
                           pipeline_config: Optional[RAGPipelineConfig] = None) -> RAGEvaluationResult:
        """
        단일 질문에 대해 RAG 파이프라인 실행
        
        Args:
            question: 질문
            ground_truth: 정답
            question_id: 질문 ID
            pipeline_config: 파이프라인 설정
            
        Returns:
            RAG 평가 결과
        """
        start_time = time.time()
        config = pipeline_config or RAGPipelineConfig()
        q_id = question_id or f"q_{int(time.time())}"
        
        self.logger.info(f"RAG 파이프라인 실행 시작: {q_id}")
        
        try:
            # 1. 문서 검색 (Retrieval)
            if not self.retriever:
                raise Exception("DocumentRetriever가 초기화되지 않았습니다.")
            
            retrieved_docs = self.retriever.retrieve(
                query=question,
                top_k=config.top_k,
                threshold=config.similarity_threshold,
                target_collections=config.target_collections,
                use_parent_chunks=config.use_parent_chunks,
                enable_query_optimization=config.enable_query_optimization,
                force_filter_expr=config.filter_expr
            )
            
            if not retrieved_docs:
                self.logger.warning(f"질문 '{question}'에 대한 검색 결과가 없습니다.")
                retrieved_contexts = ["검색된 문서가 없습니다."]
                generated_answer = "죄송합니다. 관련 정보를 찾을 수 없습니다."
            else:
                # 검색된 문서에서 컨텍스트 추출
                retrieved_contexts = [doc.get("content", "") for doc in retrieved_docs if doc.get("content")]
                
                # 2. 응답 생성 (Generation)
                if not self.generator:
                    raise Exception("ResponseGenerator가 초기화되지 않았습니다.")
                
                generated_answer = self.generator.generate(
                    query=question,
                    retrieved_docs=retrieved_docs
                )
            
            processing_time = time.time() - start_time
            
            # 결과 생성
            result = RAGEvaluationResult(
                question_id=q_id,
                question=question,
                retrieved_contexts=retrieved_contexts,
                generated_answer=generated_answer,
                ground_truth=ground_truth,
                metadata={
                    "num_retrieved_docs": len(retrieved_docs),
                    "pipeline_config": {
                        "top_k": config.top_k,
                        "similarity_threshold": config.similarity_threshold,
                        "use_parent_chunks": config.use_parent_chunks,
                        "model": config.model
                    },
                    "retrieval_scores": [doc.get("similarity", 0.0) for doc in retrieved_docs]
                },
                processing_time=processing_time
            )
            
            self.logger.info(f"RAG 파이프라인 완료: {q_id} (소요시간: {processing_time:.2f}초)")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"RAG 파이프라인 실행 실패: {str(e)}"
            self.logger.error(error_msg)
            
            return RAGEvaluationResult(
                question_id=q_id,
                question=question,
                retrieved_contexts=[],
                generated_answer="오류로 인해 답변을 생성할 수 없습니다.",
                ground_truth=ground_truth,
                processing_time=processing_time,
                error=error_msg
            )
    
    def process_evaluation_dataset(self, 
                                 dataset_path: str,
                                 output_path: Optional[str] = None,
                                 pipeline_config: Optional[RAGPipelineConfig] = None) -> Tuple[List[RAGEvaluationResult], str]:
        """
        평가 데이터셋 전체에 대해 RAG 파이프라인 실행
        
        Args:
            dataset_path: 평가 데이터셋 파일 경로
            output_path: 결과 저장 경로
            pipeline_config: 파이프라인 설정
            
        Returns:
            (RAG 평가 결과 목록, RAGAS 형식 데이터셋 파일 경로)
        """
        config = pipeline_config or RAGPipelineConfig()
        
        # 데이터셋 로드
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            questions = dataset.get('questions', [])
            self.logger.info(f"평가 데이터셋 로드 완료: {len(questions)}개 질문")
        except Exception as e:
            self.logger.error(f"데이터셋 로드 실패: {e}")
            return [], ""
        
        # 결과 저장 경로 설정
        if output_path is None:
            timestamp = int(time.time())
            dataset_name = Path(dataset_path).stem
            output_path = os.path.join(self.output_dir, f"{dataset_name}_rag_results_{timestamp}.json")
        
        # 배치 처리
        rag_results = []
        total_questions = len(questions)
        
        self.logger.info(f"RAG 파이프라인 배치 처리 시작: {total_questions}개 질문")
        
        for i, question_data in enumerate(questions):
            question_text = question_data.get('text', '')
            question_id = question_data.get('id', f'q_{i+1}')
            ground_truth = question_data.get('gold_standard', {}).get('answer', '')
            
            self.logger.info(f"처리 중: {i+1}/{total_questions} - {question_id}")
            
            # RAG 파이프라인 실행
            result = self.execute_rag_pipeline(
                question=question_text,
                ground_truth=ground_truth,
                question_id=question_id,
                pipeline_config=config
            )
            
            rag_results.append(result)
            
            # 요청 간 지연 (API 레이트 리미트 방지)
            if config.delay_between_requests > 0 and i < total_questions - 1:
                time.sleep(config.delay_between_requests)
        
        # 결과 저장
        self._save_rag_results(rag_results, output_path)
        
        # RAGAS 형식으로 변환 및 저장
        ragas_dataset_path = self._convert_to_ragas_format(rag_results, output_path)
        
        self.logger.info(f"RAG 파이프라인 배치 처리 완료: {len(rag_results)}개 결과")
        return rag_results, ragas_dataset_path
    
    def _save_rag_results(self, results: List[RAGEvaluationResult], output_path: str):
        """RAG 실행 결과 저장"""
        try:
            # dataclass를 딕셔너리로 변환
            results_dict = {
                "evaluation_info": {
                    "total_questions": len(results),
                    "successful_results": len([r for r in results if r.error is None]),
                    "failed_results": len([r for r in results if r.error is not None]),
                    "total_processing_time": sum(r.processing_time for r in results),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                },
                "results": []
            }
            
            for result in results:
                result_dict = {
                    "question_id": result.question_id,
                    "question": result.question,
                    "retrieved_contexts": result.retrieved_contexts,
                    "generated_answer": result.generated_answer,
                    "ground_truth": result.ground_truth,
                    "metadata": result.metadata,
                    "processing_time": result.processing_time,
                    "error": result.error
                }
                results_dict["results"].append(result_dict)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results_dict, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"RAG 결과 저장 완료: {output_path}")
            
        except Exception as e:
            self.logger.error(f"RAG 결과 저장 실패: {e}")
    
    def _convert_to_ragas_format(self, results: List[RAGEvaluationResult], rag_output_path: str) -> str:
        """RAG 결과를 RAGAS 형식으로 변환"""
        try:
            # RAGAS 형식 데이터 생성
            ragas_data = {
                "questions": [],
                "contexts": [],
                "ground_truths": [],
                "answers": [],
                "ids": [],
                "metadata": []
            }
            
            for result in results:
                if result.error is None:  # 성공한 결과만 포함
                    ragas_data["questions"].append(result.question)
                    ragas_data["contexts"].append(result.retrieved_contexts)
                    ragas_data["ground_truths"].append(result.ground_truth)
                    ragas_data["answers"].append(result.generated_answer)
                    ragas_data["ids"].append(result.question_id)
                    ragas_data["metadata"].append(result.metadata)
            
            # RAGAS 데이터셋 저장
            ragas_output_path = rag_output_path.replace('_rag_results_', '_ragas_dataset_').replace('.json', '_ragas.json')
            
            with open(ragas_output_path, 'w', encoding='utf-8') as f:
                json.dump(ragas_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"RAGAS 형식 데이터셋 저장 완료: {ragas_output_path}")
            return ragas_output_path
            
        except Exception as e:
            self.logger.error(f"RAGAS 형식 변환 실패: {e}")
            return ""
    
    def test_rag_pipeline_connectivity(self) -> Dict[str, Any]:
        """RAG 파이프라인 연결 상태 테스트"""
        test_results = {
            "retriever_status": "unknown",
            "generator_status": "unknown",
            "collections_available": [],
            "test_query_result": None,
            "error_messages": []
        }
        
        # Retriever 테스트
        try:
            if self.retriever:
                test_results["retriever_status"] = "connected"
                test_results["collections_available"] = self.retriever.collections
                
                # 간단한 검색 테스트
                test_docs = self.retriever.retrieve(
                    query="테스트 질문",
                    top_k=2,
                    threshold=0.1  # 낮은 임계값으로 설정
                )
                test_results["test_retrieval_count"] = len(test_docs)
            else:
                test_results["retriever_status"] = "not_initialized"
        except Exception as e:
            test_results["retriever_status"] = "error"
            test_results["error_messages"].append(f"Retriever test failed: {e}")
        
        # Generator 테스트
        try:
            if self.generator:
                test_results["generator_status"] = "connected"
                
                # 간단한 생성 테스트 (더미 문서 사용)
                dummy_docs = [{
                    "content": "이것은 테스트용 문서입니다.",
                    "metadata": {"source": "test"}
                }]
                
                test_response = self.generator.generate(
                    query="테스트 질문입니다.",
                    retrieved_docs=dummy_docs
                )
                test_results["test_query_result"] = {
                    "response_length": len(test_response),
                    "response_preview": test_response[:100] + "..." if len(test_response) > 100 else test_response
                }
            else:
                test_results["generator_status"] = "not_initialized"
        except Exception as e:
            test_results["generator_status"] = "error"
            test_results["error_messages"].append(f"Generator test failed: {e}")
        
        return test_results

def test_adapter_functionality():
    """어댑터 기능 테스트"""
    print("=== RAG 파이프라인 어댑터 테스트 ===")
    
    try:
        # 어댑터 초기화
        adapter = RAGPipelineAdapter()
        
        # 연결 상태 테스트
        print("\n1. RAG 파이프라인 연결 상태 테스트")
        connectivity_result = adapter.test_rag_pipeline_connectivity()
        
        for key, value in connectivity_result.items():
            print(f"  - {key}: {value}")
        
        # 단일 질문 테스트
        print("\n2. 단일 질문 RAG 파이프라인 테스트")
        test_question = "제1보험기간이란 무엇인가요?"
        test_ground_truth = "제1보험기간은 계약일부터 80세 계약해당일 전일까지입니다."
        
        config = RAGPipelineConfig(
            top_k=3,
            similarity_threshold=0.3,
            model="gpt-3.5-turbo"
        )
        
        result = adapter.execute_rag_pipeline(
            question=test_question,
            ground_truth=test_ground_truth,
            question_id="test_001",
            pipeline_config=config
        )
        
        print(f"  질문: {result.question}")
        print(f"  검색된 컨텍스트 수: {len(result.retrieved_contexts)}")
        print(f"  생성된 답변: {result.generated_answer[:100]}...")
        print(f"  처리 시간: {result.processing_time:.2f}초")
        if result.error:
            print(f"  오류: {result.error}")
        
        print("\n✅ RAG 파이프라인 어댑터 기본 기능 테스트 완료!")
        
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_adapter_functionality()
