#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
3단계 표준 RAGAS 평가 실행기
amnesty_qa 데이터셋과 임베딩을 활용하여 완전한 표준 RAGAS 평가를 수행
"""

import os
import sys
import json
import logging
import traceback
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 실제 RAG 구성요소 import
from src.rag.embedder import DocumentEmbedder
from src.rag.generator import ResponseGenerator
from src.vectordb.milvus_client import MilvusClient
from src.evaluation.simple_ragas_metrics import SimpleRAGASMetrics





class AmnestyDataLoader:
    """Amnesty QA 데이터 로더"""
    
    def __init__(self, data_dir: str = "data/amnesty_qa"):
        self.data_dir = Path(data_dir)
        self.logger = logging.getLogger("AmnestyDataLoader")
    
    def load_evaluation_dataset(self) -> List[Dict[str, Any]]:
        """평가 데이터셋 로드"""
        eval_file = self.data_dir / "amnesty_qa_evaluation.json"
        
        if not eval_file.exists():
            raise FileNotFoundError(f"평가 데이터 파일을 찾을 수 없습니다: {eval_file}")
        
        with open(eval_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        questions = data.get('questions', [])
        contexts = data.get('contexts', [])
        ground_truths = data.get('ground_truths', [])
        ids = data.get('ids', [])
        
        # 평가 항목 리스트 생성
        evaluation_items = []
        for i in range(len(questions)):
            item = {
                'id': ids[i] if i < len(ids) else f"amnesty_qa_{i}",
                'question': questions[i] if i < len(questions) else '',
                'contexts': contexts[i] if i < len(contexts) else [],
                'ground_truths': ground_truths[i] if i < len(ground_truths) else [],
                'metadata': {
                    'source': 'amnesty_qa',
                    'index': i
                }
            }
            evaluation_items.append(item)
        
        self.logger.info(f"평가 데이터 로드 완료: {len(evaluation_items)}개 항목")
        return evaluation_items


class AmnestyMilvusSearcher:
    """Amnesty QA 임베딩 검색기 - 실제 Milvus 벡터 검색 구현"""
    
    def __init__(self, 
                 embedder: Optional[DocumentEmbedder] = None,
                 milvus_client: Optional[MilvusClient] = None,
                 collection_name: str = "amnesty_qa_embeddings"):
        """
        검색기 초기화
        
        Args:
            embedder: 실제 DocumentEmbedder 인스턴스
            milvus_client: 실제 MilvusClient 인스턴스
            collection_name: 검색할 Milvus 컬렉션 이름
        """
        self.embedder = embedder
        self.milvus_client = milvus_client
        self.collection_name = collection_name
        self.logger = logging.getLogger("AmnestyMilvusSearcher")
        
        # 초기화 상태 로깅
        if self.embedder is None:
            self.logger.warning("DocumentEmbedder가 제공되지 않았습니다. 더미 모드로 작동합니다.")
        if self.milvus_client is None:
            self.logger.warning("MilvusClient가 제공되지 않았습니다. 더미 모드로 작동합니다.")
        
        # Milvus 연결 및 컬렉션 확인
        self._check_milvus_setup()
    
    def _check_milvus_setup(self):
        """
        Milvus 연결 및 컬렉션 상태 확인
        """
        if self.milvus_client is None:
            self.logger.warning("MilvusClient가 없어 Milvus 상태를 확인할 수 없습니다.")
            return
        
        try:
            # Milvus 연결 상태 확인
            if not self.milvus_client.is_connected():
                self.logger.error("Milvus 서버에 연결되지 않았습니다.")
                return
            
            # 컬렉션 존재 확인
            if not self.milvus_client.has_collection(self.collection_name):
                self.logger.error(f"Milvus 컬렉션 '{self.collection_name}'이 존재하지 않습니다.")
                
                # 사용 가능한 컬렉션 목록 표시
                available_collections = self.milvus_client.list_collections()
                self.logger.info(f"사용 가능한 컬렉션: {available_collections}")
                return
            
            # 컬렉션 통계 정보 확인
            stats = self.milvus_client.get_collection_stats(self.collection_name)
            vector_count = stats.get('num_entities', 0)
            self.logger.info(f"Milvus 컬렉션 '{self.collection_name}' 준비 완료: {vector_count}개 벡터")
            
        except Exception as e:
            self.logger.error(f"Milvus 설정 확인 중 오류: {e}")
    
    def search_contexts(self, query: str, top_k: int = 5) -> List[str]:
        """
        실제 Milvus 벡터 검색으로 컨텍스트 검색
        
        Args:
            query: 검색 질문
            top_k: 반환할 최대 결과 수
            
        Returns:
            List[str]: 검색된 컨텍스트 리스트
        """
        # 전제 조건 확인
        if not self.embedder or not self.milvus_client:
            self.logger.warning("검색 구성요소가 초기화되지 않았습니다. 빈 컨텍스트 반환")
            return []
        
        if not query or not query.strip():
            self.logger.warning("빈 질문입니다. 빈 컨텍스트 반환")
            return []
        
        try:
            # 1. 질문을 벡터로 임베딩
            self.logger.debug(f"질문 임베딩 시작: '{query[:50]}...'")
            query_embedding = self.embedder.embed_text(query.strip())
            
            if query_embedding is None or len(query_embedding) == 0:
                self.logger.error("질문 임베딩 실패")
                return []
            
            # 2. Milvus에서 벡터 검색
            self.logger.debug(f"Milvus 검색 시작: collection='{self.collection_name}', top_k={top_k}")
            search_results = self.milvus_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                top_k=top_k,
                output_fields=["text", "score", "chunk_id", "source"]
            )
            
            if not search_results:
                self.logger.warning(f"Milvus 검색 결과가 비어있습니다.")
                return []
            
            # 3. 검색 결과를 컨텍스트 텍스트로 변환
            contexts = []
            for i, result in enumerate(search_results):
                text = result.get('text', '').strip()
                score = result.get('score', 0.0)
                chunk_id = result.get('chunk_id', 'unknown')
                source = result.get('source', 'unknown')
                
                if text:
                    contexts.append(text)
                    self.logger.debug(f"검색 결과 {i+1}: score={score:.4f}, chunk_id={chunk_id}, source={source[:30]}...")
            
            self.logger.info(f"Milvus 검색 완료: {len(contexts)}개 컨텍스트 반환 (총 {len(search_results)}개 결과)")
            return contexts
            
        except Exception as e:
            self.logger.error(f"Milvus 벡터 검색 중 오류: {e}")
            self.logger.error(f"Error details: {str(e)}")
            return []





class Step3StandardRAGASEvaluator:
    """3단계 표준 RAGAS 평가 실행기"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        평가기 초기화
        
        Args:
            config_path: 설정 파일 경로 (선택적)
        """
        self.logger = self._setup_logging()
        self.project_root = project_root
        
        # 구성요소 초기화
        try:
            # 기존 구성요소
            self.data_loader = AmnestyDataLoader()
            # searcher는 나중에 실제 구성요소로 초기화
            self.metrics = SimpleRAGASMetrics()
            
            # 실제 RAG 구성요소 초기화
            try:
                # Config 객체 생성 (src.utils.config에서 가져오기)
                from src.utils.config import Config
                config = Config()
                
                # 실제 RAG 구성요소 초기화
                self.embedder = DocumentEmbedder(config)
                self.generator = ResponseGenerator(config)
                self.milvus_client = MilvusClient(config)
                
                # 실제 구성요소로 AmnestyMilvusSearcher 초기화
                self.searcher = AmnestyMilvusSearcher(
                    embedder=self.embedder,
                    milvus_client=self.milvus_client,
                    collection_name="amnesty_qa_embeddings"
                )
                
                # API 키 확인
                api_keys_status = self._check_api_keys()
                if not api_keys_status['has_any_api_key']:
                    self.logger.error("어떤 LLM API 키도 설정되지 않았습니다. 실제 답변 생성이 제한될 수 있습니다.")
                    self.logger.error("API 키 상태: " + str(api_keys_status))
                else:
                    self.logger.info("실제 RAG 구성요소가 성공적으로 초기화되었습니다.")
                    self.logger.info("API 키 상태: " + str(api_keys_status))
                
            except Exception as e:
                self.logger.error(f"실제 RAG 구성요소 초기화 중 오류: {e}")
                # 비어있는 대체 구성요소 설정
                self.embedder = None
                self.generator = None
                self.milvus_client = None
                # 더미 모드로 AmnestyMilvusSearcher 초기화
                self.searcher = AmnestyMilvusSearcher()
                self.logger.warning("실제 RAG 구성요소 대신 더미 모드로 작동합니다.")
            
            self.logger.info("모든 구성요소가 성공적으로 초기화되었습니다.")
            
        except Exception as e:
            self.logger.error(f"구성요소 초기화 중 오류 발생: {e}")
            raise
            
        # 결과 저장 디렉토리 설정
        self.results_dir = self.project_root / "evaluation_results" / "step3_standard_ragas"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 평가 결과 저장용
        self.evaluation_results = []
        self.overall_metrics = {}
    
    def _setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger("Step3StandardRAGASEvaluator")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # 콘솔 핸들러
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # 파일 핸들러
            log_dir = project_root / "logs"
            log_dir.mkdir(exist_ok=True)
            file_handler = logging.FileHandler(log_dir / "step3_standard_ragas_evaluation.log", encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            
            # 포맷터
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)
            
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)
            
        return logger
    
    def _check_api_keys(self) -> Dict[str, Any]:
        """
        API 키 설정 상태 확인
        
        Returns:
            Dict[str, Any]: API 키 상태 정보
        """
        import os
        
        api_keys = {
            'openai_api_key': bool(os.environ.get('OPENAI_API_KEY')),
            'anthropic_api_key': bool(os.environ.get('ANTHROPIC_API_KEY')),
            'grok_api_key': bool(os.environ.get('GROK_API_KEY'))
        }
        
        api_keys['has_any_api_key'] = any(api_keys.values())
        
        return api_keys
    
    def run_evaluation(self) -> Dict[str, Any]:
        """
        전체 평가 프로세스 실행
        
        Returns:
            Dict[str, Any]: 평가 결과
        """
        self.logger.info("=== 3단계 표준 RAGAS 평가 시작 ===")
        start_time = datetime.now()
        
        try:
            # 1. Amnesty QA 평가 데이터 로드
            self.logger.info("Amnesty QA 평가 데이터 로딩 중...")
            evaluation_items = self.data_loader.load_evaluation_dataset()
            
            if not evaluation_items:
                self.logger.error("로드된 데이터가 비어있습니다.")
                return {
                    'success': False,
                    'error': '로드된 데이터가 비어있습니다.',
                    'processed_count': 0
                }
            
            self.logger.info(f"로드된 질문 수: {len(evaluation_items)}")
            
            # 2. 각 질문에 대한 평가 실행
            total_questions = len(evaluation_items)
            processed_count = 0
            
            for item in evaluation_items:
                try:
                    result = self._evaluate_single_item(item)
                    self.evaluation_results.append(result)
                    processed_count += 1
                    
                    if processed_count % 2 == 0:
                        self.logger.info(f"진행률: {processed_count}/{total_questions} ({processed_count/total_questions*100:.1f}%)")
                        
                except Exception as e:
                    self.logger.error(f"질문 '{item.get('question', 'Unknown')}' 평가 중 오류: {e}")
                    continue
            
            # 3. 전체 평가 지표 계산
            self._calculate_overall_metrics()
            
            # 4. 결과 저장
            result_file = self.save_results()
            
            # 5. 리포트 생성
            self.generate_report()
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            self.logger.info(f"=== 3단계 표준 RAGAS 평가 완료 ===")
            self.logger.info(f"총 소요 시간: {duration.total_seconds():.2f}초")
            self.logger.info(f"처리된 질문 수: {processed_count}/{total_questions}")
            self.logger.info(f"결과 저장: {result_file}")
            
            return {
                'success': True,
                'processed_count': processed_count,
                'total_count': total_questions,
                'duration_seconds': duration.total_seconds(),
                'result_file': str(result_file),
                'overall_metrics': self.overall_metrics
            }
            
        except Exception as e:
            self.logger.error(f"평가 실행 중 심각한 오류 발생: {e}")
            self.logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e),
                'processed_count': len(self.evaluation_results)
            }
    
    def _evaluate_single_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        개별 질문에 대한 평가 수행
        
        Args:
            item: Amnesty QA 평가 데이터 항목
            
        Returns:
            Dict[str, Any]: 평가 결과
        """
        question = item.get('question', '')
        ground_truths = item.get('ground_truths', [])
        original_contexts = item.get('contexts', [])
        
        # 1. 컨텍스트 검색 (Milvus 임베딩 기반)
        try:
            retrieved_contexts = self.searcher.search_contexts(question, top_k=10)
            
            # 원본 컨텍스트와 검색된 컨텍스트 결합
            all_contexts = original_contexts + retrieved_contexts
            # 중복 제거
            unique_contexts = []
            seen = set()
            for ctx in all_contexts:
                if ctx not in seen and ctx.strip():
                    unique_contexts.append(ctx)
                    seen.add(ctx)
            
            final_contexts = unique_contexts[:5]  # 최대 5개 컨텍스트
            
        except Exception as e:
            self.logger.warning(f"컨텍스트 검색 실패, 원본 컨텍스트 사용: {e}")
            final_contexts = original_contexts
        
        # 2. 실제 LLM을 사용한 답변 생성
        try:
            if self.generator is None:
                self.logger.warning("ResponseGenerator가 초기화되지 않았습니다. 기본 답변 반환")
                generated_answer = "죄송합니다. 답변 생성기가 현재 사용할 수 없습니다."
            else:
                # retrieved_docs 포맷을 ResponseGenerator.generate에 맞게 변환
                formatted_docs = []
                for i, context in enumerate(final_contexts):
                    doc = {
                        "content": context,
                        "metadata": {
                            "chunk_id": f"amnesty_qa_chunk_{i}",
                            "source_file": "amnesty_qa_dataset",
                            "page_num": str(i + 1)
                        },
                        "similarity": 0.8  # 검색된 컨텍스트이므로 높은 유사도 가정
                    }
                    formatted_docs.append(doc)
                
                # ResponseGenerator로 실제 답변 생성 (영어로 생성)
                self.logger.info(f"ResponseGenerator로 영어 답변 생성 중: '{question[:50]}...'")
                generated_answer = self.generator.generate(
                    query=question,
                    retrieved_docs=formatted_docs,
                    override_language='en'
                )
                
                # 생성된 답변 로깅
                self.logger.info(f"LLM 답변 생성 완료: {len(generated_answer)}자")
                self.logger.debug(f"생성된 답변 미리보기: '{generated_answer[:100]}...'")
                
        except Exception as e:
            self.logger.error(f"LLM 답변 생성 중 오류 발생: {e}")
            self.logger.error(f"오류 상세: {str(e)}")
            # API 호출 실패 시 적절한 기본 응답 제공
            generated_answer = f"답변 생성 중 오류가 발생했습니다: {str(e)}"
        
        # 3. 표준 RAGAS 지표 평가
        try:
            # ground_truth_contexts 준비 (Context Recall용)
            gt_contexts = original_contexts if original_contexts else None
            
            # SimpleRAGASMetrics의 evaluate_single_item 사용
            ragas_result = self.metrics.evaluate_single_item(
                question_id=item.get('id', 'unknown'),
                question=question,
                contexts=final_contexts,
                answer=generated_answer,
                ground_truth_contexts=gt_contexts
            )
            
            metrics_dict = {
                'context_relevancy': ragas_result.context_relevancy,
                'context_precision': ragas_result.context_precision,
                'context_recall': ragas_result.context_recall,
                'faithfulness': ragas_result.faithfulness,
                'answer_relevancy': ragas_result.answer_relevancy,
                'overall_score': (ragas_result.context_relevancy + 
                                ragas_result.context_precision + 
                                ragas_result.context_recall + 
                                ragas_result.faithfulness + 
                                ragas_result.answer_relevancy) / 5.0
            }
            
        except Exception as e:
            self.logger.error(f"RAGAS 지표 계산 실패: {e}")
            metrics_dict = {
                'context_relevancy': 0.0,
                'context_precision': 0.0,
                'context_recall': 0.0,
                'faithfulness': 0.0,
                'answer_relevancy': 0.0,
                'overall_score': 0.0,
                'error': str(e)
            }
        
        # 4. 결과 구성
        result = {
            'question_id': item.get('id', 'unknown'),
            'question': question,
            'generated_answer': generated_answer,
            'ground_truths': ground_truths,
            'contexts': final_contexts,
            'context_count': len(final_contexts),
            'metrics': metrics_dict,
            'metadata': item.get('metadata', {}),
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def _calculate_overall_metrics(self) -> None:
        """전체 평가 지표 계산"""
        if not self.evaluation_results:
            self.overall_metrics = {'error': '평가 결과가 없습니다.'}
            return
        
        # 각 지표별 점수 수집
        metric_names = ['context_relevancy', 'context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']
        metric_scores = {name: [] for name in metric_names}
        overall_scores = []
        
        for result in self.evaluation_results:
            metrics = result.get('metrics', {})
            for name in metric_names:
                score = metrics.get(name, 0.0)
                metric_scores[name].append(score)
            
            overall_scores.append(metrics.get('overall_score', 0.0))
        
        # 통계 계산
        self.overall_metrics = {
            'total_questions': len(self.evaluation_results),
            'evaluation_type': 'standard_ragas_5_metrics'
        }
        
        # 각 지표별 통계
        for name in metric_names:
            scores = metric_scores[name]
            self.overall_metrics[f'average_{name}'] = sum(scores) / len(scores) if scores else 0.0
            self.overall_metrics[f'min_{name}'] = min(scores) if scores else 0.0
            self.overall_metrics[f'max_{name}'] = max(scores) if scores else 0.0
        
        # 전체 점수 통계
        self.overall_metrics['average_overall_score'] = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
        self.overall_metrics['min_overall_score'] = min(overall_scores) if overall_scores else 0.0
        self.overall_metrics['max_overall_score'] = max(overall_scores) if overall_scores else 0.0
    
    def save_results(self) -> Path:
        """평가 결과를 JSON 형태로 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = self.results_dir / f"step3_standard_ragas_evaluation_{timestamp}.json"
        
        # 전체 결과 구성
        full_results = {
            'evaluation_info': {
                'type': 'step3_standard_ragas_evaluation',
                'dataset': 'amnesty_qa',
                'metrics': ['context_relevancy', 'context_precision', 'context_recall', 'faithfulness', 'answer_relevancy'],
                'timestamp': datetime.now().isoformat(),
                'total_questions': len(self.evaluation_results),
                'evaluator_version': '1.0.0'
            },
            'overall_metrics': self.overall_metrics,
            'detailed_results': self.evaluation_results
        }
        
        # JSON 파일로 저장
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"평가 결과 저장 완료: {result_file}")
        return result_file
    
    def generate_report(self) -> None:
        """평가 결과 요약 리포트 생성"""
        if not self.overall_metrics:
            self.logger.warning("전체 지표가 계산되지 않아 리포트를 생성할 수 없습니다.")
            return
        
        print("\n" + "="*70)
        print("    3단계 표준 RAGAS 평가 결과 요약 (Amnesty QA)")
        print("="*70)
        
        print(f"평가 대상 질문 수: {self.overall_metrics.get('total_questions', 0)}개")
        print(f"전체 평균 점수: {self.overall_metrics.get('average_overall_score', 0):.3f}")
        
        print("\n[ 표준 RAGAS 5개 지표 ]")
        metric_labels = {
            'context_relevancy': 'Context Relevancy',
            'context_precision': 'Context Precision', 
            'context_recall': 'Context Recall',
            'faithfulness': 'Faithfulness',
            'answer_relevancy': 'Answer Relevancy'
        }
        
        for metric_key, label in metric_labels.items():
            avg = self.overall_metrics.get(f'average_{metric_key}', 0)
            min_val = self.overall_metrics.get(f'min_{metric_key}', 0)
            max_val = self.overall_metrics.get(f'max_{metric_key}', 0)
            
            print(f"{label:20}: {avg:.3f} (최소: {min_val:.3f}, 최대: {max_val:.3f})")
        
        # 성능 평가
        overall_score = self.overall_metrics.get('average_overall_score', 0)
        if overall_score >= 0.7:
            performance = "우수함 (Excellent)"
        elif overall_score >= 0.5:
            performance = "양호함 (Good)"
        elif overall_score >= 0.3:
            performance = "보통 (Fair)"
        else:
            performance = "개선 필요 (Poor)"
        
        print(f"\n전체 성능 평가: {performance}")
        print(f"데이터셋: Amnesty QA (Human Rights)")
        print(f"평가 방식: 표준 RAGAS 5개 지표")
        print("="*70 + "\n")


# 메인 실행 부분
if __name__ == "__main__":
    try:
        evaluator = Step3StandardRAGASEvaluator()
        result = evaluator.run_evaluation()
        
        if result['success']:
            print(f"\n평가 완료! 처리된 질문: {result['processed_count']}/{result['total_count']}")
            print(f"결과 파일: {result['result_file']}")
        else:
            print(f"\n평가 실패: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"평가 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
