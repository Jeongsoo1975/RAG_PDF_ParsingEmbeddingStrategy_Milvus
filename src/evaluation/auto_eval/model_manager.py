#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
모델 관리 모듈

임베딩 모델과 재순위화 모델을 관리하고 로드/언로드합니다.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

try:
    import yaml
except ImportError:
    print("오류: PyYAML 모듈이 설치되지 않았습니다. 'pip install PyYAML'을 실행하여 설치하세요.")
    # yaml 모듈에 대한 간단한 대체 해결책
    import json
    
    class SimpleYaml:
        @staticmethod
        def safe_load(file):
            return json.load(file)
            
        @staticmethod
        def dump(data, file, default_flow_style=False, allow_unicode=True):
            return json.dump(data, file, ensure_ascii=not allow_unicode, indent=2)
    
    yaml = SimpleYaml()

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("오류: sentence-transformers 모듈이 설치되지 않았습니다. 'pip install sentence-transformers'을 실행하여 설치하세요.")
    class SentenceTransformer:
        def __init__(self, model_name):
            raise ImportError("sentence-transformers 모듈이 설치되지 않았습니다.")

logger = logging.getLogger(__name__)

class ModelManager:
    """
    임베딩 모델과 재순위화 모델을 관리하는 클래스
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        ModelManager 초기화
        
        Args:
            config_path: 설정 파일 경로
        """
        self.logger = logger
        self.logger.info("ModelManager 초기화")
        
        self.config_path = config_path or "configs/evaluation_config.yaml"
        self.embedding_model_configs = {}
        self.reranker_model_configs = {}
        self.embedding_model = None
        self.reranker_model = None
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self.embedding_model_configs = {
                model["name"]: {
                    "dimension": model.get("dimension", 768),
                    "normalize": model.get("normalize", True),
                    "qdrant_collection": model.get("qdrant_collection", "rag-collection")
                }
                for model in config.get("evaluation", {}).get("embedding_models", [])
            }
            self.reranker_model_configs = {
                model["name"]: model
                for model in config.get("evaluation", {}).get("reranker_models", [])
            }
            self.logger.info("모델 설정 로드 완료")
        except Exception as e:
            self.logger.error(f"설정 파일 로드 실패: {e}")
            raise
    
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
    
    def load_embedding_model(self, model_name: str) -> None:
        """
        임베딩 모델 로드
        
        Args:
            model_name: 로드할 임베딩 모델 이름
        """
        try:
            self.logger.info(f"임베딩 모델 로드 중: {model_name}")
            self.embedding_model = SentenceTransformer(model_name)
            self.logger.info(f"임베딩 모델 로드 완료: {model_name}")
        except Exception as e:
            self.logger.error(f"임베딩 모델 로드 실패: {model_name}. 오류: {e}")
            raise
    
    def load_reranker_model(self, model_name: Union[str, Dict[str, Any]]) -> None:
        """
        재순위화 모델 로드
        
        Args:
            model_name: 로드할 재순위화 모델 이름 또는 설정 딕셔너리
        """
        try:
            self.logger.info(f"재순위화 모델 로드 중: {model_name}")
            from sentence_transformers import CrossEncoder
            
            # 딕셔너리 형태로 전달된 경우 처리
            actual_model_name = self._get_model_name(model_name)
            self.reranker_model = CrossEncoder(actual_model_name)
            self.logger.info(f"재순위화 모델 로드 완료: {actual_model_name}")
        except Exception as e:
            self.logger.error(f"재순위화 모델 로드 실패: {model_name}. 오류: {e}")
            raise
    
    def get_embedding_dimension(self, model_name: Union[str, Dict[str, Any]]) -> int:
        """
        임베딩 모델의 차원 수 반환
        
        Args:
            model_name: 임베딩 모델 이름 또는 설정 딕셔너리
            
        Returns:
            임베딩 차원 수
        """
        # 딕셔너리 형태로 전달된 경우 처리
        model_key = self._get_model_name(model_name)
            
        # 설정에서 차원 정보 확인
        if model_key in self.embedding_model_configs:
            dim = self.embedding_model_configs[model_key].get("dimension", 768)
            return dim
        
        # 모델이 로드되어 있다면 모델에서 직접 차원 확인
        if self.embedding_model and (
            (isinstance(model_name, str) and model_name == getattr(self.embedding_model, "_model_name", "")) or
            (isinstance(model_name, dict) and model_name.get("name") == getattr(self.embedding_model, "_model_name", ""))
        ):
            try:
                return self.embedding_model.get_sentence_embedding_dimension()
            except:
                pass
                
        # 기본값 반환
        self.logger.warning(f"모델 '{model_key}'의 차원 정보를 찾을 수 없어 기본값 768을 반환합니다.")
        return 768
    
    def unload_models(self, unload_all: bool = False) -> None:
        """
        로드된 모델 언로드
        
        Args:
            unload_all: 모든 모델을 언로드할지 여부
        """
        try:
            if self.embedding_model or unload_all:
                self.embedding_model = None
                self.logger.info("임베딩 모델 언로드 완료")
            
            if self.reranker_model or unload_all:
                self.reranker_model = None
                self.logger.info("재순위화 모델 언로드 완료")
                
        except Exception as e:
            self.logger.error(f"모델 언로드 실패: {e}")
            raise
    
    def get_qdrant_collection_name(self, embedding_model: Union[str, Dict[str, Any]]) -> str:
        """
        임베딩 모델에 대응하는 Qdrant 컬렉션 이름 반환
        
        Args:
            embedding_model: 임베딩 모델 이름 또는 설정 딕셔너리
            
        Returns:
            Qdrant 컬렉션 이름
        """
        try:
            # 딕셔너리 형태로 전달된 경우 처리
            model_key = self._get_model_name(embedding_model)
                
            model_config = self.embedding_model_configs.get(model_key, {})
            collection_name = model_config.get("milvus_collection", 
                                             model_config.get("qdrant_collection", 
                                                            "insurance-embeddings"))
            return collection_name
        except Exception as e:
            self.logger.error(f"컬렉션 이름 가져오기 실패: {e}")
            return "insurance-embeddings"
    
    def rerank_results(self, query: str, docs: List[Dict[str, Any]], model_name: Union[str, Dict[str, Any]], top_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        검색 결과를 재순위화
        
        Args:
            query: 검색 쿼리
            docs: 검색 결과 문서 목록
            model_name: 재순위화 모델 이름 또는 설정 딕셔너리
            top_n: 반환할 상위 결과 수
            
        Returns:
            재순위화된 문서 목록
        """
        try:
            if not self.reranker_model:
                self.load_reranker_model(model_name)
            
            # 실제 모델 이름 추출 (로깅 목적)
            actual_model_name = self._get_model_name(model_name)
            self.logger.info(f"모델 '{actual_model_name}'로 {len(docs)}개 문서 재순위화 중...")
            
            # content 필드 또는 text 필드 사용
            pairs = []
            for doc in docs:
                # content 필드 또는 text 필드 사용
                if "content" in doc:
                    content = doc["content"]
                elif "text" in doc:
                    content = doc["text"]
                else:
                    self.logger.warning(f"문서에 content 또는 text 필드가 없습니다: {doc.keys()}")
                    content = ""
                    
                pairs.append([query, content])
            
            # 배치 크기 지정 (메모리 관리)
            scores = self.reranker_model.predict(pairs, batch_size=32)
            
            ranked_docs = []
            for doc, score in zip(docs, scores):
                doc_copy = doc.copy()
                doc_copy["score"] = float(score)
                ranked_docs.append(doc_copy)
            
            # 점수 기준 내림차순 정렬
            ranked_docs.sort(key=lambda x: x["score"], reverse=True)
            
            # top_n이 지정된 경우 해당 개수만 반환
            if top_n is not None and isinstance(top_n, int) and top_n > 0:
                ranked_docs = ranked_docs[:top_n]
                
            self.logger.info(f"재순위화 완료: {len(ranked_docs)}개 문서")
            return ranked_docs
        except Exception as e:
            self.logger.error(f"재순위화 실패: {e}")
            return docs