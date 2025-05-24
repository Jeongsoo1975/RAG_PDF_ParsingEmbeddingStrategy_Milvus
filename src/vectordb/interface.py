#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
벡터 데이터베이스 인터페이스 모듈.
다양한 벡터 DB(Pinecone, Milvus 등)를 지원하기 위한 추상화 인터페이스 제공.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
import logging

from src.utils.logger import get_logger

logger = get_logger("vectordb_interface")

class VectorDBInterface(ABC):
    """
    벡터 데이터베이스 인터페이스 추상 클래스.
    모든 벡터 DB 구현체는 이 인터페이스를 상속해야 합니다.
    """
    
    @abstractmethod
    def connect(self) -> bool:
        """
        벡터 데이터베이스에 연결합니다.
        
        Returns:
            bool: 연결 성공 여부
        """
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """
        현재 벡터 데이터베이스 연결 상태를 확인합니다.
        
        Returns:
            bool: 연결되어 있으면 True, 아니면 False
        """
        pass
    
    @abstractmethod
    def create_collection(self, collection_name: str, dimension: int, metric_type: str = "COSINE") -> bool:
        """
        벡터 컬렉션(인덱스)을 생성합니다.
        
        Args:
            collection_name: 생성할 컬렉션 이름
            dimension: 벡터 차원 수
            metric_type: 유사도 측정 방식 ("COSINE", "L2", "IP" 등)
            
        Returns:
            bool: 생성 성공 여부
        """
        pass
    
    @abstractmethod
    def has_collection(self, collection_name: str) -> bool:
        """
        지정한 이름의 컬렉션이 존재하는지 확인합니다.
        
        Args:
            collection_name: 확인할 컬렉션 이름
            
        Returns:
            bool: 컬렉션 존재 여부
        """
        pass
    
    @abstractmethod
    def list_collections(self) -> List[str]:
        """
        사용 가능한 모든 컬렉션 이름 목록을 반환합니다.
        
        Returns:
            List[str]: 컬렉션 이름 목록
        """
        pass
    
    @abstractmethod
    def drop_collection(self, collection_name: str) -> bool:
        """
        지정한 컬렉션을 삭제합니다.
        
        Args:
            collection_name: 삭제할 컬렉션 이름
            
        Returns:
            bool: 삭제 성공 여부
        """
        pass
    
    @abstractmethod
    def insert(self, collection_name: str, ids: List[str], vectors: List[List[float]], metadata: List[Dict[str, Any]], partition: Optional[str] = None, force_flush: bool = False) -> int:
        """
        벡터와 메타데이터를 컬렉션에 삽입합니다.
        
        Args:
            collection_name: 삽입할 컬렉션 이름
            ids: 각 벡터의 고유 ID 목록
            vectors: 임베딩 벡터 목록 (2D 배열)
            metadata: 각 벡터에 대한 메타데이터 목록
            partition: 삽입할 파티션 이름 (기본값: None, _default 파티션 사용)
            force_flush: 데이터 영속화를 위해 즉시 플러시 (기본값: False)
            
        Returns:
            int: 삽입된 벡터 수
        """
        pass
    
    @abstractmethod
    def search(self, 
              collection_name: str, 
              query_vector: List[float], 
              top_k: int = 5, 
              filter_expr: Optional[str] = None,
              partition: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        쿼리 벡터와 유사한 벡터를 검색합니다.
        
        Args:
            collection_name: 검색할 컬렉션 이름
            query_vector: 쿼리 벡터
            top_k: 반환할 최대 결과 수
            filter_expr: 메타데이터 필터링 표현식
            partition: 검색할 파티션(네임스페이스)
            
        Returns:
            List[Dict[str, Any]]: 검색 결과 목록
        """
        pass
    
    @abstractmethod
    def count(self, collection_name: str, filter_expr: Optional[str] = None) -> int:
        """
        컬렉션 내 벡터 수를 계산합니다.
        
        Args:
            collection_name: 컬렉션 이름
            filter_expr: 필터링 표현식
            
        Returns:
            int: 벡터 수
        """
        pass
    
    @abstractmethod
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        컬렉션 통계 정보를 가져옵니다.
        
        Args:
            collection_name: 컬렉션 이름
            
        Returns:
            Dict[str, Any]: 통계 정보
        """
        pass
    
    @abstractmethod
    def create_partition(self, collection_name: str, partition_name: str) -> bool:
        """
        컬렉션 내에 파티션(네임스페이스)을 생성합니다.
        
        Args:
            collection_name: 컬렉션 이름
            partition_name: 생성할 파티션 이름
            
        Returns:
            bool: 생성 성공 여부
        """
        pass
    
    @abstractmethod
    def list_partitions(self, collection_name: str) -> List[str]:
        """
        컬렉션 내 모든 파티션 목록을 반환합니다.
        
        Args:
            collection_name: 컬렉션 이름
            
        Returns:
            List[str]: 파티션 이름 목록
        """
        pass
    
    @abstractmethod
    def has_partition(self, collection_name: str, partition_name: str) -> bool:
        """
        지정한 파티션이 존재하는지 확인합니다.
        
        Args:
            collection_name: 컬렉션 이름
            partition_name: 확인할 파티션 이름
            
        Returns:
            bool: 파티션 존재 여부
        """
        pass
    
    @abstractmethod
    def drop_partition(self, collection_name: str, partition_name: str) -> bool:
        """
        파티션을 삭제합니다.
        
        Args:
            collection_name: 컬렉션 이름
            partition_name: 삭제할 파티션 이름
            
        Returns:
            bool: 삭제 성공 여부
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """
        벡터 데이터베이스 연결을 종료합니다.
        """
        pass

    def sanitize_name(self, name: str) -> str:
        """
        컬렉션/파티션 이름을 벡터 DB에 사용 가능한 형식으로 변환합니다.
        
        Args:
            name: 원본 이름
            
        Returns:
            str: 정제된 이름
        """
        import re
        import uuid
        
        if not name:
            return f"db-{uuid.uuid4().hex[:8]}"
        
        # ASCII 문자 및 안전한 문자만 허용
        ascii_safe = re.sub(r'[^\x20-\x7E]+', '_', name)
        # 영문자, 숫자, 언더스코어, 하이픈, 점만 허용
        safe_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', ascii_safe)
        # 연속된 언더스코어 제거 및 양쪽 언더스코어 제거
        safe_name = re.sub(r'_+', '_', safe_name).strip('_')
        
        # 이름이 비어있으면 임의의 이름 생성
        if not safe_name:
            safe_name = f"db-{uuid.uuid4().hex[:8]}"
        
        # 길이 제한 (보통 벡터 DB는 이름 길이 제한이 있음)
        return safe_name[:64]
