#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Milvus 벡터 데이터베이스 클라이언트 구현.
VectorDBInterface를 상속받아 Milvus 특화 기능 제공.
"""

import os
import logging
import time
import re
import uuid
from typing import List, Dict, Any, Optional, Union, Tuple

# src.utils.logger와 src.vectordb.interface가 있다고 가정합니다.
# 실제 프로젝트 구조에 맞게 임포트 경로를 확인하세요.
try:
    from src.utils.logger import get_logger
    from src.vectordb.interface import VectorDBInterface
except ImportError:
    # 임시 로거 및 인터페이스 (실제 프로젝트에서는 위 경로가 맞아야 합니다)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    class VectorDBInterface: # 임시 스텁
        def sanitize_name(self, name: str) -> str: return name
    logger.warning("Could not import from src.utils or src.vectordb. Ensure paths are correct.")


logger = get_logger("milvus_client")

class MilvusClient(VectorDBInterface):
    """
    Milvus 벡터 데이터베이스 클라이언트 구현.
    
    pymilvus 라이브러리를 사용하여 Milvus 서버와 통신합니다.
    """
    
    def __init__(self, 
                config: Any, 
                host: Optional[str] = None, 
                port: Optional[int] = None, 
                user: Optional[str] = None, 
                password: Optional[str] = None):
        self.config = config
        # config 객체에서 milvus 설정을 가져오는 방식은 실제 Config 클래스 구조에 따라 달라질 수 있습니다.
        # 예시: self.milvus_config = config.get_section('milvus') 또는 config.milvus
        self.milvus_config = {}
        if hasattr(config, 'milvus'):
            self.milvus_config = config.milvus
        elif hasattr(config, 'config_data') and 'milvus' in config.config_data:
            self.milvus_config = config.config_data['milvus']
        else:
            logger.warning("Milvus configuration not found in provided config object. Using defaults.")

        self.host = host or self.milvus_config.get('host', os.environ.get('MILVUS_HOST', 'localhost'))
        self.port = port or self.milvus_config.get('port', int(os.environ.get('MILVUS_PORT', 19530)))
        self.user = user or self.milvus_config.get('user', os.environ.get('MILVUS_USER', None))
        self.password = password or self.milvus_config.get('password', os.environ.get('MILVUS_PASSWORD', None))
        
        self.default_index_type = self.milvus_config.get('index_type', 'HNSW')
        self.default_metric_type = self.milvus_config.get('metric_type', 'COSINE') # L2 또는 IP도 가능
        self.auto_id = self.milvus_config.get('auto_id', False) # 일반적으로 ID를 직접 제공하므로 False
        self.timeout = self.milvus_config.get('timeout', 30) # 초 단위
        
        self._connected = False
        self._utility = None
        
        try:
            from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType
            self._pymilvus_imported = True
            self._pymilvus_modules = {
                'connections': connections,
                'utility': utility,
                'Collection': Collection,
                'FieldSchema': FieldSchema,
                'CollectionSchema': CollectionSchema,
                'DataType': DataType
            }
            logger.info("pymilvus 라이브러리를 성공적으로 가져왔습니다.")
        except ImportError:
            self._pymilvus_imported = False
            logger.error("pymilvus 라이브러리를 찾을 수 없습니다. 'pip install pymilvus' 명령으로 설치하세요.")
            raise
        
        if self._pymilvus_imported:
            self.connect()
    
    def connect(self) -> bool:
        if not self._pymilvus_imported:
            return False
        try:
            connections = self._pymilvus_modules['connections']
            
            alias = "default" # 기본 연결 별칭
            # 이미 연결된 경우 해당 별칭의 연결을 먼저 끊습니다.
            if alias in connections.list_connections():
                logger.info(f"Existing connection for alias '{alias}' found. Disconnecting first.")
                connections.disconnect(alias)

            conn_params = {
                "host": self.host,
                "port": str(self.port), # 포트는 문자열이어야 할 수 있습니다.
                "alias": alias
            }
            if self.user and self.password:
                conn_params["user"] = self.user
                conn_params["password"] = self.password
            
            logger.info(f"Milvus 서버 연결 시도: {self.host}:{self.port}")
            connections.connect(**conn_params)
            
            # 간단한 연결 테스트 (예: 서버 버전 가져오기)
            server_version = self._get_server_version()
            self._connected = True
            logger.info(f"Milvus 서버({server_version})에 성공적으로 연결되었습니다.")
            return True
        except Exception as e:
            logger.error(f"Milvus 연결 오류: {e}", exc_info=True)
            self._connected = False
            return False
    
    def is_connected(self) -> bool:
        if not self._pymilvus_imported or not self._connected:
            return False
        try:
            # 간단한 작업으로 연결 상태 확인
            self._pymilvus_modules['utility'].get_server_version()
            return True
        except Exception:
            self._connected = False
            return False

    def _get_server_version(self) -> str:
        try:
            return self._pymilvus_modules['utility'].get_server_version()
        except Exception as e:
            logger.warning(f"서버 버전 조회 실패: {e}")
            return "unknown"

    def _check_connection(self) -> bool:
        if not self._pymilvus_imported:
            return False
        if not self.is_connected(): # is_connected가 내부적으로 _connected를 업데이트할 수 있음
            logger.warning("Milvus 연결이 끊어졌습니다. 재연결을 시도합니다.")
            return self.connect()
        return True
    
    def create_collection(self, collection_name: str, dimension: int, metric_type: str = "COSINE") -> bool:
        if not self._check_connection():
            return False
        
        collection_name = self.sanitize_name(collection_name)
        logger.info(f"컬렉션 생성 시도: {collection_name}, 차원: {dimension}, 유사도 측정: {metric_type}")

        try:
            utility = self._pymilvus_modules['utility']
            if utility.has_collection(collection_name):
                logger.info(f"컬렉션 '{collection_name}'가 이미 존재합니다.")
                # 필요하다면 여기서 스키마를 확인하고, 다르면 삭제 후 재생성하는 로직 추가 가능
                # existing_collection = self._pymilvus_modules['Collection'](collection_name)
                # if not self._is_schema_compatible(existing_collection.schema):
                #     self.drop_collection(collection_name)
                # else:
                return True 
            
            FieldSchema = self._pymilvus_modules['FieldSchema']
            DataType = self._pymilvus_modules['DataType']
            CollectionSchema = self._pymilvus_modules['CollectionSchema']
            
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=255), # UUID 길이를 고려하여 충분히 길게
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535), # Milvus의 VARCHAR 최대 길이 근접
                FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=255), # 문서 ID
                FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1024), # 파일 경로 등
                FieldSchema(name="page_num", dtype=DataType.INT32), # INT32로 변경, INT64도 가능
                FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=255), # 청크 자체 ID
                FieldSchema(name="parent_chunk_id", dtype=DataType.VARCHAR, max_length=255, default_value=""), # 부모 청크 ID
                FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=50),  # **추가된 필드** (예: "item", "article")
                # 추가 메타데이터 필드를 위한 일반적인 필드 (선택적)
                FieldSchema(name="article_title", dtype=DataType.VARCHAR, max_length=1024, default_value=""),
                FieldSchema(name="item_marker", dtype=DataType.VARCHAR, max_length=50, default_value=""),
                # 필요에 따라 더 많은 메타데이터 필드 추가 가능
            ]
            
            schema = CollectionSchema(fields=fields, description=f"RAG Document Collection: {collection_name}", enable_dynamic_field=True) # Dynamic field 추가
            
            Collection = self._pymilvus_modules['Collection']
            collection = Collection(name=collection_name, schema=schema, using='default') # using='default' 명시
            logger.info(f"컬렉션 '{collection.name}' 생성 완료.")
            
            self._create_index(collection, metric_type or self.default_metric_type)
            collection.load()
            logger.info(f"컬렉션 '{collection.name}' 로드 완료.")
            return True
            
        except Exception as e:
            logger.error(f"컬렉션 '{collection_name}' 생성 오류: {e}", exc_info=True)
            return False

    def _create_index(self, collection, metric_type: str):
        logger.info(f"'{collection.name}' 컬렉션에 대한 인덱스 생성 중, 유사도 측정: {metric_type}")
        index_params = {
            "metric_type": metric_type,
            "index_type": self.default_index_type, # HNSW, IVF_FLAT 등
            "params": self.milvus_config.get('index_params', {"M": 16, "efConstruction": 256}) # HNSW 기본 파라미터
        }
        if self.default_index_type == "IVF_FLAT":
             index_params["params"] = self.milvus_config.get('index_params_ivf', {"nlist": 1024})

        try:
            collection.create_index(field_name="vector", index_params=index_params)
            logger.info(f"'{collection.name}' 컬렉션의 'vector' 필드에 {self.default_index_type} 인덱스 생성 완료.")
        except Exception as e:
            # 인덱스가 이미 존재할 수 있음 (Milvus 동작에 따라 다름)
            if "index already exist" in str(e).lower():
                logger.warning(f"'{collection.name}' 컬렉션에 인덱스가 이미 존재합니다.")
            else:
                logger.error(f"'{collection.name}' 컬렉션 인덱스 생성 오류: {e}", exc_info=True)
                raise # 인덱스 생성 실패는 심각한 문제일 수 있으므로 예외를 다시 발생시킴

    def has_collection(self, collection_name: str) -> bool:
        if not self._check_connection(): return False
        collection_name = self.sanitize_name(collection_name)
        return self._pymilvus_modules['utility'].has_collection(collection_name)

    def list_collections(self) -> List[str]:
        if not self._check_connection(): return []
        return self._pymilvus_modules['utility'].list_collections()

    def drop_collection(self, collection_name: str) -> bool:
        if not self._check_connection(): return False
        collection_name = self.sanitize_name(collection_name)
        if not self.has_collection(collection_name):
            logger.warning(f"삭제할 컬렉션 '{collection_name}'이(가) 존재하지 않습니다.")
            return True # 이미 없으므로 성공으로 간주 가능
        try:
            self._pymilvus_modules['utility'].drop_collection(collection_name)
            logger.info(f"컬렉션 '{collection_name}' 삭제 완료.")
            return True
        except Exception as e:
            logger.error(f"컬렉션 '{collection_name}' 삭제 오류: {e}", exc_info=True)
            return False

    def insert(self, collection_name: str, ids: List[str], vectors: List[List[float]], 
               metadata: List[Dict[str, Any]], partition: Optional[str] = None, force_flush: bool = True) -> int:
        if not self._check_connection(): return 0
        
        collection_name = self.sanitize_name(collection_name)
        if not self.has_collection(collection_name):
            logger.error(f"삽입 대상 컬렉션 '{collection_name}'이(가) 존재하지 않습니다.")
            return 0
        
        if not (len(ids) == len(vectors) == len(metadata)):
            logger.error(f"데이터 길이 불일치: ids({len(ids)}), vectors({len(vectors)}), metadata({len(metadata)})")
            return 0
        if not ids:
            logger.warning("삽입할 데이터가 없습니다.")
            return 0

        collection = self._pymilvus_modules['Collection'](collection_name)
        
        # 스키마 필드명 가져오기 (매번 하는 것보다 초기화 시 한 번 하는 것이 효율적일 수 있음)
        schema_fields = [field.name for field in collection.schema.fields]

        data_to_insert = []
        for i in range(len(ids)):
            entry = {
                "id": str(ids[i]), # ID는 문자열이어야 함
                "vector": vectors[i]
            }
            # 스키마에 정의된 메타데이터 필드만 선택적으로 추가
            # metadata[i]는 parser.py의 TextChunk.metadata + TextChunk의 다른 필드들로 구성됨
            # TextChunk에서 직접 가져오는 필드들: text, doc_id, source, page_num, chunk_id, parent_chunk_id, chunk_type
            # TextChunk.metadata에 있는 필드들: article_title, item_marker 등
            
            # TextChunk의 직접적인 속성들
            entry["text"] = metadata[i].get("text", "")[:65530] # 길이 제한 고려
            entry["doc_id"] = metadata[i].get("doc_id", "")[:250]
            entry["source"] = metadata[i].get("source", metadata[i].get("source_file", ""))[:1020]
            entry["page_num"] = metadata[i].get("page_num", -1) 
            entry["chunk_id"] = metadata[i].get("chunk_id", str(ids[i]))[:250] # chunk_id가 없으면 id 사용
            entry["parent_chunk_id"] = metadata[i].get("parent_chunk_id", "")[:250]
            entry["chunk_type"] = metadata[i].get("chunk_type", "unknown")[:45] # **중요: chunk_type 값 할당**

            # TextChunk.metadata 내의 추가 정보 (스키마에 있다면)
            if "article_title" in schema_fields:
                entry["article_title"] = metadata[i].get("article_title", "")[:1020]
            if "item_marker" in schema_fields:
                entry["item_marker"] = metadata[i].get("item_marker", "")[:45]
            
            # 스키마에 없는 다른 메타데이터는 dynamic field로 들어감 (스키마에서 enable_dynamic_field=True 설정 시)
            # 또는, 필요한 모든 메타데이터 필드를 스키마에 명시적으로 추가해야 함.
            # 여기서는 명시된 필드 위주로 처리
            
            # 누락된 스키마 필드에 대해 기본값 설정 (Milvus가 기본값을 자동으로 처리하지 않는 경우)
            for field_name in schema_fields:
                if field_name not in entry:
                    if collection.schema.fields[schema_fields.index(field_name)].dtype == self._pymilvus_modules['DataType'].VARCHAR:
                        entry[field_name] = ""
                    elif collection.schema.fields[schema_fields.index(field_name)].dtype in [self._pymilvus_modules['DataType'].INT32, self._pymilvus_modules['DataType'].INT64]:
                        entry[field_name] = -1 # 또는 0
                    # 다른 타입에 대한 기본값 처리 추가 가능

            data_to_insert.append(entry)

        try:
            logger.debug(f"{collection_name}에 데이터 삽입 시도 (첫 번째 항목): {data_to_insert[0] if data_to_insert else 'N/A'}")
            
            # 파티션 처리 - 'partition' 매개변수는 무시
            # 주의: partition 매개변수는 여전히 함수 시그니처에 있지만 내부적으로 사용하지 않음
            
            # 수정된 부분: partition 매개변수 없이 insert 호출
            res = collection.insert(data_to_insert)
            
            if force_flush and res.insert_count > 0:
                collection.flush()
                logger.info(f"컬렉션 '{collection_name}' 플러시 완료. 총 {res.insert_count}개 삽입됨.")
            elif res.insert_count > 0:
                 logger.info(f"총 {res.insert_count}개 벡터 삽입됨 (플러시는 지연될 수 있음).")
            else:
                logger.warning(f"삽입된 벡터가 없습니다. 응답: {res}")

            return res.insert_count
        except Exception as e:
            logger.error(f"Milvus 데이터 삽입 오류: {e}", exc_info=True)
            logger.error(f"오류 발생 데이터 (일부): {data_to_insert[0] if data_to_insert else 'N/A'}")
            return 0

    def search(self, collection_name: str, query_vector: List[float], top_k: int = 5, 
               filter_expr: Optional[str] = None, partition_names: Optional[List[str]] = None,
               output_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]: # partition -> partition_names, output_fields 추가
        if not self._check_connection(): return []
        
        collection_name = self.sanitize_name(collection_name)
        if not self.has_collection(collection_name):
            logger.error(f"검색 대상 컬렉션 '{collection_name}'이(가) 존재하지 않습니다.")
            return []

        collection = self._pymilvus_modules['Collection'](collection_name)
        collection.load() # 검색 전 로드 확인

        search_params = self.milvus_config.get('search_params', {"ef": 64}) # HNSW 기본 파라미터
        if self.default_index_type == "IVF_FLAT":
             search_params = self.milvus_config.get('search_params_ivf', {"nprobe": 10})


        # 기본 output_fields 설정, chunk_type 포함
        if output_fields is None:
            output_fields = ["id", "text", "doc_id", "source", "page_num", "chunk_id", "parent_chunk_id", "chunk_type", "article_title", "item_marker"]
            # 컬렉션 스키마에 있는 필드만 요청하도록 필터링
            collection_schema_fields = [f.name for f in collection.schema.fields]
            output_fields = [f for f in output_fields if f in collection_schema_fields]


        final_partition_names = None
        if partition_names:
            final_partition_names = [self.sanitize_name(pn) for pn in partition_names]
            final_partition_names = [pn for pn in final_partition_names if self.has_partition(collection_name, pn)]
            if not final_partition_names:
                 logger.warning(f"지정된 파티션 {partition_names} 중 유효한 파티션이 없습니다. 전체 컬렉션에서 검색합니다.")
        
        logger.debug(f"Milvus 검색 시작: 컬렉션='{collection_name}', top_k={top_k}, 필터='{filter_expr}', 파티션='{final_partition_names}', 출력필드='{output_fields}'")

        try:
            results = collection.search(
                data=[query_vector],
                anns_field="vector",
                param={"metric_type": self.default_metric_type, "params": search_params},
                limit=top_k,
                expr=filter_expr,
                output_fields=output_fields,
                consistency_level="Strong", # 필요에 따라 조정
                partition_names=final_partition_names
            )
            
            processed_results = []
            for hits in results:
                for hit in hits:
                    entity_data = {"id": hit.id, "score": hit.distance} # score == distance
                    for field in output_fields:
                        if field not in ["id", "vector"]: # id는 이미 있고, vector는 보통 불필요
                             entity_data[field] = hit.entity.get(field)
                    processed_results.append(entity_data)
            
            logger.info(f"Milvus 검색 완료. {len(processed_results)}개 결과 반환.")
            return processed_results
        except Exception as e:
            logger.error(f"Milvus 검색 오류: {e}", exc_info=True)
            return []

    def count(self, collection_name: str, filter_expr: Optional[str] = None) -> int:
        # ... (기존 코드와 유사하게 유지 또는 필요시 수정) ...
        if not self._check_connection(): return 0
        collection_name = self.sanitize_name(collection_name)
        if not self.has_collection(collection_name): return 0
        collection = self._pymilvus_modules['Collection'](collection_name)
        collection.flush() # 정확한 카운트를 위해 플러시
        return collection.num_entities


    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        # ... (기존 코드와 유사하게 유지 또는 필요시 수정) ...
        if not self._check_connection(): return {}
        collection_name = self.sanitize_name(collection_name)
        if not self.has_collection(collection_name): return {}
        collection = self._pymilvus_modules['Collection'](collection_name)
        stats = collection.describe()
        stats['num_entities'] = collection.num_entities
        return stats


    def create_partition(self, collection_name: str, partition_name: str) -> bool:
        if not self._check_connection(): return False
        collection_name = self.sanitize_name(collection_name)
        partition_name = self.sanitize_name(partition_name)
        if not self.has_collection(collection_name):
            logger.error(f"파티션을 생성할 컬렉션 '{collection_name}'이(가) 존재하지 않습니다.")
            return False
        collection = self._pymilvus_modules['Collection'](collection_name)
        if collection.has_partition(partition_name):
            logger.info(f"파티션 '{partition_name}'이(가) 컬렉션 '{collection_name}'에 이미 존재합니다.")
            return True
        try:
            collection.create_partition(partition_name)
            logger.info(f"컬렉션 '{collection_name}'에 파티션 '{partition_name}' 생성 완료.")
            return True
        except Exception as e:
            logger.error(f"파티션 '{partition_name}' 생성 오류: {e}", exc_info=True)
            return False

    def list_partitions(self, collection_name: str) -> List[str]:
        if not self._check_connection(): return []
        collection_name = self.sanitize_name(collection_name)
        if not self.has_collection(collection_name): return []
        collection = self._pymilvus_modules['Collection'](collection_name)
        return [p.name for p in collection.partitions]

    def has_partition(self, collection_name: str, partition_name: str) -> bool:
        if not self._check_connection(): return False
        collection_name = self.sanitize_name(collection_name)
        partition_name = self.sanitize_name(partition_name)
        if not self.has_collection(collection_name): return False
        collection = self._pymilvus_modules['Collection'](collection_name)
        return collection.has_partition(partition_name)

    def drop_partition(self, collection_name: str, partition_name: str) -> bool:
        if not self._check_connection(): return False
        collection_name = self.sanitize_name(collection_name)
        partition_name = self.sanitize_name(partition_name)
        if not self.has_collection(collection_name): return False
        if not self.has_partition(collection_name, partition_name):
            logger.warning(f"삭제할 파티션 '{partition_name}'이(가) 컬렉션 '{collection_name}'에 존재하지 않습니다.")
            return True
        collection = self._pymilvus_modules['Collection'](collection_name)
        try:
            collection.drop_partition(partition_name)
            logger.info(f"컬렉션 '{collection_name}'의 파티션 '{partition_name}' 삭제 완료.")
            return True
        except Exception as e:
            logger.error(f"파티션 '{partition_name}' 삭제 오류: {e}", exc_info=True)
            return False
    
    def close(self) -> None:
        try:
            if self._connected:
                self._pymilvus_modules['connections'].disconnect("default")
                logger.info("Milvus 연결을 종료했습니다.")
                self._connected = False
        except Exception as e:
            logger.error(f"Milvus 연결 종료 오류: {e}", exc_info=True)

    def sanitize_name(self, name: str) -> str:
        """ Milvus 이름 규칙에 맞게 정제 (알파벳, 숫자, 밑줄만 허용, 첫 글자 알파벳 또는 밑줄) """
        if not name:
            return f"collection_{uuid.uuid4().hex[:8]}" # 이름이 비어있을 경우 기본값
        
        # 허용되지 않는 문자 제거 (알파벳, 숫자, 밑줄 이외의 모든 것)
        sane_name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        
        # 첫 글자가 숫자인 경우 앞에 'c_' 추가
        if sane_name[0].isdigit():
            sane_name = 'c_' + sane_name
            
        # 첫 글자가 밑줄이 아닌 경우 (이미 알파벳으로 시작하면 괜찮음)
        elif not sane_name[0].isalpha() and sane_name[0] != '_':
             sane_name = '_' + sane_name


        # 길이 제한 (Milvus는 일반적으로 255자 제한)
        sane_name = sane_name[:250] # 여유를 두고 자름

        if not sane_name: # 모든 문자가 제거된 극단적인 경우
             return f"collection_{uuid.uuid4().hex[:8]}"
        return sane_name