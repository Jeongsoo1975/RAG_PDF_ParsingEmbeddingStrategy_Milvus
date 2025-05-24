# Milvus 설정 가이드

이 가이드는 Milvus 벡터 데이터베이스 설정 및 사용 방법을 설명합니다. RAG 시스템에서 Milvus를 활용하여 효율적인 벡터 검색을 구현할 수 있습니다.

## 1. Milvus 설치 방법

### Docker 컴포즈를 이용한 설치 (권장)

1. Docker와 Docker Compose가 설치되어 있는지 확인합니다.
2. 다음 내용으로 `docker-compose.yml` 파일을 생성합니다:

```yaml
version: '3'

services:
  etcd:
    container_name: milvus-etcd
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_SNAPSHOT_COUNT=50000
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/etcd:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd

  minio:
    container_name: milvus-minio
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/minio:/minio_data
    command: minio server /minio_data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  standalone:
    container_name: milvus-standalone
    image: milvusdb/milvus:v2.3.1
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/milvus:/var/lib/milvus
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      - "etcd"
      - "minio"

networks:
  default:
    name: milvus
```

3. 다음 명령어로 Milvus를 실행합니다:

```bash
docker-compose up -d
```

### 단독 실행 파일 설치 (로컬 개발용)

1. [Milvus 공식 홈페이지](https://milvus.io/docs/install_standalone-docker.md)에서 운영체제에 맞는 설치 파일을 다운로드합니다.
2. 설치 프로그램을 실행하고 지시에 따라 설치합니다.

## 2. 시스템 설정

### 필수 라이브러리 설치

```bash
pip install pymilvus
```

### Milvus 설정 (configs/default_config.yaml)

```yaml
milvus:
  host: "localhost"  # Milvus 서버 호스트
  port: 19530  # Milvus 서버 포트
  user: ""  # 인증 사용자 (빈 문자열은 인증 없음)
  password: ""  # 인증 비밀번호 (빈 문자열은 인증 없음)
  index_type: "HNSW"  # 인덱스 타입 (HNSW, IVF_FLAT, FLAT)
  metric_type: "COSINE"  # 거리 측정 방식 (COSINE, L2, IP)
  auto_id: false  # 자동 ID 생성 여부
  timeout: 30  # 작업 타임아웃 (초)
```

### Milvus 사용 설정 (configs/default_config.yaml)

```yaml
retrieval:
  use_milvus: true  # Milvus 벡터 DB 사용 여부 (true=Milvus, false=Pinecone)
```

## 3. Milvus 주요 개념

Milvus는 다음과 같은 주요 개념을 사용합니다:

| Milvus 개념 | Pinecone 대응 개념 | 설명 |
|------------|------------------|------|
| Collection | Index | 벡터와 메타데이터를 저장하는 컨테이너 |
| Partition | Namespace | 컬렉션 내 데이터를 논리적으로 분할하는 단위 |
| Field | Metadata Field | 벡터 또는 메타데이터의 필드 (스키마 기반) |
| Index | - | 벡터 검색을 위한 인덱스 구조 (HNSW, IVF_FLAT 등) |

## 4. 인덱스 유형 및 성능 최적화

Milvus는 다양한 인덱스 유형을 지원합니다:

1. **FLAT**: 전체 벡터를 스캔하는 기본 인덱스. 정확도는 가장 높지만 속도가 느림.
2. **IVF_FLAT**: 군집화 기반의 인덱스. 속도와 정확도의 균형이 좋음. `nlist` 파라미터로 조정.
3. **HNSW**: 계층적 네비게이션 스몰 월드 그래프 인덱스. 높은 속도와 정확도. `M`과 `efConstruction` 파라미터로 조정.

권장 설정:
- 소규모 컬렉션 (< 100만 벡터): HNSW (M=16, efConstruction=200)
- 대규모 컬렉션 (> 100만 벡터): IVF_FLAT (nlist=클러스터 수, 보통 벡터 수의 제곱근)

## 5. 데이터 마이그레이션 (Pinecone → Milvus)

Pinecone에서 Milvus로 데이터를 마이그레이션하는 단계:

1. Pinecone 인덱스에서 벡터와 메타데이터 추출
2. Milvus 컬렉션 및 스키마 생성
3. 벡터와 메타데이터를 Milvus에 업로드
4. 인덱스 생성 및 로드

예제 스크립트:

```python
import os
import pinecone
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

# Pinecone 설정
pinecone.init(api_key=os.environ.get("PINECONE_API_KEY"), environment="us-east-1")
pinecone_index = pinecone.Index("your-index-name")

# Milvus 연결
connections.connect("default", host="localhost", port="19530")

# Milvus 컬렉션 스키마 생성
fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768),  # 벡터 차원에 맞게 수정
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=255),
    FieldSchema(name="page_num", dtype=DataType.INT64),
    FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="section_title", dtype=DataType.VARCHAR, max_length=255)
]

schema = CollectionSchema(fields=fields, description="Migrated from Pinecone")
collection = Collection(name="migrated_collection", schema=schema)

# Pinecone 데이터 가져오기 (배치로 처리)
batch_size = 100
fetch_response = pinecone_index.fetch(ids=["list-of-ids"], namespace="your-namespace")

# Milvus에 데이터 삽입
ids = []
vectors = []
texts = []
sources = []
page_nums = []
chunk_ids = []
section_titles = []

for id, data in fetch_response["vectors"].items():
    ids.append(id)
    vectors.append(data["values"])
    
    # 메타데이터 추출 및 기본값 설정
    metadata = data.get("metadata", {})
    texts.append(metadata.get("text", ""))
    sources.append(metadata.get("source", ""))
    page_nums.append(metadata.get("page_num", -1))
    chunk_ids.append(metadata.get("chunk_id", ""))
    section_titles.append(metadata.get("section_title", ""))

# 데이터 삽입
collection.insert([
    ids, vectors, texts, sources, page_nums, chunk_ids, section_titles
])

# 인덱스 생성 (HNSW)
index_params = {
    "metric_type": "COSINE",
    "index_type": "HNSW",
    "params": {"M": 16, "efConstruction": 200}
}

collection.create_index(field_name="vector", index_params=index_params)
collection.load()  # 메모리에 로드
```

## 6. Milvus 관리 및 모니터링

### Milvus 관리 도구

- [Attu](https://github.com/zilliztech/attu): Milvus용 웹 기반 관리 UI
- [PyMilvus CLI](https://github.com/milvus-io/pymilvus-cli): 명령행 인터페이스

### 모니터링 및 유지 관리

1. **컬렉션 통계 확인**:
   ```python
   collection.get_stats()
   ```

2. **컬렉션 압축**:
   ```python
   collection.compact()
   ```

3. **백업 생성**:
   ```bash
   docker exec -it milvus-standalone mkdir -p /var/lib/milvus/backup
   docker exec -it milvus-standalone milvus_backup -database default -collection your_collection -backup-path /var/lib/milvus/backup
   ```

## 7. Milvus 설정 및 성능 최적화 팁

1. **배치 삽입 최적화**: 최적의 배치 크기는 10,000~50,000개 벡터입니다.
2. **인덱스 선택**: HNSW는 작은 데이터셋에, IVF_FLAT은 큰 데이터셋에 적합합니다.
3. **파티션 활용**: 관련 데이터를 파티션으로 그룹화하여 검색 성능을 향상시킵니다.
4. **메모리 관리**: `collection.release()` 명령으로 불필요한 컬렉션을 메모리에서 해제합니다.
5. **필터 최적화**: 자주 사용하는 필드에 대해 스칼라 인덱스를 생성합니다.

## 문제 해결

### 일반적인 문제

1. **연결 오류**: 
   - Milvus 서버가 실행 중인지 확인
   - 방화벽 설정 확인
   - 호스트 및 포트 설정 확인

2. **메모리 부족 오류**:
   - 불필요한 컬렉션 해제: `collection.release()`
   - Docker 리소스 할당 증가
   - 인덱스 타입 변경 (HNSW에서 IVF_FLAT으로)

3. **검색 결과 없음**:
   - 임계값 설정 확인
   - 컬렉션이 로드되었는지 확인: `collection.load()`
   - 파티션 이름 확인
   - 검색 파라미터 최적화

### 벡터 검색 결과 품질 개선

1. 임계값 조정: `similarity_threshold` 값을 낮추면 더 많은 결과가 반환됨
2. 하이브리드 검색 활용: 벡터 검색과 키워드 검색 결합
3. 재순위화 모델 적용: 검색 결과를 정확도에 따라 재정렬
