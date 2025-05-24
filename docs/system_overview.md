# 한국어 RAG 시스템 개요

## 시스템 구조

이 프로젝트는 한국어 문서를 처리하고 질의응답할 수 있는 RAG(Retrieval-Augmented Generation) 시스템입니다. 주요 구성 요소는 다음과 같습니다:

```
src/
├── rag/                       # 핵심 RAG 컴포넌트
│   ├── embedder.py            # 임베딩 생성 모듈
│   ├── generator.py           # 응답 생성 모듈
│   ├── parser.py              # 문서 파싱 모듈
│   └── retriever.py           # 문서 검색 모듈
├── vectordb/                  # 벡터 데이터베이스 인터페이스
│   ├── interface.py           # 추상 인터페이스
│   └── milvus_client.py       # Milvus 구현체
└── utils/                     # 유틸리티
    ├── config.py              # 설정 모듈
    ├── logger.py              # 로깅 모듈
    └── sample_data.py         # 테스트용 샘플 데이터
```

## 주요 워크플로우

1. **문서 처리**: PDF, CSV, 텍스트 파일을 파싱하여 청크로 분할
2. **임베딩 생성**: 각 청크를 벡터 임베딩으로 변환
3. **벡터 저장**: 임베딩을 Milvus 벡터 데이터베이스에 저장
4. **쿼리 처리**: 사용자 질의를 임베딩하고, 확장 및 최적화
5. **벡터 검색**: 유사한 문서 청크를 검색
6. **응답 생성**: 검색된 컨텍스트를 사용하여 LLM으로 응답 생성

## 주요 기능 설명

### 문서 파싱

파싱 모듈은 다양한 유형의 문서를 처리합니다:
- **PDF**: PyMuPDF(fitz)를 사용하여 텍스트, 레이아웃, 이미지를 추출
- **CSV**: pandas를 사용하여 구조화된 데이터 처리
- **텍스트**: 일반 텍스트 파일 처리

문서는 의미있는 청크로 분할되어 처리됩니다. 최적의 청크 크기(700자)와 중첩(150자)은 한국어 텍스트에 맞게 조정되었습니다.

### 임베딩

한국어 텍스트 임베딩에 최적화된 모델을 지원합니다:
- **KoSimCSE**: 한국어에 최적화된 SimCSE 모델
- **다국어 MPNet**: 다양한 언어를 지원하는 일반 모델
- **KoSBERT**: 한국어 문장 유사도 특화 모델

임베딩은 batch_size 설정을 통해 GPU 성능에 맞게 최적화할 수 있습니다.

### 벡터 검색

검색 모듈은 다음 기능을 제공합니다:
- **벡터 유사도 검색**: cosine, dot product, L2 거리 지원
- **하이브리드 검색**: 벡터 검색과 키워드 매칭 결합
- **결과 재순위화**: 정확도 향상을 위한 검색 결과 재정렬

쿼리 처리 과정에서 다음 최적화가 적용됩니다:
- **쿼리 확장**: 여러 변형 쿼리 생성 (예: 공백 추가, 한글-숫자 변환)
- **불필요한 요소 제거**: 조사, 물음표 등 제거
- **키워드 추출**: 중요 키워드만 사용하여 검색 효율성 향상

### 응답 생성

LLM을 사용하여 검색된 컨텍스트에 기반한 응답을 생성합니다:
- **Grok API** 지원: Xai의 Grok-3-mini-beta 모델 사용
- **컨텍스트 가공**: 유사도 높은 결과를 정렬하고 적절한 컨텍스트 구성
- **프롬프트 최적화**: 한국어 응답 생성에 최적화된 프롬프트 사용

## 주요 설정 옵션

설정은 `configs/default_config.yaml`에서 관리됩니다. 주요 설정 항목:

### 청킹 설정
```yaml
chunking:
  chunk_size: 700  # 청크당 최대 문자 수
  chunk_overlap: 150  # 청크 간 겹치는 문자 수
  split_by: "sentence"  # 청크 분할 기준 (character, sentence, paragraph)
  sentence_splitter: "kss"  # 한국어 문장 분리 라이브러리
  respect_section_boundaries: true  # 섹션 경계 존중
```

### 검색 설정
```yaml
retrieval:
  top_k: 15  # 검색 결과 수
  similarity_threshold: 0.65  # 유사도 임계값
  hybrid_search: true  # 하이브리드 검색 사용 여부
  hybrid_alpha: 0.7  # 벡터:키워드 가중치 비율
  reranking: true  # 재순위화 사용 여부
```

### 임베딩 설정
```yaml
embedding:
  model: "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
  dimension: 768
  batch_size: 16
  normalize: true
```

### Milvus 설정
```yaml
milvus:
  host: "localhost"
  port: 19530
  index_type: "HNSW"  # HNSW, IVF_FLAT, FLAT
  metric_type: "COSINE"  # COSINE, L2, IP
```

## 사용 예시

### 문서 처리 및 임베딩

```bash
# 단일 PDF 파일 처리, 임베딩 생성, 벡터 DB 저장
python src/main.py parse --file path/to/document.pdf --output data/processed --embed --store-db
```

### 쿼리 실행

```bash
# 하이브리드 검색으로 질의응답
python src/main.py query --query "제1보험기간이란?" --hybrid --generate
```

### 대화형 모드

```bash
# 대화형 모드 시작
python src/main.py interactive --hybrid --top-k 15 --threshold 0.65
```

## 성능 최적화 팁

1. **청킹 전략 최적화**: 문서 특성에 맞게 `chunk_size`와 `chunk_overlap` 조정
2. **임베딩 모델 선택**: 한국어 특화 모델과 다국어 모델 비교 평가
3. **하이브리드 가중치 조정**: `hybrid_alpha` 값을 데이터 특성에 맞게 조정
4. **검색 매개변수 최적화**: `top_k`와 `similarity_threshold` 조정으로 정확도 개선
5. **GPU 가속**: 대용량 임베딩 처리 시 GPU 사용 권장

## 문제 해결

### "문서를 찾을 수 없음" 오류

- 인덱스/컬렉션 이름이 올바른지 확인
- Milvus 서버가 실행 중인지 확인
- `docker ps` 명령으로 Milvus 컨테이너 상태 확인

### 검색 결과 품질 저하

- 청크 크기와 중첩 설정 확인
- 임베딩 모델 변경 고려
- 하이브리드 검색 매개변수 조정

### API 키 인증 오류

`.env` 파일에 API 키가 올바른 형식으로 저장되어 있는지 확인:
```
GROK_API_KEY=xai-boj5ecq53AljzYHvSITHlG3Ti3B4EjOtUZa5MesGU9GY1kRltFgi2pFjCsrgv8eIDn9Lw5yood0E0dTm
```

## 확장 계획

1. **추가 벡터 데이터베이스 지원**: Qdrant, Weaviate 등 추가 벡터 DB 지원
2. **분산 처리**: 대용량 문서 처리를 위한 분산 아키텍처
3. **웹 인터페이스**: 사용자 친화적 웹 기반 인터페이스 구현
4. **실시간 학습**: 사용자 피드백을 통한 검색 및 응답 품질 개선
5. **다양한 LLM 지원**: 추가 LLM API 및 로컬 모델 지원

## 참고 문헌

- [Milvus 공식 문서](https://milvus.io/docs)
- [PyMilvus API 참조](https://milvus.io/api-reference/pymilvus/v2.2.x/About.md)
- [RAG 시스템 아키텍처](https://www.pinecone.io/learn/retrieval-augmented-generation)
- [한국어 임베딩 모델 비교](https://github.com/BM-K/KoSimCSE)
