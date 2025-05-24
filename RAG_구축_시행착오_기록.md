# RAG 시스템 구축 시행착오 및 해결 기록

## 📋 프로젝트 개요
- **목표**: Milvus + RAGAS를 활용한 RAG 시스템 구축 및 평가
- **데이터**: Amnesty QA (인권 관련 질문-답변 데이터)
- **주요 기능**: 벡터 검색, 영어 답변 생성, RAGAS 평가

---

## 🚨 주요 시행착오 및 해결방법

### 1. Python 환경 및 의존성 문제

#### 문제
- Python 3.10과 3.13 환경 혼재
- `regex._regex` 모듈 오류
- `sentence_transformers`, `transformers` 임포트 실패

#### 해결
```powershell
# 1. regex 모듈 재설치
pip uninstall regex
pip install regex

# 2. Python 3.13 환경 사용
py -3.13 [스크립트명]

# 3. 환경변수 설정
$env:PYTHONPATH="C:\Users\USER\Documents\MCPData\RAG_PDF_ParsingEmbeddingStrategy_Milvus"
$env:GROK_API_KEY="xai-M7Jm15afb6FCzihBErymKFPhRQ8Fbe0lHafKBOwhs27yUzUXQG1XQXhnhxXtGhcG4AOazX8YsUa2LVI3"
```

### 2. 데이터 형식 불일치 문제

#### 문제
- `embedder.py`는 `*_chunks.json` 파일 형식 요구
- 실제 데이터는 `amnesty_qa_documents.json` (다른 구조)
- Document 클래스의 `chunks` 키 없음 오류

#### 해결
```python
# split_documents.py 생성하여 데이터 변환
# 단일 documents.json → 개별 *_chunks.json 파일들로 분할

# 실행 순서:
python split_documents.py
python src/rag/embedder.py --input_path data/amnesty_qa/chunks --store_in_db
```

### 3. Milvus 서버 연결 문제

#### 문제
- 초기 Milvus 서버 미실행
- 연결 실패로 임베딩 불가

#### 해결
```powershell
# Docker Compose로 Milvus 실행
docker-compose up -d
docker-compose ps  # 상태 확인

# 컨테이너 목록:
# - milvus-standalone (19530:19530)
# - milvus-etcd
# - milvus-minio
```

### 4. 벡터 검색 필터 문제

#### 문제
- 기본 필터: `chunk_type == "item" or chunk_type == "item_sub_chunk"`
- 실제 데이터: `chunk_type = "context"`
- 필터 불일치로 검색 결과 0개

#### 해결
```python
# 검색 시 필터 비활성화
results = retriever.retrieve(query, top_k=3, force_filter_expr=None)
```

### 5. 결과 출력 포맷팅 오류

#### 문제
```python
# 오류 코드
print(f"점수: {score:.3f}")  # score가 문자열일 때 오류
```

#### 해결
```python
# 수정된 코드
score = result.get("score", result.get("similarity", "N/A"))
print(f"점수: {score}")  # 타입 안전 처리
```

---

## ✅ 성공적인 실행 순서

### 1단계: 환경 설정
```powershell
$env:GROK_API_KEY="xai-M7Jm15afb6FCzihBErymKFPhRQ8Fbe0lHafKBOwhs27yUzUXQG1XQXhnhxXtGhcG4AOazX8YsUa2LVI3"
$env:PYTHONPATH="C:\Users\USER\Documents\MCPData\RAG_PDF_ParsingEmbeddingStrategy_Milvus"
```

### 2단계: Milvus 서버 실행
```powershell
docker-compose up -d
docker-compose ps  # 상태 확인
```

### 3단계: 데이터 준비
```powershell
python split_documents.py  # 데이터 형식 변환
```

### 4단계: 임베딩 실행
```powershell
py -3.13 src/rag/embedder.py --input_path data/amnesty_qa/chunks --store_in_db
```

### 5단계: 검색 테스트
```powershell
py -3.13 test_vector_search.py
```

### 6단계: 최종 평가
```powershell
py -3.13 src/evaluation/step3_standard_ragas_evaluator.py
```

---

## 🎯 최종 성과

### 성공적으로 구현된 기능
- ✅ Milvus 벡터 데이터베이스 연동
- ✅ 10개 문서 임베딩 및 저장 (각각 개별 컬렉션)
- ✅ HNSW 인덱스 생성 및 최적화
- ✅ 벡터 검색 기능 구현
- ✅ 영어 답변 생성 (언어 오버라이드)
- ✅ RAGAS 5개 지표 평가

### 최종 성능 지표
| 지표 | 점수 | 상태 |
|------|------|------|
| Context Precision | 1.000 | 완벽 |
| Context Recall | 1.000 | 완벽 |
| Faithfulness | 0.665 | 양호 |
| Context Relevancy | 0.288 | 보통 |
| Answer Relevancy | 0.127 | 개선 필요 |
| **Overall Score** | **0.616** | **양호** |

---

## 🚨 주의사항 (다음 실행 시 참고)

### 필수 환경 설정
1. **Python 버전**: 3.13 사용 (`py -3.13`)
2. **API 키**: GROK_API_KEY 환경변수 필수
3. **PYTHONPATH**: 프로젝트 루트 경로 설정
4. **Milvus 서버**: 실행 전 반드시 `docker-compose up -d`

### 일반적인 오류 해결
1. **regex 오류**: `pip uninstall regex && pip install regex`
2. **임베딩 실패**: 데이터 형식 확인 (`*_chunks.json` 필요)
3. **검색 결과 없음**: 필터 확인 (`force_filter_expr=None`)
4. **포맷팅 오류**: 타입 안전 처리 (`str(text)[:100]`)

### 파일 구조 요구사항
```
data/amnesty_qa/chunks/
├── amnesty_qa_document_00_chunks.json
├── amnesty_qa_document_01_chunks.json
└── ... (10개 파일)
```

---

## 📁 생성된 주요 파일들

### 유틸리티 스크립트
- `split_documents.py`: 데이터 형식 변환
- `test_vector_search.py`: 벡터 검색 테스트
- `integration_example.py`: 통합 예제 (수정됨)

### 결과 파일
- `evaluation_results/step3_standard_ragas/step3_standard_ragas_evaluation_20250524_111750.json`

### Docker 환경
- `docker-compose.yml`: Milvus 서버 설정
- 컨테이너: milvus-standalone, milvus-etcd, milvus-minio

---

## 💡 핵심 교훈

1. **환경 일관성**: Python 버전과 의존성 관리 중요
2. **데이터 형식**: embedder 요구사항에 맞는 데이터 준비 필수
3. **서버 상태**: Milvus 서버 실행 상태 사전 확인
4. **필터 설정**: 실제 데이터와 필터 조건 일치 확인
5. **타입 안전성**: 동적 데이터 처리 시 타입 검증 필요

이 기록을 바탕으로 향후 RAG 시스템 구축 시 동일한 오류를 방지하고 빠른 구현이 가능할 것입니다.
