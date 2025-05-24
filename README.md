# 한국어 RAG 시스템 (Korean RAG System)

한국어 문서 처리에 최적화된 검색 증강 생성(Retrieval-Augmented Generation, RAG) 시스템입니다. 
PDF, CSV, 텍스트 파일을 처리하고 임베딩하여 질의응답 시스템을 구축할 수 있습니다.

## 주요 특징

- **Small-to-Big 청킹 전략** 구현으로 정확도와 맥락 유지 균형
- **다양한 한국어 임베딩 모델** 지원 및 성능 비교
- **재순위화(Reranking)** 모듈로 검색 정확도 향상
- **자동화된 평가 시스템**으로 다양한 구성 테스트
- **계층적 문서 구조 인식**과 메타데이터 보존

## 기능

- **문서 파싱**: PDF, CSV, 텍스트 파일 지원
  - PDF에서 텍스트 추출 및 구조 분석
  - 계층적 구조(조항, 항목, 문장 등) 인식
  - Small-to-Big 청킹 전략 구현
  - 메타데이터 추출 및 보존
  
- **임베딩**: 다양한 한국어 임베딩 모델 지원
  - KoSimCSE, KoSRoBERTa, KR-SBERT 등 한국어 특화 모델
  - MPNet, MiniLM 등 다국어 모델
  - 부모-자식 청크 관계 유지
  - GPU 가속 지원
  
- **검색**: 고급 검색 기능
  - Milvus 벡터 데이터베이스 기반 유사도 검색
  - Small-to-Big 전략으로 정확도와 컨텍스트 균형 유지
  - 다양한 재순위화 모델 지원
  - 필터링 및 메타데이터 기반 검색
  
- **평가**: 자동화된 성능 평가 시스템
  - 다양한 임베딩 및 재순위화 모델 조합 테스트
  - 정밀도, 재현율, F1 점수, MRR, NDCG 등 지표 측정
  - 결과 시각화 및 보고서 생성
  - 질문 유형별 성능 분석

## 설치

### 요구 사항

- Python 3.8 이상
- CUDA 지원 GPU (선택 사항이지만 권장)
- Milvus 서버 (로컬 또는 클라우드)

### 설치 방법

1. 저장소 클론

```bash
git clone <repository-url>
cd RAG_PDF_ParsingEmbeddingStrategy_Milvus
```

2. 필요 패키지 설치

```bash
pip install -r requirements.txt
```

3. 추가 패키지 설치 (선택 사항)

```bash
# GPU 사용을 위한 CUDA 지원 PyTorch 설치 (CUDA 버전에 맞게 설치)
pip install torch==2.0.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

4. Milvus 서버 실행 (Docker 사용)

```bash
docker-compose up -d
```

5. 환경 변수 설정

```bash
# .env 파일 생성
MILVUS_HOST=localhost
MILVUS_PORT=19530
DEFAULT_COLLECTION=insurance_ko_sroberta
```

## 사용 방법

### 1. 문서 처리 및 임베딩

```bash
# PDF 파일 처리 및 청킹
python src/rag/parser.py path/to/document.pdf

# 임베딩 생성 및 Milvus 저장 (KoSRoBERTa 모델)
python create_milvus_embeddings.py

# 다른 임베딩 모델 사용 (KoSimCSE)
python create_milvus_embeddings_kosimcse.py

# SBERT 모델 사용
python create_milvus_embeddings_krsbert.py
```

### 2. 검색 시스템 테스트

시스템의 검색 기능이 정상적으로 작동하는지 확인하려면 다음 명령어를 실행하세요:

```bash
# 전체 검색 시스템 테스트 실행
python test_search_system.py
```

테스트는 다음 항목들을 검증합니다:
- **컬렉션 정보 확인**: Milvus 컬렉션 존재 여부 및 문서 수 확인
- **검색 쿼리 테스트**: 다양한 검색 쿼리 (영어/한국어) 성능 측정
- **필터링된 검색**: chunk_type별 필터링 검색 테스트
- **RAG 파이프라인**: 전체 RAG 워크플로우 시뮬레이션

테스트 결과 해석:
- ✅ **성공**: 해당 기능이 정상적으로 작동함
- ❌ **실패**: 오류 발생, 로그에서 구체적인 해결 방안 확인
- **성능 지표**: 각 테스트별 실행 시간 및 유사도 통계 제공

### 3. 평가 실행

```bash
# 기본 임베딩 모델(KoSRoBERTa) 평가
python evaluate_with_ko_sroberta.py

# 다양한 재순위 모델 비교 평가
python run_eval_reranker_comparison.py

# 특정 모델 직접 평가
python run_eval_kosimcse_direct.py
python run_eval_ko_sroberta_direct.py
python run_eval_krsbert_direct.py
```

### 4. 결과 분석

평가 결과는 `evaluation_results` 디렉토리에 저장됩니다:
- `ALL_EVALUATION_RUNS_SUMMARY_*.json`: 전체 평가 실행 요약
- `PERFORMANCE_RANKING_*.json`: 모델 성능 순위
- `reranker_comparison_*.json`: 재순위 모델 비교 결과

## 주요 모듈

### 1. 문서 파싱 및 청킹 (src/rag/parser.py)

PDF, CSV, 텍스트 파일을 처리하고 계층적 청크로 변환하는 모듈입니다:

```python
# 사용 예시
parser = DocumentParser(config=my_config)
document = parser.process_pdf("path/to/document.pdf")
```

### 2. 임베딩 생성 (src/rag/embedder.py)

텍스트 청크를 벡터로 변환하고 Milvus에 저장하는 모듈입니다:

```python
# 사용 예시
embedder = DocumentEmbedder(config=my_config)
embedder.embed_document(document)
```

### 3. 문서 검색 (src/rag/retriever.py)

쿼리를 기반으로 관련 문서를 검색하는 모듈입니다:

```python
# 사용 예시
retriever = DocumentRetriever(config=my_config)
results = retriever.retrieve(
    query="한국어 RAG 시스템의 특징은?",
    top_k=5,
    threshold=0.7,
    use_parent_chunks=True
)
```

### 4. 자동 평가 (src/evaluation/auto_eval/evaluator.py)

다양한 모델 구성을 자동으로 평가하는 모듈입니다:

```python
# 사용 예시
evaluator = AutoEvaluator(config_path_str="configs/evaluation_config.yaml")
result = evaluator.evaluate_embedding_model(
    embedding_model_name="jhgan/ko-sroberta-multitask",
    dataset_path="src/evaluation/data/insurance_eval_dataset.json",
    top_k=5,
    similarity_threshold=0.05,
    reranker_model_name=None
)
```

## 프로젝트 구조

```
RAG_PDF_ParsingEmbeddingStrategy_Milvus/
├── configs/                       # 설정 파일
│   └── evaluation_config.yaml     # 평가 설정 파일
├── data/                          # 데이터 파일 디렉토리
│   └── parsed_output/             # 파싱된 문서 저장
├── docs/                          # 문서 및 가이드
│   ├── milvus_setup_guide.md      # Milvus 설정 가이드
│   ├── project_plan.md            # 프로젝트 계획
│   ├── small_to_big_chunking_guide.md # Small-to-Big 청킹 가이드
│   └── system_overview.md         # 시스템 개요
├── evaluation_results/            # 평가 결과 저장 디렉토리
├── logs/                          # 로그 디렉토리
├── src/                           # 소스 코드
│   ├── evaluation/                # 평가 모듈
│   │   ├── auto_eval/             # 자동 평가 시스템
│   │   │   ├── evaluator.py       # 평가 엔진
│   │   │   └── model_manager.py   # 모델 관리자
│   │   ├── data/                  # 평가용 데이터셋
│   │   └── web/                   # 웹 기반 평가 도구
│   ├── parsers/                   # 특화된 파서 모듈
│   │   └── insurance_pdf_parser.py # 보험 문서 특화 파서
│   ├── rag/                       # RAG 핵심 모듈
│   │   ├── embedder.py            # 임베딩 생성 모듈
│   │   ├── generator.py           # 응답 생성 모듈
│   │   ├── parser.py              # 문서 파싱 모듈
│   │   └── retriever.py           # 문서 검색 모듈
│   ├── utils/                     # 유틸리티 모듈
│   │   ├── config.py              # 설정 클래스
│   │   └── logger.py              # 로깅 유틸리티
│   └── vectordb/                  # 벡터 DB 관련 모듈
├── tests/                         # 테스트 코드
├── volumes/                       # Milvus 데이터 볼륨
├── create_milvus_embeddings.py    # 임베딩 생성 스크립트
├── evaluate_with_ko_sroberta.py   # 평가 실행 스크립트
├── run_eval_reranker_comparison.py # 재순위 모델 비교 스크립트
├── docker-compose.yml             # Docker 구성 파일
├── requirements.txt               # 필요 패키지 목록
└── README.md                      # 이 파일
```

## Small-to-Big 청킹 전략

Small-to-Big 청킹 전략은 이 프로젝트의 핵심 기능 중 하나입니다. 이 전략은 다음과 같은 장점을 제공합니다:

1. **정확도 향상**: 작은 청크는 높은 벡터 유사도 점수를 제공하여 정확한 정보 검색
2. **맥락 유지**: 부모 청크는 연관 정보의 더 넓은 맥락을 제공하여 부분적 이해 방지
3. **앵커링 효과**: 관련 세부 정보의 출처와 맥락을 명확히 파악 가능
4. **응답 품질 개선**: LLM이 관련 정보의 맥락을 더 잘 이해하여 응답 품질 향상

자세한 구현 가이드는 `docs/small_to_big_chunking_guide.md` 문서를 참조하세요.

## 벤치마크 결과

### 임베딩 모델 비교

| 모델 | F1 점수 | MRR | NDCG | 평균 검색 시간 |
|------|---------|-----|------|--------------|
| jhgan/ko-sroberta-multitask | 0.9259 | 0.9611 | 0.9913 | 0.59s |
| BM-K/KoSimCSE-roberta-multitask | 0.8870 | 0.9380 | 0.9782 | 0.65s |
| snunlp/KR-SBERT-V40K-klueNLI | 0.8705 | 0.9125 | 0.9656 | 0.61s |

### 재순위 모델 비교

| 재순위 모델 | F1 점수 | MRR | NDCG | 재순위 시간 |
|------------|---------|-----|------|------------|
| 없음 (기본 검색) | 0.9259 | 0.9611 | 0.9913 | - |
| jhgan/ko-sroberta-multitask | 0.9259 | 0.9233 | 0.9476 | 0.45s |
| cross-encoder/ms-marco-MiniLM-L-6-v2 | 0.9259 | 0.9611 | 0.9933 | 0.10s |

### 질문 유형별 성능 (ko-sroberta-multitask)

| 질문 유형 | 정밀도 | 재현율 | F1 점수 | MRR | NDCG |
|----------|-------|-------|---------|-----|------|
| 사실적 | 0.86 | 0.95 | 0.897 | 0.95 | 0.987 |
| 구조적 | 0.70 | 0.79 | 0.729 | 0.83 | 0.968 |
| 절/항 | 1.00 | 1.00 | 1.000 | 1.00 | 1.000 |
| 추론적 | 0.96 | 1.00 | 0.978 | 1.00 | 1.000 |
| 관계적 | 1.00 | 1.00 | 1.000 | 1.00 | 1.000 |

## 한국어 임베딩 모델 추천

본 프로젝트에서 평가한 결과, 다음 모델들이 한국어 RAG 시스템에 적합합니다:

1. **jhgan/ko-sroberta-multitask**
   - 전반적인 성능이 가장 우수
   - 한국어 문장 유사도에 최적화됨
   - 대부분의 질문 유형에서 높은 성능 발휘

2. **BM-K/KoSimCSE-roberta-multitask**
   - 유사도 검색에서 좋은 성능
   - 컨텍스트 이해도가 높음
   - 구조적 질문에서 특히 강점

3. **cross-encoder/ms-marco-MiniLM-L-6-v2** (재순위 모델로 사용)
   - 빠른 처리 속도와 우수한 정확도
   - 검색 결과 재정렬에 효과적
   - 다국어 지원에도 한국어 성능 우수

## 성능 향상 팁

1. **임베딩 모델 선택**:
   - 일반적 용도: jhgan/ko-sroberta-multitask
   - 다국어 지원 필요: sentence-transformers/paraphrase-multilingual-mpnet-base-v2
   - 컨텍스트 집중: BM-K/KoSimCSE-roberta-multitask

2. **검색 파라미터 최적화**:
   - 유사도 임계값: 0.05-0.1 (청크 품질에 따라 조정)
   - 상위 K: 5-10 (정밀도와 재현율 균형)
   - Small-to-Big 활성화로 맥락 이해 향상

3. **청킹 최적화**:
   - 작은 청크: 150-300 토큰
   - 큰 청크: 1000-2000 토큰
   - 오버랩: 작은 청크는 30 토큰, 큰 청크는 0 토큰

4. **재순위화 활용**:
   - cross-encoder/ms-marco-MiniLM-L-6-v2: 빠른 처리 속도
   - jhgan/ko-sroberta-crossencoder: 한국어 최적화 (느림)

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 기여하기

이슈 및 풀 리퀘스트를 통해 프로젝트 개선에 기여할 수 있습니다. 버그 신고, 기능 제안, 코드 개선 등 모든 형태의 기여를 환영합니다.

1. 저장소 포크
2. 기능 브랜치 생성 (`git checkout -b feature/amazing-feature`)
3. 변경사항 커밋 (`git commit -m 'Add amazing feature'`)
4. 브랜치 푸시 (`git push origin feature/amazing-feature`)
5. 풀 리퀘스트 제출

## 업데이트 내역

- **2025.05.24**: 검색 시스템 테스트 기능 대폭 개선 - 성능 측정, 유사도 통계 분석, 상세한 디버깅 정보 추가로 100% 테스트 성공률 달성
- **2025.05.19**: Small-to-Big 청킹 전략 가이드 추가, 재순위 모델 비교 평가 기능 추가
- **2025.05.15**: 자동 평가 시스템 개선, 다양한 한국어 임베딩 모델 테스트 추가
- **2025.05.10**: Milvus 컬렉션 관리 기능 개선, 에러 처리 강화
- **2025.05.01**: 초기 버전 출시, 기본 RAG 기능 구현

## 구현 예정 기능

- **멀티모달 지원**: 이미지와 텍스트 함께 처리
- **하이브리드 검색**: 키워드와 벡터 검색 결합
- **추가 벡터 DB 지원**: Pinecone, ChromaDB 통합
- **프롬프트 최적화**: 응답 생성을 위한 프롬프트 템플릿 개선
- **웹 인터페이스**: 검색 및 평가 결과 시각화
