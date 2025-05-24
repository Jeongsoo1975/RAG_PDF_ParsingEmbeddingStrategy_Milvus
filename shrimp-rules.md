# RAG 시스템 개발 가이드라인

## 프로젝트 개요

**목적**: RAG(Retrieval Augmented Generation) 시스템 성능 개선 및 유지보수
**기술 스택**: Python, Milvus, sentence-transformers, Grok API, Docker
**핵심 기능**: PDF/CSV 파싱, 임베딩 생성, 벡터 검색, 하이브리드 검색, 응답 생성

## 프로젝트 아키텍처

### 핵심 모듈 구조
- `src/main.py` - CLI 인터페이스 및 전체 파이프라인 조정
- `src/rag/parser.py` - 문서 파싱 (PDF, CSV, TXT)
- `src/rag/embedder.py` - 임베딩 생성 및 벡터 DB 저장
- `src/rag/retriever.py` - 검색 및 재순위화
- `src/rag/generator.py` - 응답 생성
- `src/evaluation/` - 성능 평가 모듈
- `configs/default_config.yaml` - 시스템 설정

### 데이터 흐름
1. 문서 파싱 → 청킹 → 임베딩 생성 → Milvus 저장
2. 쿼리 입력 → 벡터 검색 → 재순위화 → 컨텍스트 추출 → LLM 응답 생성

## 코딩 표준

### 파일 수정 순서
- **필수**: 설정 변경 시 항상 `configs/default_config.yaml` 먼저 수정
- **필수**: 코드 변경 후 반드시 관련 테스트 실행
- **필수**: 성능 개선 작업 후 평가 실행으로 효과 검증

### 명명 규칙
- 변수명: snake_case (예: `embedding_model`, `top_k_results`)
- 함수명: snake_case (예: `generate_embeddings`, `hybrid_retrieve`)
- 클래스명: PascalCase (예: `DocumentRetriever`, `ResponseGenerator`)
- 상수명: UPPER_SNAKE_CASE (예: `DEFAULT_CHUNK_SIZE`, `MAX_TOKENS`)

### 로깅 규칙
- **필수**: 모든 성능 관련 작업에 INFO 레벨 로그 기록
- **필수**: 에러 발생 시 ERROR 레벨로 스택 트레이스 포함
- **필수**: 로그는 `logs/` 폴더에 저장
- 로그 포맷: `[TIMESTAMP] [LEVEL] [MODULE] - MESSAGE`

## 성능 개선 작업 표준

### 검색 성능 개선 우선순위
1. **하이브리드 검색 활성화** (즉시 적용 가능)
2. **Re-ranking 모델 도입** (단기)
3. **임베딩 모델 업그레이드** (중기)
4. **데이터 품질 개선** (장기)

### 하이브리드 검색 구현 규칙
- **필수**: `configs/default_config.yaml`에서 `retrieval.hybrid_search: true` 설정
- **필수**: `retrieval.hybrid_alpha`는 0.7 (벡터 70%, 키워드 30%) 유지
- **필수**: `retrieval.top_k`를 15로 설정하여 재순위화용 후보군 확보
- **금지**: 하이브리드 검색 없이 단순 벡터 검색만 사용

### Re-ranking 모델 적용 규칙
- **필수**: `configs/default_config.yaml`에서 `retrieval.reranking: true` 설정
- **우선 모델**: `jhgan/ko-sroberta-multitask` (한국어 최적화)
- **백업 모델**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **필수**: 배치 크기는 32로 설정 (`retrieval.reranking_batch_size: 32`)

### 임베딩 모델 변경 규칙
- **필수**: 모델 변경 시 기존 임베딩 데이터 백업
- **필수**: 새 모델로 전체 데이터 재임베딩 필요
- **필수**: 변경 후 성능 평가로 개선 효과 검증
- **금지**: 다른 차원의 임베딩 모델 혼용

## 설정 파일 관리 표준

### configs/default_config.yaml 수정 규칙
- **필수**: 변경 전 현재 설정값 주석으로 백업
- **필수**: 변경 이유와 예상 효과를 주석으로 기록
- **금지**: 한 번에 여러 영역 동시 변경 (디버깅 어려움)
- **필수**: 변경 후 즉시 Git 커밋

### 성능 튜닝 파라미터
```yaml
# 검색 성능 개선용 권장 설정
retrieval:
  top_k: 15  # 재순위화용 후보군 확보
  similarity_threshold: 0.65  # 너무 높으면 결과 부족
  hybrid_search: true  # 필수 활성화
  hybrid_alpha: 0.7  # 벡터 검색 가중치
  reranking: true  # 정밀도 향상
```

## 테스트 및 평가 표준

### 필수 테스트 순서
1. **기본 연결 테스트**: `test_milvus_connection.py`
2. **임베딩 테스트**: `test_language_override.py`
3. **검색 테스트**: `test_vector_search.py`
4. **전체 파이프라인**: `integration_example.py`

### 성능 평가 실행 규칙
- **필수**: 모든 성능 개선 작업 후 평가 실행
- **명령어**: `python -m src.evaluation.run_evaluation`
- **필수 지표**: Context Relevancy, Answer Relevancy, Faithfulness
- **목표 지표**: 전체 점수 0.75 이상
- **필수**: 평가 결과를 `evaluation_results/` 폴더에 저장

### A/B 테스트 규칙
- **필수**: 기준선(baseline) 성능 먼저 측정
- **필수**: 한 번에 하나의 변수만 변경
- **필수**: 최소 10개 이상의 테스트 쿼리로 검증
- **필수**: 통계적 유의성 확인

## 데이터 관리 표준

### 벡터 데이터베이스 관리
- **필수**: Milvus 컬렉션명은 `sanitize_name()` 함수로 정규화
- **필수**: 임베딩 변경 시 기존 컬렉션 백업 후 삭제
- **금지**: 개발 중 프로덕션 컬렉션 직접 수정
- **필수**: 컬렉션 생성 시 인덱스 타입 HNSW 사용

### 청킹 전략 관리
- **현재 전략**: Small-to-Big 청킹 (자식 청크 600자, 부모 청크 1400자)
- **필수**: 청크 크기 변경 시 기존 데이터와 호환성 확인
- **필수**: `chunk_overlap`은 `chunk_size`의 20-25% 유지
- **금지**: 청킹 전략 변경 시 기존 메타데이터 손실

## 프레임워크 및 라이브러리 사용 표준

### sentence-transformers 사용 규칙
- **필수**: 모델 로드 시 `normalize_embeddings=True` 설정
- **권장 모델**: `paraphrase-multilingual-mpnet-base-v2` (기본)
- **한국어 특화**: `jhgan/ko-sroberta-multitask` (성능 우수)
- **필수**: GPU 메모리 부족 시 배치 크기 16 이하로 조정

### Milvus 클라이언트 사용 규칙
- **필수**: 연결 타임아웃 30초 설정
- **필수**: 에러 발생 시 자동 재연결 로직 포함
- **필수**: 인덱스 파라미터: M=16, efConstruction=256
- **필수**: 검색 파라미터: ef=64

### Docker 환경 관리
- **필수**: `docker-compose.yml` 사용하여 Milvus 실행
- **필수**: 볼륨 마운트로 데이터 영속성 보장
- **금지**: 컨테이너 내부 데이터 직접 수정

## 워크플로우 표준

### 성능 개선 워크플로우
1. **현재 성능 측정**: 기준선 평가 실행
2. **설정 변경**: `configs/default_config.yaml` 수정
3. **코드 수정**: 필요시 관련 모듈 업데이트
4. **테스트 실행**: 기본 기능 동작 확인
5. **성능 재평가**: 개선 효과 측정
6. **결과 비교**: A/B 테스트 결과 분석
7. **커밋**: 개선 사항 Git 커밋

### 문제 해결 워크플로우
1. **로그 확인**: `logs/` 폴더의 최신 로그 검토
2. **연결 테스트**: Milvus 연결 상태 확인
3. **설정 검증**: `configs/default_config.yaml` 문법 확인
4. **격리 테스트**: 문제 모듈 단독 테스트
5. **이전 상태 복원**: 필요시 이전 커밋으로 롤백

## 핵심 파일 상호작용 표준

### 동시 수정 필요 파일들
- **임베딩 모델 변경 시**:
  - `configs/default_config.yaml` (모델명, 차원수)
  - `src/rag/embedder.py` (모델 로드 로직)
  - 기존 벡터 데이터 재생성 필요

- **검색 파라미터 변경 시**:
  - `configs/default_config.yaml` (검색 설정)
  - `src/rag/retriever.py` (검색 로직)
  - 평가 스크립트로 효과 검증

- **청킹 전략 변경 시**:
  - `configs/default_config.yaml` (청킹 설정)
  - `src/rag/parser.py` (청킹 로직)
  - 기존 파싱 데이터 재생성 필요

## AI 의사결정 표준

### 성능 개선 우선순위 결정
1. **현재 지표 분석**: Context Relevancy < 0.3 → 검색 개선 우선
2. **Answer Relevancy < 0.2** → 응답 생성 개선 우선
3. **Faithfulness < 0.5** → 컨텍스트 품질 개선 우선
4. **응답 시간 > 5초** → 효율성 개선 우선

### 모델 선택 기준
- **한국어 데이터 70% 이상**: 한국어 특화 모델 우선
- **다국어 데이터**: 다국어 모델 사용
- **응답 속도 중요**: 경량 모델 선택
- **정확도 최우선**: 대형 모델 사용

### 하이퍼파라미터 튜닝 기준
- **top_k**: 재순위화 시 15-20, 단순 검색 시 5-10
- **similarity_threshold**: 0.6-0.7 (너무 높으면 결과 부족)
- **chunk_size**: 500-800자 (한국어 기준)
- **chunk_overlap**: chunk_size의 20-25%

## 금지사항

### 절대 금지
- **프로덕션 데이터 직접 삭제**
- **설정 파일 백업 없이 대규모 변경**
- **평가 없이 성능 개선 주장**
- **하드코딩된 임베딩 차원 수정**
- **Milvus 컨테이너 내부 파일 직접 수정**

### 주의사항
- **임베딩 모델 변경**: 기존 벡터 데이터와 호환되지 않음
- **청킹 전략 변경**: 전체 데이터 재처리 필요
- **Docker 컨테이너 재시작**: 임시 데이터 손실 가능
- **설정 값 극단적 변경**: 시스템 불안정 야기 가능

### 경고 상황
- **메모리 사용량 > 80%**: 배치 크기 조정 필요
- **응답 시간 > 10초**: 파라미터 최적화 필요
- **검색 결과 < 3개**: 임계값 조정 필요
- **오류율 > 5%**: 시스템 점검 필요

## 예시 - 올바른 접근

### ✅ 하이브리드 검색 활성화
```python
# 1. 설정 파일 수정
config['retrieval']['hybrid_search'] = True
config['retrieval']['top_k'] = 15

# 2. 기능 테스트
results = retriever.hybrid_retrieve(query, top_k=15)

# 3. 성능 평가
evaluation_results = run_evaluation()

# 4. 결과 비교 및 커밋
if evaluation_results['overall_score'] > baseline_score:
    git_commit("feat: enable hybrid search")
```

### ❌ 잘못된 접근
```python
# 설정 변경 없이 코드만 수정 (일관성 없음)
results = retriever.hybrid_retrieve(query, hybrid_alpha=0.8)

# 평가 없이 성능 개선 주장
print("하이브리드 검색으로 성능 향상됨")  # 검증 없음

# 여러 변경사항 동시 적용
config['embedding']['model'] = 'new-model'
config['retrieval']['top_k'] = 20
config['chunking']['chunk_size'] = 1000  # 디버깅 어려움
```

## 예시 - 올바른 성능 개선 프로세스

### ✅ 단계적 개선
```python
# Phase 1: 기준선 측정
baseline = run_evaluation()
print(f"기준선 성능: {baseline['overall_score']}")

# Phase 2: 하이브리드 검색 적용
config['retrieval']['hybrid_search'] = True
hybrid_results = run_evaluation()
print(f"하이브리드 성능: {hybrid_results['overall_score']}")

# Phase 3: 재순위화 추가
config['retrieval']['reranking'] = True
final_results = run_evaluation()
print(f"최종 성능: {final_results['overall_score']}")

# Phase 4: 결과 분석 및 결정
if final_results['overall_score'] > baseline['overall_score']:
    commit_changes("성능 개선 확인됨")
else:
    revert_changes("성능 개선 효과 없음")
```

이 가이드라인을 준수하여 RAG 시스템의 성능을 체계적이고 안전하게 개선하세요.# RAG PDF 파싱 임베딩 전략 Milvus 프로젝트 - AI Agent 개발 규칙

## 프로젝트 개요

**핵심 목적**: RAG(Retrieval-Augmented Generation) 시스템에서 PDF 문서 파싱, 임베딩 생성, Milvus 벡터 DB 활용을 통한 문서 검색 및 평가 시스템
**기술 스택**: Python, Milvus, RAGAS, 한국어 임베딩 모델, TF-IDF, 다양한 리랭커
**핵심 기능**: 문서 임베딩, 벡터 검색, RAG 평가, 성능 최적화

## 프로젝트 아키텍처

### 주요 디렉토리 구조
- `src/` - 메인 소스 코드
  - `evaluation/` - 평가 관련 모듈
  - `parsers/` - 문서 파싱 모듈  
  - `rag/` - RAG 시스템 구현
  - `vectordb/` - Milvus 벡터 DB 연결
  - `utils/` - 공통 유틸리티
- `data/` - 데이터 파일 (amnesty_qa 등)
- `logs/` - 모든 로그 파일 저장소
- `evaluation_results/` - 평가 결과 저장소
- `tests/` - 테스트 파일

### 핵심 모듈 연관성
- evaluation 모듈들은 상호 의존적이며 동시 수정 필요
- Milvus 연결 설정 변경 시 vectordb 및 모든 evaluation 파일 점검 필요
- 로그 설정은 모든 모듈에서 일관성 유지 필요

## 🚨 **가짜 데이터 처리 금지 규칙**

### **절대 금지 사항**
- **더미 데이터(dummy data) 생성 금지**
- **샘플 데이터(sample data) 임의 생성 금지**  
- **모의(mock) 답변 생성기 사용 금지**
- **가짜 임베딩 생성 금지**
- **임의의 테스트 값 생성 금지**

### **실제 데이터만 사용 원칙**
- 제공된 실제 데이터만 사용
- 실제 LLM API 연결만 허용
- 실제 벡터 임베딩 모델만 사용
- 실제 Milvus 벡터 DB 연결만 사용

### **기존 가짜 데이터 처리 코드 제거 대상**
- `MockAnswerGenerator` 클래스 (step3_standard_ragas_evaluator.py)
- `SimpleTFIDFEmbedder` 클래스 (step3_standard_ragas_evaluator.py)
- 보험 데이터 변환 로직 (insurance_eval_dataset.json)
- 기타 모든 가짜/더미 데이터 생성 코드

## 코딩 표준

### 파일 작업 규칙
- **파일 수정 시 3-5개 섹션으로 분할하여 순차 작업**
- **write_file 후 반드시 edit_file_lines로 추가 작업**
- **edit_file_lines 작업 전 반드시 "dryRun": true 설정**
- **각 edit 작업 전 파일 내용 재확인 필수**

### 로깅 표준
- 모든 로그는 `C:\Users\USER\Documents\MCPData\RAG_PDF_ParsingEmbeddingStrategy_Milvus\logs`에 저장
- 로그 파일명: `{모듈명}_{timestamp}.log` 형식
- 로그 레벨: INFO (콘솔), DEBUG (파일)
- UTF-8 인코딩 사용

### Git 작업 플로우
- 파일 생성/수정 후 반드시 `git add` + `git commit` 실행
- 커밋 메시지 형식: `feat:`, `fix:`, `test:`, `docs:` 접두사 사용
- 테스트 브랜치에서 검증 후 master 병합
- 삭제 시 `git rm` 사용

## 프레임워크/라이브러리 사용 표준

### Milvus 연결
- MilvusClient 사용
- 연결 실패 시 로그 기록 후 graceful 종료
- 벡터 차원 일관성 유지

### RAGAS 평가
- SimpleRAGASMetrics 클래스 사용
- 5개 표준 지표: context_relevancy, context_precision, context_recall, faithfulness, answer_relevancy
- 평가 결과는 JSON 형태로 evaluation_results에 저장

### MySQL 연결 (필요시)
- 명령 형식: `mysql -uroot -p -e "SHOW DATABASES;" database_name`
- 쿼리문은 반드시 따옴표로 감싸기

## 워크플로우 표준

### 평가 워크플로우
1. 데이터 로드 (amnesty_qa)
2. 실제 벡터 검색 (Milvus)
3. 실제 LLM 답변 생성
4. RAGAS 지표 계산
5. 결과 저장 및 리포트 생성

### 테스트 워크플로우
- 기능별 단위 테스트 우선
- 통합 테스트는 실제 DB 연결 필요
- 테스트 실패 시 로그 확인 후 오류 수정

## 핵심 파일 상호작용 표준

### 동시 수정 필요 파일들
- `step3_standard_ragas_evaluator.py` 수정 시 → `simple_ragas_metrics.py` 점검
- Milvus 설정 변경 시 → 모든 evaluation 파일 점검
- 로그 설정 변경 시 → src/ 하위 모든 모듈 점검

### 데이터 파일 의존성
- `amnesty_qa_evaluation.json` → evaluation 모듈들
- `amnesty_qa_milvus_data.json` → 검색 관련 모듈들
- 설정 파일 변경 시 관련 모든 모듈 재시작 필요

## AI 의사결정 표준

### 가짜 데이터 관련 결정
1. **모든 가짜 데이터 코드는 무조건 제거**
2. **실제 구현체로 교체 불가능한 경우 기능 비활성화**
3. **테스트 목적이라도 가짜 데이터 생성 금지**

### 오류 처리 우선순위
1. 로그 확인
2. 실제 데이터 연결 상태 점검
3. Milvus 연결 상태 확인
4. 실제 API 키/연결 설정 점검

### 성능 최적화 우선순위
1. 실제 벡터 검색 최적화
2. 실제 LLM 응답 시간 개선
3. Milvus 인덱스 튜닝
4. 메모리 사용량 최적화

## 금지 행동

### **절대 수행 금지**
- 가짜/더미/샘플 데이터 생성
- MockAnswerGenerator 같은 모의 구현체 사용
- SimpleTFIDFEmbedder 같은 간소화된 임베더 사용
- 임의의 테스트 값이나 하드코딩된 응답 생성
- 실제 API 없이 가상의 응답 생성

### **신중히 수행**
- Milvus 스키마 변경
- 평가 지표 수정
- 로그 레벨 변경
- Git 히스토리 수정

### **사전 동의 필요**
- 새로운 evaluation 모듈 추가
- 기존 데이터 구조 변경
- 프로젝트 아키텍처 수정
- Shrimp Task Manager 작업 초기화

## 특별 지침

### TaskPlanner 모드
- 새 기능 개발 요청 시 plan_task 도구만 사용
- 작업을 1-2일 단위로 분할
- 최대 10개 이하 작업으로 제한
- 명확한 완료 기준 포함

### TaskExecutor 모드  
- execute_task → verify_task → complete_task 순서 준수
- 중간 결과만 간결히 보고
- 터미널/파일 작업은 MCP 도구 사용

### 연속 실행 모드
- 여러 작업 자동 처리 시 사용자 동의 필요
- 각 작업 완료 후 검증 단계 거치기
- 실패 시 즉시 중단 후 보고

---

**중요**: 이 규칙들은 AI Agent의 작업 수행을 위한 것이며, 모든 가짜 데이터 처리를 완전히 제거하여 실제 RAG 시스템 구축에 집중합니다.