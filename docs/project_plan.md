# RAG PDF Parsing & Embedding Strategy with Milvus

## 프로젝트 개요

본 프로젝트는 한국어 문서 처리에 최적화된 검색 증강 생성(Retrieval-Augmented Generation, RAG) 시스템을 개발하는 것을 목표로 합니다. 특히 PDF 문서의 효과적인 파싱과 임베딩 전략, 그리고 Milvus 벡터 데이터베이스를 활용한 검색 성능 향상에 중점을 두고 있습니다.

## 프로젝트 상태 업데이트

### 2025-05-18: 프로젝트 구조 및 기능 파악, Shrimp 규칙 문서 생성

현재 프로젝트 상태를 파악하였습니다. 프로젝트는 다음과 같은 주요 구성 요소로 이루어져 있습니다:

1. **문서 파싱 (parser.py)**
   - PDF, CSV, 텍스트 파일 등 다양한 문서 형식 지원
   - Small-to-Big 청킹 전략 구현 (작은 청크로 검색, 큰 청크로 컨텍스트 제공)
   - 토큰 기반 청킹 및 섹션 경계 인식

2. **임베딩 생성 (embedder.py)**
   - 다양한 한국어 임베딩 모델 지원 (KoSimCSE, KoSBERT, 다국어 MPNet 등)
   - 부모-자식 청크 관계 유지 및 메타데이터 관리
   - GPU 가속 지원

3. **문서 검색 (retriever.py)**
   - Milvus 벡터 데이터베이스 기반 유사도 검색
   - 하이브리드 검색 (벡터 + 키워드)
   - 쿼리 최적화 및 결과 재순위화

4. **응답 생성 (generator.py)**
   - 다양한 LLM API 지원 (Grok, OpenAI, Anthropic)
   - 컨텍스트 기반 응답 생성
   - 한국어 특화 프롬프트 템플릿

5. **CLI 도구 (main.py)**
   - 문서 처리, 임베딩, 쿼리 명령 지원
   - 인터랙티브 모드 제공
   - 다양한 설정 옵션

#### 주요 구현 기능 및 특징

프로젝트 구조와 코드 분석을 마치고 Shrimp Task Manager를 초기화했습니다. 개발 표준과 코드 규칙을 정의한 shrimp-rules.md 문서를 생성하여 프로젝트의 일관성을 유지할 수 있게 되었습니다.

#### 규칙 문서 주요 내용

- 프로젝트 구조 및 아키텍처 가이드라인
- 코드 표준 (명명 규칙, 주석, 오류 처리)
- 기능 구현 표준 (모듈별 확장 방법)
- 프레임워크/라이브러리 사용 표준
- 워크플로우 표준 및 핵심 파일 상호작용
- AI 의사결정 표준 및 금지 사항



- **Small-to-Big 전략**: 작은 청크로 정확한 검색을 수행하고, 검색된 청크의 부모를 통해 더 넓은 컨텍스트 제공
- **하이브리드 검색**: 벡터 검색과 키워드 검색을 결합하여 더 정확한 결과 제공
- **한국어 최적화**: 한국어 특화 임베딩 모델 및 문장 분리기 활용
- **Milvus 통합**: 고성능 벡터 검색을 위한 Milvus 벡터 데이터베이스 지원
- **유연한 설정**: YAML 기반 설정 파일을 통한 다양한 옵션 제공

### 2025-05-17: Small-to-Big 전략 구현 완료

Small-to-Big 청킹 전략 구현을 통해 RAG 시스템의 검색 정확도를 향상시켰습니다. 
이 전략은 작은 청크(자식 청크)로 정확한 검색을 수행하고, 검색된 청크의 부모 청크를 LLM에 제공하여 
풍부한 컨텍스트를 제공하는 방식으로 동작합니다.

#### 완료된 작업

1. **DocumentParser 클래스 개선**
   - InsurancePDFParser에서 더 일반적인 DocumentParser로 이름 변경
   - PDF, CSV, TXT 등 다양한 문서 유형 지원
   - 토큰 기반 청킹 구현 (정확한 토큰 길이 계산)
   - Small-to-Big 전략을 위한 계층 구조 생성

2. **Embedder 개선**
   - 자식 청크와 부모 청크 모두 처리 가능
   - 부모-자식 관계 메타데이터 유지
   - Milvus에 청크 타입 정보 저장

3. **Retriever 개선**
   - Small-to-Big 검색 전략 구현
   - 부모 청크 ID를 기반으로 부모 컨텍스트 검색
   - 하이브리드 검색에서도 Small-to-Big 지원

4. **CLI 및 인터랙티브 모드 개선**
   - 새로운 Small-to-Big 토글 명령어 추가
   - 부모/자식 청크 전환 기능
   - 검색 및 컨텍스트 정보 표시

#### 주요 청킹 파라미터

- **자식 청크 (검색용)**: 
  - 크기: 250 토큰
  - 오버랩: 30 토큰

- **부모 청크 (컨텍스트용)**: 
  - 크기: 1500 토큰
  - 오버랩: 0 토큰

- **일반 텍스트 파일 청킹**:
  - 크기: 300 토큰
  - 오버랩: 30 토큰

## 현재 과제 및 문제점

1. **특수 용어 검색 문제**
   - "제1보험기간"과 같은 특수 용어/법률 용어 검색 성능 개선 필요
   - 쿼리 최적화 과정에서 정확한 용어 검색이 방해받는 문제
   - 한국어 형태소 분석 및 토큰화 개선 필요

2. **Milvus API 호환성 문제**
   - 현재 사용 중인 Milvus Python SDK 버전이 일부 매개변수 지원하지 않음
   - API 변경에 따른 코드 수정 필요

3. **Small-to-Big 확장 로직 개선**
   - 부모-자식 관계 추적 메커니즘 개선
   - 검색 결과에서 부모 청크 정보 로드 효율화

## 다음 개선 계획

1. **검색 성능 개선**
   - 특수 용어 검색을 위한 쿼리 최적화 비활성화 옵션 추가
   - 유사도 임계값 자동 조정 메커니즘 구현
   - 청크 타입 필터 유연성 강화

2. **시스템 안정성 확보**
   - Milvus 클라이언트 에러 처리 강화
   - API 변경에 대비한 유연한 인터페이스 구현
   - 로깅 및 모니터링 개선

3. **성능 테스트 및 최적화**
   - 다양한 청크 크기 및 오버랩 설정 테스트
   - 특수 쿼리("제1보험기간" 등) 검색 성능 벤치마크
   - 임베딩 모델 성능 비교 평가

4. **문서화 및 사용성 개선**
   - 상세 사용 설명서 작성
   - 예제 및 튜토리얼 추가
   - 설정 옵션 가이드 개선

## 완료된 작업

- [x] 기본 프로젝트 구조 설계
- [x] PDF 파싱 모듈 구현
- [x] 임베딩 모듈 구현
- [x] Milvus 클라이언트 구현
- [x] 검색 모듈 구현
- [x] 응답 생성 모듈 구현
- [x] CLI 인터페이스 구현
- [x] Small-to-Big 전략 구현
- [x] 하이브리드 검색 구현
- [x] Docker 기반 Milvus 설정
- [x] Shrimp Task Manager 초기화
- [x] 프로젝트 규칙 문서(shrimp-rules.md) 생성

## 진행 중인 작업

- [ ] 특수 용어 검색 성능 개선
- [ ] Milvus API 호환성 문제 해결
- [ ] Small-to-Big 확장 로직 최적화
- [ ] 성능 테스트 및 벤치마크

## 참고 자료

- [LangChain ParentDocumentRetriever](https://python.langchain.com/docs/modules/data_connection/retrievers/parent_document_retriever)
- [Small-to-Big RAG 전략](https://blog.langchain.dev/small-to-big-retrieval-for-rag/)
- [청킹 전략과 오버랩 설정 가이드](https://korn-sudo.github.io/korn-sudo.github.io/posts/chunking-strategies/)
- [Milvus Python SDK 문서](https://milvus.io/docs/install-pymilvus.md)
