# RAG PDF Parsing & Embedding Strategy with Milvus

## 프로젝트 상태 업데이트

### 2025-05-22: RAGAS 평가 결과 시각화 모듈 구현 완료

RAGAS 평가 결과를 종합적으로 분석하고 시각화하는 웹 기반 보고서 시스템을 구현했습니다. 이를 통해 다양한 임베딩 모델과 리랭커 조합의 성능을 효과적으로 비교 분석할 수 있습니다.

#### 구현된 시각화 기능

1. **종합 성능 대시보드**
   - Bootstrap 기반 반응형 웹 인터페이스
   - 모든 차트와 테이블을 단일 HTML 파일로 통합
   - Base64 인코딩으로 이미지 임베딩 (외부 파일 의존성 없음)

2. **다양한 시각화 차트**
   - **지표별 성능 비교**: 막대 그래프로 RAGAS 5개 지표 동시 비교
   - **레이더 차트**: 각 모델 구성의 종합적 성능 프로필
   - **랭킹 히트맵**: 지표별 순위를 직관적으로 표현
   - **시간별 성능 추이**: 평가 시점별 성능 변화 추적
   - **임베딩-리랭커 조합 효과**: 조합별 성능 매트릭스 히트맵

3. **데이터 처리 및 분석**
   - JSON 평가 결과 자동 로딩 및 파싱
   - 모델명 자동 추출 및 표시명 변환
   - 타임스탬프 기반 시계열 분석
   - 평균 점수 자동 계산 및 순위 매김

#### 기술적 구현 특징

**🏗️ 모듈화 아키텍처**
```
src/evaluation/web/
├── visualization.py          # 기본 설정 및 데이터 로딩
├── visualization_part2.py    # 데이터 처리 및 기본 시각화
├── visualization_part3.py    # 시간별 추이 분석
├── visualization_part4.py    # 고급 시각화 함수
├── visualization_part5.py    # HTML 템플릿 시스템
├── visualization_part6.py    # 통합 보고서 생성
└── visualization_main.py     # 메인 인터페이스
```

**📊 지원하는 평가 지표**
- Context Precision: 검색된 컨텍스트의 정확도
- Context Recall: 필요한 컨텍스트의 재현율
- Faithfulness: 생성된 답변의 신뢰도
- Answer Relevancy: 답변의 관련성
- Context Relevancy: 컨텍스트의 관련성
- Average Score: 전체 평균 점수

**🎨 시각화 스타일링**
- 일관된 색상 팔레트 적용
- seaborn 기반 전문적 차트 스타일
- 고해상도(300 DPI) 이미지 생성
- 모바일/데스크톱 반응형 디자인

#### 사용 방법

**기본 실행:**
```bash
python run_visualization.py
```

**특정 평가 결과 시각화:**
```bash
python run_visualization.py --timestamp 20250519_203504 --output-dir custom_results
```

**프로그래밍 방식 사용:**
```python
from src.evaluation.web.visualization_main import create_visualization_report

# 전체 보고서 생성
report_path = create_visualization_report(output_dir='results')

# 개별 차트만 생성
from src.evaluation.web.visualization_main import plot_all_charts
charts = plot_all_charts(output_dir='charts')
```

#### 데이터 플로우

```
[evaluation_results/*.json] 
    ↓ 자동 스캔
[데이터 로딩 & 파싱]
    ↓ 모델명 추출 & 변환
[pandas DataFrame 변환]
    ↓ 타임스탬프별 그룹화
[각종 시각화 생성]
    ↓ Base64 인코딩
[HTML 템플릿 렌더링]
    ↓ 단일 파일 출력
[results/ragas_report_[timestamp].html]
```

#### 분석 가능한 인사이트

1. **모델 성능 비교**: 임베딩 모델별 강점/약점 파악
2. **리랭커 효과**: 리랭커 적용 시 성능 개선 정도
3. **시간별 변화**: 모델 튜닝 및 데이터 변화에 따른 성능 추이
4. **최적 조합 발견**: 특정 지표에서 최고 성능을 보이는 구성
5. **균형잡힌 성능**: 모든 지표에서 안정적인 성능을 보이는 구성

### 2025-05-18: 검색 문제 해결 및 성능 개선

### 2025-05-18: 검색 문제 해결 및 성능 개선

RAG 시스템의 검색 문제를 해결하여 "제1보험기간"과 같은 특수 용어에 대한 검색 성능을 크게 개선했습니다. 이 과정에서 발견된 여러 문제점들과 해결 방법을 정리합니다.

#### 해결된 문제

1. **Milvus API 호환성 오류 수정**
   - 오류: `MilvusClient.insert() got an unexpected keyword argument 'partition'`
   - 원인: 현재 사용 중인 Milvus Python SDK 버전이 `partition` 매개변수를 지원하지 않음
   - 해결책: 
     - embedder.py의 `_store_chunks_in_milvus` 메서드에서 `partition` 매개변수 제거
     - milvus_client.py의 `insert` 메서드에서 호출 시 `partition_name` 매개변수 제거

2. **특수 용어 검색 실패 문제 개선**
   - 문제: "제1보험기간"과 같은 특수 용어/법률 용어 검색 결과 부재
   - 원인들:
     - 쿼리 최적화 과정에서 "정의" 자동 추가 로직이 정확한 용어 검색 방해
     - 과도하게 높은 유사도 임계값(threshold)으로 결과 필터링 
     - 청크 타입 필터가 관련 결과 제외
   - 해결책:
     - 쿼리 최적화 비활성화 (`enable_query_optimization=False`)
     - 유사도 임계값 0.7에서 0.5로 낮춤
     - 필터 표현식 옵션 제공 (`force_filter_expr=None`)으로 필터 비활성화

3. **Small-to-Big 확장 로직 개선**
   - 청크 타입 메타데이터 필드 추가로 부모-자식 관계 추적 용이
   - 임베딩 과정에서 부모 청크 메타데이터 저장 및 연결
   - 검색 결과에서 부모 청크 정보 로드 및 컨텍스트 확장

#### 최적화된 검색 워크플로우

1. **단순 벡터 검색부터 시작**
   - 쿼리 최적화 없이 원본 쿼리 그대로 사용
   - 필터 없이 검색하여 낮은 유사도(0.3~0.5)라도 관련 결과 확인
   - 유사도 임계값을 낮게 설정하고 점진적으로 조정

2. **필터 점진적 적용**
   - 검색 결과 확인 후 필요시 chunk_type 필터 활성화
   - Small-to-Big 전략을 위한 부모 청크 로딩 추가
   - 필요한 경우 쿼리 최적화 활성화

3. **파라미터 튜닝**
   - 유사도 임계값(threshold)과 top_k 조정으로 결과 품질 향상
   - 하이브리드 검색에서 벡터 검색과 키워드 검색 비중 조정

#### 기술적 개선 사항

1. **키워드 추출 개선**
   - 원본 쿼리에서 공백 제거한 용어 자체를 키워드에 추가
   - 한국어 형태소 분석기(konlpy.tag.Okt) 도입 권장
   - 불필요한 조사/어미 제거 로직 보수적 수정

2. **오프라인 검색 대체 메커니즘**
   - 검색 결과가 없을 때 오프라인 검색으로 폴백
   - 더미 데이터 추가로 최소한의 관련성 있는 결과 제공

3. **새로운 CLI 명령어 추가**
   - `Query> mode vector` - 벡터 검색만 활성화
   - `Query> s2b off` - Small-to-Big 비활성화
   - `Query> details on` - 상세 검색 정보 표시

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

## 다음 개선 계획

1. **시각화 모듈 고도화**
   - 인터랙티브 차트 추가 (Plotly 연동)
   - 리얼타임 데이터 모니터링 대시보드
   - A/B 테스트 결과 비교 기능
   - 더 상세한 성능 지표 및 비용 분석

2. **시스템 안정성 확보**
   - Milvus 클라이언트 에러 처리 강화
   - API 변경에 대비한 유연한 인터페이스 유지
   - 로깅 및 모니터링 강화로 문제 조기 발견

2. **성능 테스트 및 최적화**
   - 다양한 청크 크기 및 오버랩 설정 테스트
   - 특수 쿼리("제1보험기간" 등) 검색 성능 지속 모니터링
   - 벤치마크 테스트 자동화

3. **추가 기능**
   - 향상된 메타데이터 필터링
   - 부모 청크 임베딩 선택적 저장 옵션
   - 다양한 문서 타입에 대한 커스텀 청킹 전략
   - 한국어 특화 키워드 추출 및 쿼리 전처리 개선

## 참고 자료

- [LangChain ParentDocumentRetriever](https://python.langchain.com/docs/modules/data_connection/retrievers/parent_document_retriever)
- [Small-to-Big RAG 전략](https://blog.langchain.dev/small-to-big-retrieval-for-rag/)
- [청킹 전략과 오버랩 설정 가이드](https://korn-sudo.github.io/korn-sudo.github.io/posts/chunking-strategies/)
- [Milvus Python SDK 문서](https://milvus.io/docs/install-pymilvus.md)
