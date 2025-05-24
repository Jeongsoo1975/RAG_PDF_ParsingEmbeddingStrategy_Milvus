            # 부모 청크 점수 조정 (자식보다 약간 낮게)
            for parent in parent_chunks:
                parent["similarity"] = threshold * 0.95
        
        # 결과 결합
        combined_results = []
        
        # 1. 먼저 모든 상위 결과의 부모 청크가 포함되었는지 확인
        covered_parent_ids = set()
        
        # Small 결과 먼저 추가
        for result in small_results:
            combined_results.append(result)
            if result.get("parent_chunk_id"):
                covered_parent_ids.add(result["parent_chunk_id"])
        
        # 아직 포함되지 않은 부모 청크 추가
        for parent in parent_chunks:
            if parent["chunk_id"] not in covered_parent_ids:
                combined_results.append(parent)
                if len(combined_results) >= top_k:
                    break
        
        # 최종 정렬 및 제한
        combined_results.sort(key=lambda x: x["similarity"], reverse=True)
        return combined_results[:top_k]
```

## 고급 최적화 및 확장

### 1. 동적 임계값 조정

쿼리 복잡성에 따라 임계값을 자동으로 조정하여 검색 결과를 개선할 수 있습니다:

```python
def calculate_dynamic_threshold(self, query, base_threshold=0.7):
    """쿼리 복잡성에 따라 동적 임계값 계산"""
    # 쿼리 길이 기반 조정
    query_length = len(query.split())
    
    if query_length <= 3:  # 매우 짧은 쿼리
        return base_threshold - 0.1  # 더 낮은 임계값 (더 많은 결과)
    elif query_length >= 10:  # 매우 긴 쿼리
        return base_threshold + 0.05  # 더 높은 임계값 (더 정확한 결과)
    
    return base_threshold
```

### 2. 가중치 기반 다중 검색 전략

단일 쿼리가 아닌 다중 검색 전략을 적용하여 검색 결과의 다양성을 높일 수 있습니다:

```python
def multi_strategy_search(self, query, top_k=5):
    """다중 검색 전략 적용"""
    # 주요 검색 (원본 쿼리)
    primary_results = self.retrieve(query, top_k=top_k)
    
    # 가능한 전략들
    strategies = []
    
    # 1. 키워드 추출 검색
    keywords = self._extract_keywords(query)
    if keywords:
        keyword_query = " ".join(keywords)
        strategies.append({
            "query": keyword_query,
            "weight": 0.8
        })
    
    # 2. 질문 재구성 검색
    reformulated_query = self._reformulate_query(query)
    if reformulated_query != query:
        strategies.append({
            "query": reformulated_query,
            "weight": 0.7
        })
    
    # 결과 결합
    all_results = {r["chunk_id"]: r for r in primary_results}
    
    for strategy in strategies:
        strategy_results = self.retrieve(
            strategy["query"], 
            top_k=top_k // 2
        )
        
        for result in strategy_results:
            chunk_id = result["chunk_id"]
            if chunk_id in all_results:
                # 기존 결과의 점수 강화
                all_results[chunk_id]["similarity"] = max(
                    all_results[chunk_id]["similarity"],
                    result["similarity"] * strategy["weight"]
                )
            else:
                # 새 결과 추가 (가중치 적용)
                result["similarity"] *= strategy["weight"]
                all_results[chunk_id] = result
    
    # 최종 정렬 및 제한
    combined_results = list(all_results.values())
    combined_results.sort(key=lambda x: x["similarity"], reverse=True)
    return combined_results[:top_k]
```

### 3. 청크 타입 가중치 적용

청크 유형에 따라 가중치를 적용하여 특정 유형의 청크를 강조할 수 있습니다:

```python
def apply_chunk_type_weights(self, results):
    """청크 타입에 따른 가중치 적용"""
    # 청크 타입별 가중치 정의
    type_weights = {
        "item": 1.0,        # 기본 가중치
        "item_sub_chunk": 0.95,
        "article": 0.9,
        "paragraph": 0.85,
        "text_block": 0.8,
        "csv_row": 1.0
    }
    
    # 가중치 적용
    for result in results:
        chunk_type = result.get("chunk_type", "")
        if chunk_type in type_weights:
            result["similarity"] *= type_weights[chunk_type]
    
    # 결과 재정렬
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results
```

## 벤치마킹 및 평가

Small-to-Big 청킹 전략의 효과를 측정하기 위한 평가 방법:

### 1. 검색 정확도 측정

```python
def evaluate_retrieval_accuracy(retriever, test_queries, ground_truth, use_parents=True):
    """검색 정확도 평가"""
    metrics = {
        "precision": [],
        "recall": [],
        "f1_score": [],
        "mrr": []  # Mean Reciprocal Rank
    }
    
    for query, relevant_ids in zip(test_queries, ground_truth):
        # 결과 검색
        results = retriever.retrieve(
            query=query,
            top_k=5,
            use_parent_chunks=use_parents
        )
        
        # 검색된 청크 ID
        retrieved_ids = [r["chunk_id"] for r in results]
        
        # 정확도 계산
        relevant_count = len(set(retrieved_ids) & set(relevant_ids))
        precision = relevant_count / len(retrieved_ids) if retrieved_ids else 0
        recall = relevant_count / len(relevant_ids) if relevant_ids else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # MRR 계산
        mrr = 0
        for i, chunk_id in enumerate(retrieved_ids):
            if chunk_id in relevant_ids:
                mrr = 1.0 / (i + 1)
                break
        
        metrics["precision"].append(precision)
        metrics["recall"].append(recall)
        metrics["f1_score"].append(f1)
        metrics["mrr"].append(mrr)
    
    # 평균 메트릭 계산
    avg_metrics = {
        metric: sum(values) / len(values) if values else 0
        for metric, values in metrics.items()
    }
    
    return avg_metrics
```

### 2. 처리 시간 및 리소스 사용량 비교

```python
def benchmark_performance(retriever, test_queries, with_parents=True):
    """처리 시간 및 리소스 사용량 측정"""
    import time
    import psutil
    import tracemalloc
    
    metrics = {
        "avg_retrieval_time": 0,
        "max_memory_usage": 0,
    }
    
    # 메모리 추적 시작
    tracemalloc.start()
    
    # 시간 측정
    start_time = time.time()
    
    for query in test_queries:
        retriever.retrieve(
            query=query,
            top_k=5,
            use_parent_chunks=with_parents
        )
    
    # 총 실행 시간
    total_time = time.time() - start_time
    metrics["avg_retrieval_time"] = total_time / len(test_queries)
    
    # 메모리 사용량
    current, peak = tracemalloc.get_traced_memory()
    metrics["max_memory_usage"] = peak / (1024 * 1024)  # MB
    
    tracemalloc.stop()
    
    return metrics
```

## 결론 및 모범 사례

### 1. 최적의 청크 크기 선택

텍스트 유형과 사용 사례에 따라 적절한 청크 크기를 선택하는 것이 중요합니다:

- **Small 청크**: 150-300 토큰 (정확한 정보 검색에 적합)
- **Big 청크**: 1000-2000 토큰 (넓은 맥락 제공에 적합)
- **오버랩**: Small 청크는 25-50 토큰, Big 청크는 100-200 토큰

### 2. 적절한 청크 유형 선택

문서 구조에 따라 적절한 청크 유형을 선택하세요:

- **구조화된 문서 (PDF, Word)**: 계층적 청킹 (조항 → 항목)
- **반구조화된 문서 (HTML, 위키)**: 섹션 및 단락 기반 청킹
- **비구조화된 문서 (TXT)**: 의미 단위 청킹

### 3. 벡터 DB 고려사항

- **스키마 설계**: 계층 관계를 명확히 표현하는 스키마 설계
- **인덱싱 전략**: 검색 성능을 고려한 인덱스 설정
- **메타데이터 설계**: 필터링에 필요한 메타데이터 필드 포함

### 4. 벡터 DB 선택 기준

| 벡터 DB | 장점 | 단점 | 적합한 사용 사례 |
|---------|------|------|-----------------|
| **Milvus** | 확장성, 고성능, 유연한 스키마 | 관리 복잡성, 리소스 사용량 | 대규모 운영 환경, 고성능 필요 시 |
| **Pinecone** | 관리 용이성, 확장성, 빠른 구축 | 상대적으로 높은 비용 | 빠른 개발 주기, 관리 리소스 제한적일 때 |
| **ChromaDB** | 경량, 쉬운 설치, 로컬 실행 | 대규모 확장성 제한 | 개발/테스트, 소규모 응용 프로그램 |
| **FAISS** | 매우 높은 성능, 세밀한 제어 | 관리 기능 제한, 개발 노력 | 연구, 고성능 필요 시, 맞춤형 솔루션 |

### 5. 성능 최적화 팁

- **배치 처리**: 문서 처리, 임베딩, 저장 작업을 배치로 처리
- **캐싱**: 자주 사용되는 쿼리 결과 캐싱
- **비동기 처리**: 대규모 인덱싱 작업에 비동기 처리 적용
- **임베딩 모델 선택**: 적절한 성능과 품질의 임베딩 모델 선택

## 구현 체크리스트

Small-to-Big 청킹 전략을 새 프로젝트에 적용할 때 다음 체크리스트를 참고하세요:

- [ ] 문서 분석 및 청크 구조 설계
- [ ] 계층적 청킹 로직 구현 (parent-child 관계 유지)
- [ ] 벡터 DB 스키마 설계 및 구성
- [ ] 임베딩 및 저장 로직 구현
- [ ] Small-to-Big 검색 로직 구현
- [ ] 검색 결과 후처리 및 최적화
- [ ] 성능 및 정확도 평가
- [ ] 실제 쿼리로 테스트 및 조정

이 가이드를 통해 Small-to-Big 청킹 전략을 다양한 벡터 데이터베이스 환경에 성공적으로 구현할 수 있습니다. 현재 프로젝트의 parser.py 코드를 최대한 활용하여 효율적으로 전략을 적용할 수 있으며, 이를 통해 RAG 시스템의 검색 성능과 응답 품질을 크게 향상시킬 수 있습니다.
