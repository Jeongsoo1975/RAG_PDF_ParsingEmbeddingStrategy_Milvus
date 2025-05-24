#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RAG 파이프라인과 RAGAS 연동 테스트 스크립트
"""

import os
import sys
import json
import time

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.abspath('.'))

from src.evaluation.rag_pipeline_adapter import RAGPipelineAdapter, RAGPipelineConfig

def test_insurance_dataset():
    """보험 데이터셋으로 RAG 파이프라인 테스트"""
    print("=== 보험 데이터셋 RAG 파이프라인 테스트 ===")
    
    try:
        # 어댑터 초기화
        adapter = RAGPipelineAdapter()
        
        # 설정
        config = RAGPipelineConfig(
            top_k=3,
            similarity_threshold=0.3,
            use_parent_chunks=True,
            enable_query_optimization=True,
            model="grok-3-mini-beta",
            temperature=0.3,
            delay_between_requests=0.5  # API 호출 간격
        )
        
        # 몇 개의 질문을 수동으로 테스트
        test_questions = [
            {
                "id": "test_001",
                "question": "제1보험기간이란 무엇인가요?",
                "ground_truth": "제1보험기간은 계약일부터 80세 계약해당일 전일까지입니다."
            },
            {
                "id": "test_002", 
                "question": "CI보험금이란 무엇인가요?",
                "ground_truth": "CI보험금은 중대한 질병, 중대한 수술, 중대한 화상 등 약관에서 정한 특정 CI 발생 시 지급되는 보험금입니다."
            },
            {
                "id": "test_003",
                "question": "보험료 납입이 면제되는 경우는?",
                "ground_truth": "합산장해 지급률이 50% 이상 80% 미만인 장해상태가 되거나 CI보험금이 지급된 경우 차회 이후 보험료 납입이 면제됩니다."
            }
        ]
        
        print(f"\n{len(test_questions)}개 질문으로 RAG 파이프라인 테스트 시작...")
        
        results = []
        ragas_items = []
        
        for i, test_q in enumerate(test_questions):
            print(f"\n--- 테스트 {i+1}: {test_q['id']} ---")
            print(f"질문: {test_q['question']}")
            
            # RAG 파이프라인 실행
            result = adapter.execute_rag_pipeline(
                question=test_q['question'],
                ground_truth=test_q['ground_truth'],
                question_id=test_q['id'],
                pipeline_config=config
            )
            
            results.append(result)
            
            # 결과 출력
            print(f"검색된 컨텍스트 수: {len(result.retrieved_contexts)}")
            if result.retrieved_contexts:
                print(f"첫 번째 컨텍스트: {result.retrieved_contexts[0][:100]}...")
            
            print(f"생성된 답변: {result.generated_answer[:150]}...")
            print(f"정답: {result.ground_truth}")
            print(f"처리 시간: {result.processing_time:.2f}초")
            
            if result.error:
                print(f"오류: {result.error}")
            
            # RAGAS 형식으로 변환
            if result.error is None:
                ragas_items.append({
                    "question": result.question,
                    "contexts": result.retrieved_contexts,
                    "ground_truth": result.ground_truth,
                    "answer": result.generated_answer
                })
        
        # RAGAS 형식 데이터셋 생성
        ragas_dataset = {
            "questions": [item["question"] for item in ragas_items],
            "contexts": [item["contexts"] for item in ragas_items],
            "ground_truths": [item["ground_truth"] for item in ragas_items],
            "answers": [item["answer"] for item in ragas_items]
        }
        
        # 결과 저장
        timestamp = int(time.time())
        rag_results_path = f"evaluation_results/rag_pipeline_results/test_rag_results_{timestamp}.json"
        ragas_dataset_path = f"evaluation_results/rag_pipeline_results/test_ragas_dataset_{timestamp}.json"
        
        os.makedirs("evaluation_results/rag_pipeline_results", exist_ok=True)
        
        # RAG 결과 저장
        with open(rag_results_path, 'w', encoding='utf-8') as f:
            results_data = {
                "test_info": {
                    "total_questions": len(results),
                    "successful_results": len([r for r in results if r.error is None]),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                },
                "results": [
                    {
                        "question_id": r.question_id,
                        "question": r.question,
                        "retrieved_contexts": r.retrieved_contexts,
                        "generated_answer": r.generated_answer,
                        "ground_truth": r.ground_truth,
                        "processing_time": r.processing_time,
                        "error": r.error
                    }
                    for r in results
                ]
            }
            json.dump(results_data, f, ensure_ascii=False, indent=2)
        
        # RAGAS 데이터셋 저장
        with open(ragas_dataset_path, 'w', encoding='utf-8') as f:
            json.dump(ragas_dataset, f, ensure_ascii=False, indent=2)
        
        print(f"\n=== 테스트 완료 ===")
        print(f"성공한 질문: {len([r for r in results if r.error is None])}/{len(results)}")
        print(f"RAG 결과 저장: {rag_results_path}")
        print(f"RAGAS 데이터셋 저장: {ragas_dataset_path}")
        print(f"평균 처리 시간: {sum(r.processing_time for r in results) / len(results):.2f}초")
        
        return ragas_dataset_path
        
    except Exception as e:
        print(f"테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    ragas_dataset_path = test_insurance_dataset()
    
    if ragas_dataset_path:
        print(f"\n생성된 RAGAS 데이터셋: {ragas_dataset_path}")
        print("이제 이 데이터셋을 사용하여 RAGAS 평가를 실행할 수 있습니다!")
