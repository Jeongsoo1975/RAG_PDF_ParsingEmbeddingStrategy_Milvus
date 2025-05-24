    # 개선된 컨텍스트 검색
    improved_contexts = context_searcher.improved_context_search(question, contexts, top_k=2)
    print(f"개선된 컨텍스트 검색: {len(improved_contexts)}개 선택")
    
    # 개선된 답변 생성
    improved_answer = answer_gen.generate_improved_answer(question, improved_contexts)
    print(f"개선된 답변: {improved_answer}")


if __name__ == "__main__":
    test_improved_generators()
