"""
Step3 평가기에 개선된 생성기들을 통합하는 방법
"""
from typing import Dict, Any

# Step3StandardRAGASEvaluator 클래스의 _evaluate_single_item 메서드 수정

def _evaluate_single_item_improved(self, item: Dict[str, Any]) -> Dict[str, Any]:
    """개선된 개별 질문 평가"""
    question = item.get('question', '')
    ground_truths = item.get('ground_truths', [])
    original_contexts = item.get('contexts', [])
    
    # 1. 개선된 컨텍스트 검색
    from src.evaluation.improved_answer_generator import ImprovedContextSearcher
    context_searcher = ImprovedContextSearcher()
    
    try:
        retrieved_contexts = self.searcher.search_contexts(question, top_k=5)
        all_contexts = original_contexts + retrieved_contexts
        
        # 개선된 컨텍스트 선별
        improved_contexts = context_searcher.improved_context_search(
            question, all_contexts, top_k=3
        )
        
    except Exception as e:
        self.logger.warning(f"개선된 컨텍스트 검색 실패: {e}")
        improved_contexts = original_contexts
    
    # 2. 개선된 답변 생성
    from src.evaluation.improved_answer_generator import ImprovedAnswerGenerator
    answer_generator = ImprovedAnswerGenerator()
    
    try:
        generated_answer = answer_generator.generate_improved_answer(
            question, improved_contexts
        )
    except Exception as e:
        self.logger.warning(f"개선된 답변 생성 실패: {e}")
        generated_answer = self.answer_generator.generate_answer(question, improved_contexts)
    
    # 3. RAGAS 평가 (기존과 동일)
    # ... 나머지 코드는 동일
    
    return {
        'question': question,
        'generated_answer': generated_answer,
        'contexts': improved_contexts,
        'ground_truths': ground_truths
    }


def main():
    """통합 테스트 실행"""
    print("Integration example - 개선된 평가기 통합 예제")
    print("이 파일은 Step3 평가기에 개선사항을 통합하는 방법을 보여줍니다.")
    
    # 실제 사용시에는 Step3StandardRAGASEvaluator 클래스 내부에 
    # _evaluate_single_item_improved 메서드를 추가하거나 
    # 기존 _evaluate_single_item 메서드를 수정합니다.
    
    print("파일 문법 오류가 수정되었습니다.")

if __name__ == "__main__":
    main()
