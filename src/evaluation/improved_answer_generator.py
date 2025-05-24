#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
개선된 답변 생성기 - Answer Relevancy 향상을 위한 전략적 구현
"""

import re
import logging
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class QuestionAnalysis:
    """질문 분석 결과"""
    question_type: str  # what, how, why, when, where, who
    key_entities: List[str]
    intent: str
    expected_answer_structure: str

class ImprovedAnswerGenerator:
    """개선된 답변 생성기"""
    
    def __init__(self):
        self.logger = logging.getLogger("ImprovedAnswerGenerator")
        
        # 질문 유형별 답변 패턴
        self.answer_patterns = {
            "what": {
                "prefix": "",
                "structure": "direct_definition",
                "focus": "definition_and_characteristics"
            },
            "how": {
                "prefix": "",
                "structure": "process_steps",
                "focus": "methods_and_procedures"
            },
            "why": {
                "prefix": "",
                "structure": "causal_explanation",
                "focus": "reasons_and_causes"
            },
            "who": {
                "prefix": "",
                "structure": "entity_description",
                "focus": "actors_and_roles"
            },
            "when": {
                "prefix": "",
                "structure": "temporal_information",
                "focus": "time_and_sequence"
            },
            "where": {
                "prefix": "",
                "structure": "location_description",
                "focus": "places_and_locations"
            }
        }
        
        # 제거할 불필요한 접두사들
        self.unwanted_prefixes = [
            "Based on the available information,",
            "The process involves",
            "According to the context,",
            "From the provided information,",
            "The available data shows that"
        ]
    
    def analyze_question(self, question: str) -> QuestionAnalysis:
        """질문을 분석하여 유형과 의도 파악"""
        question_lower = question.lower().strip()
        
        # 질문 유형 식별
        question_type = "what"  # 기본값
        if question_lower.startswith("what"):
            question_type = "what"
        elif question_lower.startswith("how"):
            question_type = "how"
        elif question_lower.startswith("why"):
            question_type = "why"
        elif question_lower.startswith("who"):
            question_type = "who"
        elif question_lower.startswith("when"):
            question_type = "when"
        elif question_lower.startswith("where"):
            question_type = "where"
        
        # 핵심 엔티티 추출
        key_entities = self._extract_key_entities(question)
        
        # 의도 파악
        intent = self._determine_intent(question, question_type)
        
        # 예상 답변 구조
        expected_structure = self.answer_patterns[question_type]["structure"]
        
        return QuestionAnalysis(
            question_type=question_type,
            key_entities=key_entities,
            intent=intent,
            expected_answer_structure=expected_structure
        )
    
    def _extract_key_entities(self, question: str) -> List[str]:
        """질문에서 핵심 엔티티 추출"""
        # 중요한 명사/개념 추출
        important_terms = [
            "human rights", "international law", "amnesty international",
            "civil rights", "political rights", "economic rights", 
            "social rights", "cultural rights", "universal declaration",
            "poverty", "education", "protection", "challenges",
            "principles", "obligations", "treaties"
        ]
        
        question_lower = question.lower()
        found_entities = []
        
        for term in important_terms:
            if term in question_lower:
                found_entities.append(term)
        
        return found_entities
    
    def _determine_intent(self, question: str, question_type: str) -> str:
        """질문의 의도 파악"""
        question_lower = question.lower()
        
        if "basic" in question_lower or "fundamental" in question_lower:
            return "foundational_concept"
        elif "role" in question_lower or "function" in question_lower:
            return "functional_explanation"
        elif "work" in question_lower or "operate" in question_lower:
            return "operational_process"
        elif "challenges" in question_lower or "problems" in question_lower:
            return "issue_identification"
        elif "importance" in question_lower or "significance" in question_lower:
            return "value_explanation"
        else:
            return "general_information"
    
    def generate_improved_answer(self, question: str, contexts: List[str]) -> str:
        """개선된 답변 생성"""
        if not contexts:
            return "I cannot provide an answer as no relevant context was found."
        
        # 1. 질문 분석
        analysis = self.analyze_question(question)
        
        # 2. 가장 관련성 높은 컨텍스트 선택
        primary_context = self._select_most_relevant_context(question, contexts, analysis)
        
        # 3. 질문 유형에 맞는 답변 생성
        answer = self._generate_type_specific_answer(question, primary_context, analysis)
        
        # 4. 답변 정제
        refined_answer = self._refine_answer(answer, analysis)
        
        return refined_answer
    
    def _select_most_relevant_context(self, question: str, contexts: List[str], analysis: QuestionAnalysis) -> str:
        """가장 관련성 높은 컨텍스트 선택"""
        if not contexts:
            return ""
        
        # 질문의 핵심 엔티티가 가장 많이 포함된 컨텍스트 선택
        best_context = contexts[0]
        best_score = 0
        
        question_lower = question.lower()
        
        for context in contexts:
            context_lower = context.lower()
            score = 0
            
            # 핵심 엔티티 매칭 점수
            for entity in analysis.key_entities:
                if entity in context_lower:
                    score += 2
            
            # 질문 키워드 매칭 점수
            question_words = question_lower.split()
            for word in question_words:
                if len(word) > 3 and word in context_lower:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_context = context
        
        return best_context
    
    def _generate_type_specific_answer(self, question: str, context: str, analysis: QuestionAnalysis) -> str:
        """질문 유형별 특화된 답변 생성"""
        
        if analysis.question_type == "what":
            return self._generate_what_answer(context, analysis)
        elif analysis.question_type == "how":
            return self._generate_how_answer(context, analysis)
        elif analysis.question_type == "why":
            return self._generate_why_answer(context, analysis)
        else:
            # 기본적인 답변 생성
            return self._generate_default_answer(context)
    
    def _generate_what_answer(self, context: str, analysis: QuestionAnalysis) -> str:
        """What 질문에 대한 답변 생성"""
        sentences = context.split('. ')
        
        # 정의나 설명이 포함된 문장 우선 선택
        definition_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in ['are', 'is', 'include', 'refers to', 'means']):
                definition_sentences.append(sentence.strip())
        
        if definition_sentences:
            # 가장 완전한 정의 선택
            answer = max(definition_sentences, key=len)
            if not answer.endswith('.'):
                answer += '.'
            return answer
        else:
            # 첫 번째 문장 사용
            return sentences[0].strip() + '.'
    
    def _generate_how_answer(self, context: str, analysis: QuestionAnalysis) -> str:
        """How 질문에 대한 답변 생성"""
        sentences = context.split('. ')
        
        # 프로세스나 방법이 설명된 문장들 찾기
        process_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any