#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
임베딩 모델 차원 확인 스크립트
"""

from sentence_transformers import SentenceTransformer

# 모델 로드
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
print(f"모델 로드 중: {model_name}")
model = SentenceTransformer(model_name)

# 테스트 텍스트
test_text = "This is a test sentence."
print(f"테스트 텍스트: {test_text}")

# 임베딩 생성
embedding = model.encode([test_text])[0]
print(f"임베딩 형태: {embedding.shape}")
print(f"임베딩 차원: {len(embedding)}")
print(f"임베딩 타입: {type(embedding)}")
print(f"첫 10개 값: {embedding[:10]}")
