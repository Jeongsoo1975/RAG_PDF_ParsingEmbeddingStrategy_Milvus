#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Re-ranking module for improving search result ordering using Cross-encoder models.
Implements Korean-optimized re-ranking for better document relevance scoring.
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np

# Try to import cross-encoder models
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    logging.warning("sentence-transformers not available, re-ranking will use fallback scoring")

from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger("reranker")

class DocumentReranker:
    """Class for re-ranking search results using Cross-encoder models"""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the document re-ranker with configuration."""
        self.config = config or Config()
        self.logger = logger
        
        # Re-ranking 설정
        self.retrieval_config = self.config.get_section('retrieval') if hasattr(self.config, 'get_section') else self.config.retrieval if hasattr(self.config, 'retrieval') else {}
        
        self.reranking_enabled = self.retrieval_config.get('reranking', True)
        self.preserve_original_order = self.retrieval_config.get('preserve_original_order', False)
        self.reranking_batch_size = self.retrieval_config.get('reranking_batch_size', 32)
        self.reranking_models = self.retrieval_config.get('reranking_models', [
            "jhgan/ko-sroberta-multitask",  # 한국어 특화 모델 (최우선)
            "cross-encoder/ms-marco-MiniLM-L-6-v2",  # 영어 기본 모델
            "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"  # 다국어 모델
        ])
        
        self.cross_encoder = None
        self.current_model_name = None
        self.offline_mode = self.retrieval_config.get('offline_mode', False)
        
        if not self.offline_mode and self.reranking_enabled and CROSS_ENCODER_AVAILABLE:
            self._load_best_available_model()
        else:
            logger.info("Re-ranking running in OFFLINE mode or disabled.")
    
    def _load_best_available_model(self):
        """Load the best available cross-encoder model from the configured list."""
        for model_name in self.reranking_models:
            try:
                logger.info(f"Attempting to load cross-encoder model: {model_name}")
                self.cross_encoder = CrossEncoder(model_name)
                self.current_model_name = model_name
                logger.info(f"Successfully loaded cross-encoder model: {model_name}")
                return
            except Exception as e:
                logger.warning(f"Failed to load cross-encoder model '{model_name}': {e}")
                continue
        
        logger.error("Failed to load any cross-encoder model. Re-ranking will use fallback scoring.")
        self.cross_encoder = None
        self.current_model_name = None
    
    def rerank_results(self, 
                      query: str, 
                      search_results: List[Dict[str, Any]], 
                      top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Re-rank search results using cross-encoder model.
        
        Args:
            query: Search query
            search_results: List of search result dictionaries
            top_k: Number of top results to return (None = return all)
            
        Returns:
            Re-ranked list of search results
        """
        if not self.reranking_enabled:
            logger.info("Re-ranking is disabled. Returning original results.")
            return search_results[:top_k] if top_k else search_results
        
        if not search_results:
            return []
        
        if len(search_results) == 1:
            return search_results  # No need to re-rank single result
        
        logger.info(f"Re-ranking {len(search_results)} results for query: '{query[:50]}...'")
        
        if self.offline_mode or not self.cross_encoder:
            return self._fallback_rerank(query, search_results, top_k)
        
        try:
            return self._cross_encoder_rerank(query, search_results, top_k)
        except Exception as e:
            logger.error(f"Cross-encoder re-ranking failed: {e}. Using fallback.")
            return self._fallback_rerank(query, search_results, top_k)
    
    def _cross_encoder_rerank(self, 
                             query: str, 
                             search_results: List[Dict[str, Any]], 
                             top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Re-rank using cross-encoder model."""
        logger.debug(f"Using cross-encoder model: {self.current_model_name}")
        
        # Prepare query-document pairs for cross-encoder
        query_doc_pairs = []
        for result in search_results:
            content = result.get('content', '')
            if not content:
                # Try to get content from metadata if main content is empty
                metadata = result.get('metadata', {})
                content = metadata.get('text', metadata.get('article_title', ''))
            
            # Truncate content to reasonable length for cross-encoder (usually max 512 tokens)
            # Rough approximation: 1 token ≈ 4 characters for Korean/English mix
            max_content_length = 1800  # ~450 tokens for content + query
            if len(content) > max_content_length:
                content = content[:max_content_length] + "..."
            
            query_doc_pairs.append([query, content])
        
        # Process in batches to manage memory
        reranking_scores = []
        batch_size = self.reranking_batch_size
        
        for i in range(0, len(query_doc_pairs), batch_size):
            batch = query_doc_pairs[i:i + batch_size]
            try:
                batch_scores = self.cross_encoder.predict(batch)
                reranking_scores.extend(batch_scores.tolist() if hasattr(batch_scores, 'tolist') else batch_scores)
            except Exception as e:
                logger.warning(f"Failed to process batch {i//batch_size + 1}: {e}")
                # Fallback: use original similarity scores for failed batch
                for j in range(len(batch)):
                    original_idx = i + j
                    if original_idx < len(search_results):
                        reranking_scores.append(search_results[original_idx].get('similarity', 0.0))
        
        # Add re-ranking scores to results
        reranked_results = []
        for idx, (result, rerank_score) in enumerate(zip(search_results, reranking_scores)):
            result_copy = result.copy()
            result_copy['rerank_score'] = float(rerank_score)
            result_copy['original_rank'] = idx + 1
            result_copy['original_similarity'] = result.get('similarity', 0.0)
            reranked_results.append(result_copy)
        
        # Sort by re-ranking score (higher is better for cross-encoder)
        reranked_results.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        # Apply preserve_original_order logic if enabled
        if self.preserve_original_order:
            reranked_results = self._preserve_partial_order(reranked_results)
        
        final_results = reranked_results[:top_k] if top_k else reranked_results
        
        # Log re-ranking performance
        if len(final_results) > 0:
            original_top_score = search_results[0].get('similarity', 0.0)
            reranked_top_score = final_results[0].get('rerank_score', 0.0)
            rank_changes = sum(1 for r in final_results if r.get('original_rank', 0) != final_results.index(r) + 1)
            
            logger.info(f"Re-ranking completed: {rank_changes}/{len(final_results)} positions changed")
            logger.debug(f"Top result - Original sim: {original_top_score:.4f}, Rerank score: {reranked_top_score:.4f}")
        
        return final_results
    
    def _fallback_rerank(self, 
                        query: str, 
                        search_results: List[Dict[str, Any]], 
                        top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Fallback re-ranking using keyword matching and content features."""
        logger.info("Using fallback re-ranking (keyword-based scoring)")
        
        query_keywords = self._extract_keywords(query.lower())
        
        enhanced_results = []
        for idx, result in enumerate(search_results):
            content = result.get('content', '').lower()
            metadata = result.get('metadata', {})
            
            # Calculate fallback re-ranking score
            fallback_score = self._calculate_fallback_score(
                query_keywords, content, metadata, result.get('similarity', 0.0)
            )
            
            result_copy = result.copy()
            result_copy['rerank_score'] = fallback_score
            result_copy['original_rank'] = idx + 1
            result_copy['original_similarity'] = result.get('similarity', 0.0)
            enhanced_results.append(result_copy)
        
        # Sort by fallback re-ranking score
        enhanced_results.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        return enhanced_results[:top_k] if top_k else enhanced_results
    
    def _calculate_fallback_score(self, 
                                 query_keywords: List[str], 
                                 content: str, 
                                 metadata: Dict[str, Any], 
                                 original_similarity: float) -> float:
        """Calculate fallback re-ranking score based on various features."""
        score = original_similarity * 0.6  # Base score from original similarity
        
        # Keyword matching bonus
        keyword_matches = 0
        for keyword in query_keywords:
            if keyword in content:
                keyword_matches += 1
                # Higher bonus for exact keyword matches
                score += 0.1
        
        # Keyword density bonus
        if query_keywords:
            keyword_density = keyword_matches / len(query_keywords)
            score += keyword_density * 0.2
        
        # Content quality indicators
        content_length = len(content)
        if 100 <= content_length <= 1000:  # Preferred content length
            score += 0.05
        elif content_length > 1000:
            score += 0.03
        
        # Metadata relevance (if available)
        chunk_type = metadata.get('chunk_type', '')
        if chunk_type in ['item', 'article', 'section']:  # Structured content
            score += 0.05
        
        # Article title relevance
        article_title = metadata.get('article_title', '').lower()
        if article_title and any(kw in article_title for kw in query_keywords):
            score += 0.1
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text (simplified version)."""
        import re
        
        # Remove common Korean particles and conjunctions
        stop_words = {
            '이', '가', '은', '는', '을', '를', '의', '에', '에서', '와', '과', 
            '하고', '이란', '라는', '무엇', '뭐', '어떤', '어디', '언제', '왜', '어떻게'
        }
        
        # Extract Korean and English words
        words = re.findall(r'[a-zA-Z가-힣]+', text.lower())
        keywords = [word for word in words if len(word) >= 2 and word not in stop_words]
        
        return list(set(keywords))  # Remove duplicates
    
    def _preserve_partial_order(self, reranked_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Preserve some original ordering to balance between re-ranking improvements and stability.
        This is a conservative approach that limits dramatic rank changes.
        """
        if not reranked_results:
            return reranked_results
        
        # Allow top results to be reordered more freely, but keep lower results more stable
        stable_portion = max(1, len(reranked_results) // 3)  # Bottom 1/3 stays more stable
        
        # Keep top results as re-ranked, but adjust lower results to reduce rank changes
        final_results = reranked_results.copy()
        
        # For the stable portion, limit rank changes to +/- 2 positions from original
        for i in range(stable_portion, len(final_results)):
            result = final_results[i]
            original_rank = result.get('original_rank', i + 1) - 1  # Convert to 0-based
            current_pos = i
            
            # If rank change is too dramatic, move it closer to original position
            max_change = 2
            if abs(current_pos - original_rank) > max_change:
                # Find a better position within allowed range
                target_pos = min(max(current_pos - max_change, 0), 
                               min(current_pos + max_change, len(final_results) - 1))
                
                # Move the result to target position
                result = final_results.pop(i)
                final_results.insert(target_pos, result)
        
        return final_results

# Export main class
__all__ = ['DocumentReranker']
