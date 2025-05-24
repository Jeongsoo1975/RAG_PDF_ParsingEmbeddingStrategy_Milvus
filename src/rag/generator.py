#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Response Generator module for generating responses using language models.
Supports various LLM APIs and configurations.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Union
import re
import time
import requests

# Import for OpenAI API
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("openai not available, OpenAI API will not work")

# Import for Anthropic API
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logging.warning("anthropic not available, Anthropic API will not work")

# Check for Grok API key
GROK_API_KEY = os.environ.get('GROK_API_KEY')
if not GROK_API_KEY:
    logging.warning("Grok API key not available, Grok API will not work")
else:
    logging.info(f"Grok API key found, Grok API is available: {GROK_API_KEY[:4]}...{GROK_API_KEY[-4:]}")

from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger("generator")


class ResponseGenerator:
    """Class to generate responses using language models"""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the response generator with configuration.
        
        Args:
            config: Configuration object. If None, a default configuration is used.
        """
        self.config = config or Config()
        self.logger = logger
        
        # Set generation related parameters
        self.model = self.config.generation.get('model', 'gpt-3.5-turbo')
        self.temperature = self.config.generation.get('temperature', 0.3)
        self.max_tokens = self.config.generation.get('max_tokens', 1024)
        self.response_language = self.config.generation.get('response_language', 'ko')
        self.prompt_template = self.config.generation.get('prompt_template', 
                                                       self._get_default_prompt_template())
        
        # Initialize APIs
        self._init_openai_api()
        self._init_anthropic_api()
        
        # Set LLM handler based on model name
        self.llm_handler = self._get_llm_handler()
    
    def _get_prompt_template_for_language(self, language: str) -> str:
        """
        Get prompt template for specific language.
        
        Args:
            language: Language code ('ko', 'en', etc.)
            
        Returns:
            Prompt template for the specified language.
        """
        if language == 'ko':
            return (
                "다음 정보를 바탕으로 질문에 답변하세요. 정보에 없는 내용은 '제공된 문서에는 해당 정보가 없습니다.'라고 말하세요.\n\n"
                "### 정보:\n{context}\n\n"
                "### 질문:\n{query}\n\n"
                "### 답변:\n"
            )
        else:  # English or other languages
            return (
                "Based on the following information, please answer the question. If the information is not in the provided documents, say 'The information is not available in the provided documents.'\n\n"
                "### Information:\n{context}\n\n"
                "### Question:\n{query}\n\n"
                "### Answer:\n"
            )
    
    def _get_default_prompt_template(self) -> str:
        """
        Get default prompt template based on response language.
        
        Returns:
            Default prompt template.
        """
        if self.response_language == 'ko':
            return (
                "다음 정보를 바탕으로 질문에 답변하세요. 정보에 없는 내용은 '제공된 문서에는 해당 정보가 없습니다.'라고 말하세요.\n\n"
                "### 정보:\n{context}\n\n"
                "### 질문:\n{query}\n\n"
                "### 답변:\n"
            )
        else:  # English or other languages
            return (
                "Based on the following information, please answer the question. If the information is not in the provided documents, say 'The information is not available in the provided documents.'\n\n"
                "### Information:\n{context}\n\n"
                "### Question:\n{query}\n\n"
                "### Answer:\n"
            )
    
    def _init_openai_api(self):
        """Initialize OpenAI API if available."""
        if not OPENAI_AVAILABLE:
            self.logger.warning("OpenAI API not available, skipping initialization")
            return
        
        # Check for API key
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            self.logger.warning("OpenAI API key not found in environment variables")
            return
        
        # Initialize client
        try:
            self.openai_client = openai.OpenAI(api_key=api_key)
            self.logger.info("OpenAI API initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing OpenAI API: {e}")
            self.openai_client = None
    
    def _init_anthropic_api(self):
        """Initialize Anthropic API if available."""
        if not ANTHROPIC_AVAILABLE:
            self.logger.warning("Anthropic API not available, skipping initialization")
            return
        
        # Check for API key
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            self.logger.warning("Anthropic API key not found in environment variables")
            return
        
        # Initialize client
        try:
            self.anthropic_client = anthropic.Anthropic(api_key=api_key)
            self.logger.info("Anthropic API initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing Anthropic API: {e}")
            self.anthropic_client = None
    
    def _get_llm_handler(self):
        """
        Get the appropriate LLM handler based on model name.
        
        Returns:
            Function to handle LLM API calls.
        """
        # 모델명 로깅
        self.logger.info(f"Setting up LLM handler for model: {self.model}")
        
        # API 키 상태 로깅
        self.logger.info(f"API Keys status: GROK_API_KEY={'Available' if GROK_API_KEY else 'Not available'}")
        
        # Grok 모델 먼저 확인 (모델 이름에 grok이 들어가 있다면 최우선적으로 Grok API 사용)
        if "grok" in self.model.lower():
            if not GROK_API_KEY:
                self.logger.error("Grok API key not available, cannot use Grok model")
                return self._dummy_llm_handler
            self.logger.info("Using Grok API handler")
            return self._grok_handler
            
        # OpenAI 모델 확인
        elif "gpt" in self.model.lower() or "text-davinci" in self.model.lower():
            if not OPENAI_AVAILABLE or not hasattr(self, 'openai_client') or self.openai_client is None:
                self.logger.error("OpenAI API not available or not initialized")
                # Grok API 키가 있으면 대체로 사용
                if GROK_API_KEY:
                    self.logger.info("Falling back to Grok API handler")
                    return self._grok_handler
                return self._dummy_llm_handler
            return self._openai_chat_handler
        
        # Claude 모델 확인
        elif "claude" in self.model.lower():
            if not ANTHROPIC_AVAILABLE or not hasattr(self, 'anthropic_client') or self.anthropic_client is None:
                self.logger.error("Anthropic API not available or not initialized")
                # Grok API 키가 있으면 대체로 사용
                if GROK_API_KEY:
                    self.logger.info("Falling back to Grok API handler")
                    return self._grok_handler
                return self._dummy_llm_handler
            return self._anthropic_handler
        
        # 알 수 없는 모델이지만 Grok API 키가 있으면 사용
        elif GROK_API_KEY:
            self.logger.warning(f"Unknown model: {self.model}, using available Grok API")
            return self._grok_handler
            
        # 알 수 없는 모델이고 API 키도 없음
        else:
            self.logger.warning(f"Unknown model: {self.model}, using dummy handler")
            return self._dummy_llm_handler
    
    def _dummy_llm_handler(self, prompt: str) -> str:
        """
        Dummy LLM handler for testing or when API is not available.
        
        Args:
            prompt: Prompt for the LLM.
            
        Returns:
            Dummy response.
        """
        self.logger.warning("Using dummy LLM handler, no actual API call will be made")
        return f"This is a dummy response. No LLM API was called. Prompt: {prompt[:100]}..."
    
    def _openai_chat_handler(self, prompt: str) -> str:
        """
        Handler for OpenAI Chat API.
        
        Args:
            prompt: Prompt for the LLM.
            
        Returns:
            Generated response.
        """
        self.logger.info(f"Generating response using OpenAI model: {self.model}")
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": f"You are a helpful assistant that answers questions based on the provided information. Respond in {self.response_language}."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Extract response text
            reply = response.choices[0].message.content.strip()
            
            self.logger.info(f"Generated response of length {len(reply)}")
            return reply
        
        except Exception as e:
            self.logger.error(f"Error generating response with OpenAI: {e}")
            return f"죄송합니다. 응답 생성 중에 오류가 발생했습니다: {str(e)}" if self.response_language == 'ko' else f"Sorry, an error occurred while generating the response: {str(e)}"
    
    def _anthropic_handler(self, prompt: str) -> str:
        """
        Handler for Anthropic Claude API.
        
        Args:
            prompt: Prompt for the LLM.
            
        Returns:
            Generated response.
        """
        self.logger.info(f"Generating response using Anthropic model: {self.model}")
        
        try:
            response = self.anthropic_client.messages.create(
                model=self.model,
                system=f"You are a helpful assistant that answers questions based on the provided information. Respond in {self.response_language}.",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Extract response text
            reply = response.content[0].text
            
            self.logger.info(f"Generated response of length {len(reply)}")
            return reply
        
        except Exception as e:
            self.logger.error(f"Error generating response with Anthropic: {e}")
            return f"죄송합니다. 응답 생성 중에 오류가 발생했습니다: {str(e)}" if self.response_language == 'ko' else f"Sorry, an error occurred while generating the response: {str(e)}"
    
    def _grok_handler(self, prompt: str) -> str:
        """
        Handler for Grok API.
        
        Args:
            prompt: Prompt for the LLM.
            
        Returns:
            Generated response.
        """
        # 모델명을 그대로 사용 (grok-3-mini 유지)
        actual_model = self.model
        self.logger.info(f"Generating response using Grok model: {actual_model}")
        
        # Check if API key is available
        if not GROK_API_KEY:
            self.logger.error("GROK_API_KEY not found in environment variables")
            return "Error: GROK_API_KEY not found. Please set the GROK_API_KEY environment variable."
        
        try:
            # Grok API endpoint
            api_url = "https://api.x.ai/v1/chat/completions"
            
            # Prepare headers with API key
            headers = {
                "Authorization": f"Bearer {GROK_API_KEY}",
                "Content-Type": "application/json"
            }
            
            # Prepare request payload
            payload = {
                "model": actual_model,
                "messages": [
                    {"role": "system", "content": f"You are a helpful assistant that answers questions based on the provided information. Respond in {self.response_language}."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            
            self.logger.info(f"Making request to Grok API with API key: {GROK_API_KEY[:4]}...{GROK_API_KEY[-4:]}")
            
            # Make API request
            response = requests.post(api_url, headers=headers, json=payload)
            
            # Check response status
            if response.status_code == 200:
                result = response.json()
                self.logger.info(f"Grok API response: {result}")
                if "choices" in result and len(result["choices"]) > 0:
                    reply = result["choices"][0]["message"]["content"].strip()
                    self.logger.info(f"Generated response of length {len(reply)}")
                    return reply
                else:
                    self.logger.error(f"Unexpected response format from Grok API: {result}")
                    return "Error: Unexpected response format from Grok API."
            else:
                error_message = f"Error from Grok API: {response.status_code}, {response.text}"
                self.logger.error(error_message)
                return f"죄송합니다. 응답 생성 중에 오류가 발생했습니다: {error_message}" if self.response_language == 'ko' else f"Sorry, an error occurred while generating the response: {error_message}"
        
        except Exception as e:
            self.logger.error(f"Error generating response with Grok: {e}")
            return f"죄송합니다. 응답 생성 중에 오류가 발생했습니다: {str(e)}" if self.response_language == 'ko' else f"Sorry, an error occurred while generating the response: {str(e)}"
    
    def generate(self, 
                query: str, 
                retrieved_docs: List[Dict[str, Any]],
                prompt_template: Optional[str] = None,
                override_language: Optional[str] = None) -> str:
        """
        Generate a response based on a query and retrieved documents.
        
        Args:
            query: User query.
            retrieved_docs: List of retrieved documents.
            prompt_template: Custom prompt template. If None, uses the default template.
            override_language: Override language for this generation only. If None, uses default response_language.
            
        Returns:
            Generated response.
        """
        # Store original language setting
        original_language = self.response_language
        
        try:
            # Apply language override if provided
            if override_language is not None:
                self.logger.info(f"Overriding response language from '{original_language}' to '{override_language}'")
                self.response_language = override_language
            
            # Format context with potentially overridden language
            context = self._format_context(retrieved_docs)
            
            # Use custom template or get appropriate template for current language
            if prompt_template:
                template = prompt_template
            elif override_language is not None:
                template = self._get_prompt_template_for_language(override_language)
            else:
                template = self.prompt_template
            
            # Fill in template
            prompt = template.format(context=context, query=query)
            
            # 핸들러 확인
            if self.llm_handler == self._dummy_llm_handler:
                self.logger.warning("Using dummy LLM handler - trying to get a real handler")
                self.llm_handler = self._get_llm_handler()
                
            # Generate response
            self.logger.info(f"Generating response with handler type: {self.llm_handler.__name__}")
            response = self.llm_handler(prompt)
            
            # Add citations if needed
            if retrieved_docs and self._should_add_citations(response):
                response = self._add_citations(response, retrieved_docs)
            
            return response
            
        finally:
            # Always restore original language setting
            self.response_language = original_language
    
    def _format_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into a context string.
        
        Args:
            retrieved_docs: List of retrieved documents.
            
        Returns:
            Formatted context string.
        """
        if not retrieved_docs:
            if self.response_language == 'ko':
                return "제공된 문서가 없습니다."
            else:
                return "No documents were provided."
        
        # Format each document with metadata
        contexts = []
        
        # 유사도 기준으로 문서 정렬
        sorted_docs = sorted(retrieved_docs, key=lambda x: x.get("similarity", 0), reverse=True)
        
        for i, doc in enumerate(sorted_docs):
            content = doc["content"]
            metadata = doc.get("metadata", {})
            similarity = doc.get("similarity", 0)
            
            # Extract useful metadata
            doc_source = metadata.get("source_file", "")
            page_num = metadata.get("page_num", "")
            section_title = metadata.get("section_title", "")
            chunk_id = metadata.get("chunk_id", "")
            
            # Format context entry
            header = f"[문서 {i+1}]" if self.response_language == 'ko' else f"[Document {i+1}]"
            
            if doc_source:
                file_name = os.path.basename(doc_source)
                header += f" {file_name}"
            
            if page_num:
                header += f" - 페이지 {page_num}" if self.response_language == 'ko' else f" - Page {page_num}"
            
            if section_title:
                header += f" - 섹션: {section_title}" if self.response_language == 'ko' else f" - Section: {section_title}"
            
            # 유사도 점수 추가
            header += f" (유사도: {similarity:.4f})" if self.response_language == 'ko' else f" (Similarity: {similarity:.4f})"
            
            # Combine header and content
            entry = f"{header}\n{content}\n"
            contexts.append(entry)
        
        # Join all context entries
        return "\n".join(contexts)
    
    def _should_add_citations(self, response: str) -> bool:
        """
        Determine if citations should be added to the response.
        
        Args:
            response: Generated response.
            
        Returns:
            True if citations should be added, False otherwise.
        """
        # Don't add citations if response indicates no information was found
        no_info_phrases = [
            "제공된 문서에는 해당 정보가 없습니다",
            "The information is not available in the provided documents",
            "information is not provided",
            "not mentioned in the",
            "not found in the",
            "no information about"
        ]
        
        for phrase in no_info_phrases:
            if phrase.lower() in response.lower():
                return False
        
        return True
    
    def _add_citations(self, 
                      response: str, 
                      retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Add citations to the response.
        
        Args:
            response: Generated response.
            retrieved_docs: List of retrieved documents.
            
        Returns:
            Response with added citations.
        """
        # Add a citation section
        if self.response_language == 'ko':
            citation_header = "\n\n### 출처:\n"
        else:
            citation_header = "\n\n### Sources:\n"
        
        citations = []
        for i, doc in enumerate(retrieved_docs):
            metadata = doc.get("metadata", {})
            doc_source = metadata.get("source_file", "")
            page_num = metadata.get("page_num", "")
            
            if doc_source:
                file_name = os.path.basename(doc_source)
                citation = f"[{i+1}] {file_name}"
                
                if page_num:
                    citation += f" (Page {page_num})"
                
                citations.append(citation)
            else:
                # Use collection name if source file is not available
                collection = doc.get("collection", "")
                if collection:
                    citation = f"[{i+1}] {collection}"
                    citations.append(citation)
        
        if citations:
            return response + citation_header + "\n".join(citations)
        
        return response


# Entry point for direct execution
if __name__ == "__main__":
    config = Config()
    generator = ResponseGenerator(config)
    
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        query = sys.argv[1]
        
        # Create dummy retrieved documents for testing
        dummy_docs = [
            {
                "content": "This is a sample document content for testing.",
                "metadata": {
                    "source_file": "sample.pdf",
                    "page_num": "1"
                }
            }
        ]
        
        print(f"Generating response for query: {query}")
        response = generator.generate(query, dummy_docs)
        print("\nGenerated Response:")
        print(response)
    else:
        print("Usage: python generator.py <query>")
