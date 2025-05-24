#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration module for the RAG system.
Loads settings from YAML files and provides a centralized configuration object.
"""

import os
try:
    import yaml
except ImportError:
    print("오류: PyYAML 모듈이 설치되지 않았습니다. 'pip install PyYAML'을 실행하여 설치하세요.")
    # yaml 모듈에 대한 간단한 대체 해결책
    import json
    
    class SimpleYaml:
        @staticmethod
        def safe_load(file):
            return json.load(file)
            
        @staticmethod
        def dump(data, file, default_flow_style=False, allow_unicode=True):
            return json.dump(data, file, ensure_ascii=not allow_unicode, indent=2)
    
    yaml = SimpleYaml()
from pathlib import Path
import logging
from typing import Any, Dict, List, Optional, Union


class Config:
    """Configuration class for RAG system settings."""
    
    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration with default values or from a YAML file or dictionary.
        
        Args:
            config_path: Path to the configuration YAML file. If None, uses default.
            config_dict: Dictionary containing configuration settings. If provided, takes precedence over config_path.
        """
        # Set default config file path if not provided
        if config_path is None and config_dict is None:
            # Try to locate the default config file
            self.base_dir = self._find_project_root()
            default_config = os.path.join(self.base_dir, "configs", "default_config.yaml")
            if os.path.exists(default_config):
                config_path = default_config
                logging.info(f"Using default config file: {default_config}")
            else:
                logging.warning("Default config file not found. Using built-in defaults.")
        else:
            self.base_dir = self._find_project_root()
            if config_path and os.path.exists(config_path):
                # Use provided config path and derive base_dir from it
                self.base_dir = str(Path(config_path).parent.parent)
                logging.info(f"Using specified config file: {config_path}")
            else:
                if config_path and not os.path.exists(config_path):
                    logging.warning(f"Config file not found at {config_path}. Using built-in defaults.")
                    config_path = None
        
        # Load configuration from dictionary if provided
        if config_dict:
            # Transfer config data to class attributes
            for key, value in config_dict.items():
                setattr(self, key, value)
            logging.info("Configuration loaded from dictionary")
        # Otherwise, load from file if available
        elif config_path and os.path.exists(config_path):
            self._load_from_yaml(config_path)
        else:
            self._set_defaults()
        
        # Override base_dir in the loaded config
        if 'paths' in self.__dict__ and isinstance(self.__dict__['paths'], dict):
            self.__dict__['paths']['base_dir'] = self.base_dir
        
        # Setup absolute paths
        self._setup_paths()
    
    def _find_project_root(self) -> str:
        """
        Find and return the project root directory.
        
        Returns:
            The absolute path to the project root directory.
        """
        # Start from the current file and search upward for project markers
        current_dir = Path(__file__).resolve().parent
        
        # Go up to utils' parent (should be src)
        src_dir = current_dir.parent
        
        # Go up to src's parent (should be project root)
        project_root = src_dir.parent
        
        # Verify this is indeed the project root by checking for key directories
        if (project_root / "src").exists() and (project_root / "configs").exists():
            return str(project_root)
        
        # Fallback: use current working directory
        return os.getcwd()
    
    def _load_from_yaml(self, config_path: str) -> None:
        """
        Load configuration from a YAML file.
        
        Args:
            config_path: Path to the YAML configuration file.
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # Transfer config data to class attributes
            for key, value in config_data.items():
                setattr(self, key, value)
            
            logging.info(f"Configuration loaded from {config_path}")
        except Exception as e:
            logging.error(f"Error loading configuration from {config_path}: {e}")
            self._set_defaults()
    
    def load_from_dict(self, config_data: Dict[str, Any]) -> None:
        """
        Load configuration from a dictionary.
        
        Args:
            config_data: Dictionary containing configuration settings.
        """
        try:
            # Transfer config data to class attributes
            for key, value in config_data.items():
                setattr(self, key, value)
            
            # Override base_dir in the loaded config if paths exists
            if 'paths' in self.__dict__ and isinstance(self.__dict__['paths'], dict):
                self.__dict__['paths']['base_dir'] = self.base_dir
            
            # Setup absolute paths
            self._setup_paths()
            
            logging.info(f"Configuration loaded from dictionary with {len(config_data)} sections")
        except Exception as e:
            logging.error(f"Error loading configuration from dictionary: {e}")
            self._set_defaults()
    
    def _set_defaults(self) -> None:
        """Set default configuration values."""
        # General settings
        self.general = {
            'log_level': 'INFO',
            'timezone': 'UTC',
            'language': 'ko'
        }
        
        # Paths
        self.paths = {
            'base_dir': self.base_dir,
            'raw_data_dir': "data/raw",
            'processed_data_dir': "data/processed",
            'embeddings_dir': "data/embeddings",
            'chroma_db_dir': "chroma_db"
        }
        
        # Embedding settings
        self.embedding = {
            'model': "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            'korean_model': "BM-K/KoSimCSE-roberta-multitask",
            'dimension': 768,
            'batch_size': 16,
            'normalize': True
        }
        
        # PDF parsing settings
        self.pdf_parsing = {
            'extract_images': True,
            'image_format': "png",
            'min_image_size': 100,
            'caption_max_distance': 150,
            'caption_keywords': [
                "그림", "Figure", "Fig.", "표", "Table", "도표", "Chart"
            ],
            'respect_sections': True
        }
        
        # CSV parsing settings
        self.csv_parsing = {
            'header': True,
            'encoding': "utf-8",
            'delimiter': ",",
            'quotechar': "\""
        }
        
        # Chunking settings
        self.chunking = {
            'chunk_size': 500,
            'chunk_overlap': 50,
            'split_by': "sentence",
            'sentence_splitter': "kss"
        }
        
        # Retrieval settings
        self.retrieval = {
            'top_k': 5,
            'similarity_threshold': 0.7,
            'similarity_metric': "cosine",
            'hybrid_search': True,
            'reranking': True,
            'collections': []
        }
        
        # Milvus vector database settings
        self.milvus = {
            'host': "localhost",
            'port': 19530,
            'user': "",
            'password': "",
            'secure': False,
            'timeout': 30.0,
            'collection_name': "documents",
            'index_type': "IVF_FLAT",
            'metric_type': "L2",
            'nlist': 128,
            'nprobe': 16
        }
        
        # Logging settings
        self.logging = {
            'level': 'INFO',
            'log_dir': 'logs',
            'log_file': 'rag_system.log',
            'max_file_size': '10MB',
            'backup_count': 5
        }
        
        # Generation settings
        self.generation = {
            'model': "grok-3-mini",
            'temperature': 0.3,
            'max_tokens': 1024,
            'response_language': "ko",
            'prompt_template': (
                "다음 정보를 바탕으로 질문에 답변하세요. 정보에 없는 내용은 '제공된 문서에는 해당 정보가 없습니다.'라고 말하세요.\n\n"
                "### 정보:\n{context}\n\n"
                "### 질문:\n{query}\n\n"
                "### 답변:\n"
            )
        }
        
        # Evaluation settings
        self.evaluation = {
            'metrics': ["precision", "recall", "f1_score", "mrr", "ndcg"],
            'eval_questions_file': "",
            'query_ground_truth_pairs': []
        }
        
        logging.info("Default configuration values set")
    
    def _setup_paths(self) -> None:
        """Setup absolute paths from relative paths in configuration."""
        if 'paths' not in self.__dict__:
            return
        
        # Create a copy of paths to avoid modifying during iteration
        paths_copy = self.paths.copy()
        
        # Create absolute paths for all relative paths
        for key, path in paths_copy.items():
            if key == 'base_dir':
                continue
            
            # Create absolute path
            abs_path = os.path.join(self.base_dir, path)
            
            # Store both relative and absolute paths
            self.paths[key] = path  # Keep relative path
            self.paths[f"{key}_abs"] = abs_path  # Add absolute path
            
            # Ensure directory exists
            if not os.path.exists(abs_path):
                os.makedirs(abs_path, exist_ok=True)
            setattr(self, key, path)
            setattr(self, f"{key}_abs", abs_path)
            
            # Ensure directory exists
            os.makedirs(abs_path, exist_ok=True)
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save current configuration to a YAML file.
        
        Args:
            path: Path to save the configuration. If None, uses default location.
        """
        if path is None:
            path = os.path.join(self.base_dir, "configs", "current_config.yaml")
        
        # Prepare config data to save
        config_data = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_') and key != 'base_dir':
                config_data[key] = value
        
        # Ensure paths don't have absolute versions
        if 'paths' in config_data:
            paths_clean = {}
            for k, v in config_data['paths'].items():
                if not k.endswith('_abs'):
                    paths_clean[k] = v
            config_data['paths'] = paths_clean
        
        # Save to YAML
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
            logging.info(f"Configuration saved to {path}")
        except Exception as e:
            logging.error(f"Error saving configuration to {path}: {e}")
    
    def __getattr__(self, name: str) -> Any:
        """
        Access nested configuration parameters directly.
        
        Args:
            name: Attribute name to access.
            
        Returns:
            The requested attribute value.
            
        Raises:
            AttributeError: If the attribute doesn't exist.
        """
        # 재귀 호출 방지: 내부 딕셔너리에서 직접 확인
        if name == 'embedding' and '_embedding' in self.__dict__:
            return self.__dict__['_embedding']
        if name == 'pdf_parsing' and '_pdf_parsing' in self.__dict__:
            return self.__dict__['_pdf_parsing']
        if name == 'csv_parsing' and '_csv_parsing' in self.__dict__:
            return self.__dict__['_csv_parsing']
        if name == 'chunking' and '_chunking' in self.__dict__:
            return self.__dict__['_chunking']
        if name == 'retrieval' and '_retrieval' in self.__dict__:
            return self.__dict__['_retrieval']
        if name == 'generation' and '_generation' in self.__dict__:
            return self.__dict__['_generation']
        if name == 'evaluation' and '_evaluation' in self.__dict__:
            return self.__dict__['_evaluation']
            
        # 각 섹션 내의 항목 확인
        if 'embedding' in self.__dict__ and isinstance(self.__dict__['embedding'], dict) and name in self.__dict__['embedding']:
            return self.__dict__['embedding'][name]
        if 'pdf_parsing' in self.__dict__ and isinstance(self.__dict__['pdf_parsing'], dict) and name in self.__dict__['pdf_parsing']:
            return self.__dict__['pdf_parsing'][name]
        if 'csv_parsing' in self.__dict__ and isinstance(self.__dict__['csv_parsing'], dict) and name in self.__dict__['csv_parsing']:
            return self.__dict__['csv_parsing'][name]
        if 'chunking' in self.__dict__ and isinstance(self.__dict__['chunking'], dict) and name in self.__dict__['chunking']:
            return self.__dict__['chunking'][name]
        if 'retrieval' in self.__dict__ and isinstance(self.__dict__['retrieval'], dict) and name in self.__dict__['retrieval']:
            return self.__dict__['retrieval'][name]
        if 'generation' in self.__dict__ and isinstance(self.__dict__['generation'], dict) and name in self.__dict__['generation']:
            return self.__dict__['generation'][name]
        if 'evaluation' in self.__dict__ and isinstance(self.__dict__['evaluation'], dict) and name in self.__dict__['evaluation']:
            return self.__dict__['evaluation'][name]
        
        # 속성이 없으면 AttributeError 발생
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def get_general(self, name: str, default: Any = None) -> Any:
        """
        일반적인 설정 값을 가져옵니다.
        
        Args:
            name: 설정 키 이름
            default: 기본값 (키가 없을 경우 반환)
            
        Returns:
            설정 값 또는 기본값
        """
        if 'general' in self.__dict__ and isinstance(self.__dict__['general'], dict):
            return self.__dict__['general'].get(name, default)
        return default
    
    def __str__(self) -> str:
        """Return a string representation of the configuration."""
        return f"RAG Configuration at {self.base_dir}"
    
    def __repr__(self) -> str:
        """Return a detailed string representation of the configuration."""
        sections = []
        for section in ['paths', 'embedding', 'pdf_parsing', 'csv_parsing', 
                        'chunking', 'retrieval', 'generation', 'evaluation']:
            if hasattr(self, section):
                sections.append(f"{section}: {getattr(self, section)}")
        
        return f"RAG Configuration:\n" + "\n".join(sections)


if __name__ == "__main__":
    # Test configuration loading
    logging.basicConfig(level=logging.INFO)
    config = Config()
    print(config)
    print(f"Embedding model: {config.embedding_model}")
    print(f"Raw data directory: {config.raw_data_dir_abs}")
