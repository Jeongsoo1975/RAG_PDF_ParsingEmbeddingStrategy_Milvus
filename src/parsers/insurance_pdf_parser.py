#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PDF Parser for Korean Insurance Documents
특화된 보험 약관 PDF 파서 - 조항 단위 청킹 적용
"""

import os
import re
import json
import uuid
import fitz  # PyMuPDF
import logging
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("insurance_pdf_parser")

@dataclass
class TextChunk:
    """텍스트 청크를 저장하는 클래스"""
    chunk_id: str
    page_num: int
    text: str
    bbox: Optional[Tuple[float, float, float, float]] = None
    section_title: Optional[str] = None
    is_caption: bool = False
    related_image_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Document:
    """문서 정보를 저장하는 클래스"""
    doc_id: str
    source: str
    title: str
    chunks: List[TextChunk]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self):
        """문서를 dict로 변환"""
        return {
            "doc_id": self.doc_id,
            "source": self.source,
            "title": self.title,
            "chunks": [asdict(chunk) for chunk in self.chunks],
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data):
        """dict에서 문서 객체 생성"""
        chunks = [TextChunk(**chunk) for chunk in data["chunks"]]
        return cls(
            doc_id=data["doc_id"],
            source=data["source"],
            title=data["title"],
            chunks=chunks,
            metadata=data["metadata"]
        )

class InsurancePDFParser:
    """보험 약관 PDF 파서 클래스"""
    
    def __init__(self, config=None):
        """
        파서 초기화
        
        Args:
            config: 설정 파라미터 (선택 사항)
        """
        self.config = config if config else {}
        
        # 기본 설정값
        self.chunk_size = self.config.get('chunk_size', 1000)
        self.chunk_overlap = self.config.get('chunk_overlap', 100)
        self.respect_sections = self.config.get('respect_sections', True)
        self.extract_images = self.config.get('extract_images', False)
        self.debug = self.config.get('debug', False)
        
        logger.info(f"초기화: chunk_size={self.chunk_size}, respect_sections={self.respect_sections}")
    
    def process_pdf(self, pdf_path, output_dir=None):
        """
        PDF 파일 처리
        
        Args:
            pdf_path: PDF 파일 경로
            output_dir: 출력 디렉토리 (없으면 같은 위치에 저장)
            
        Returns:
            Document 객체
        """
        logger.info(f"PDF 처리 시작: {pdf_path}")
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF 파일을 찾을 수 없음: {pdf_path}")
        
        # 출력 디렉토리 설정
        if not output_dir:
            output_dir = os.path.dirname(pdf_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # PDF 파일 열기
        doc = fitz.open(pdf_path)
        
        # 기본 메타데이터 추출
        metadata = self._extract_metadata(doc)
        
        # 모든 페이지의 텍스트 블록 추출
        all_blocks = []
        for page_idx, page in enumerate(doc):
            blocks = self._extract_page_blocks(page, page_idx)
            all_blocks.extend(blocks)
        
        logger.info(f"총 {len(all_blocks)}개의 텍스트 블록 추출됨")
        
        # 조항 단위로 청킹
        chunks = self._create_article_based_chunks(all_blocks)
        logger.info(f"총 {len(chunks)}개의 조항 기반 청크 생성됨")
        
        # Document 객체 생성
        document = Document(
            doc_id=str(uuid.uuid4()),
            source=pdf_path,
            title=metadata.get("title", os.path.basename(pdf_path)),
            chunks=chunks,
            metadata=metadata
        )
        
        # 결과 저장
        if output_dir:
            output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(pdf_path))[0]}_chunks.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(document.to_dict(), f, ensure_ascii=False, indent=2)
            logger.info(f"청크 저장 완료: {output_path}")
        
        return document
    
    def _extract_metadata(self, doc):
        """PDF 메타데이터 추출"""
        metadata = {}
        
        # 기본 메타데이터
        for key, value in doc.metadata.items():
            if value:
                metadata[key.lower()] = value
        
        # 페이지 수
        metadata["page_count"] = len(doc)
        
        return metadata
    
    def _extract_page_blocks(self, page, page_idx):
        """페이지에서 텍스트 블록 추출"""
        blocks = []
        
        # dict 형식으로 텍스트 추출 (레이아웃 정보 포함)
        page_dict = page.get_text("dict")
        
        for block_idx, block in enumerate(page_dict["blocks"]):
            if "lines" not in block:
                continue
            
            # 블록의 텍스트 추출
            text = ""
            for line in block["lines"]:
                for span in line["spans"]:
                    text += span["text"] + " "
            
            text = text.strip()
            if not text:
                continue
            
            # 텍스트 블록 생성
            chunk = TextChunk(
                chunk_id=f"block_{page_idx}_{block_idx}_{uuid.uuid4().hex[:8]}",
                page_num=page_idx,
                text=text,
                bbox=(block["bbox"][0], block["bbox"][1], block["bbox"][2], block["bbox"][3])
            )
            blocks.append(chunk)
        
        return blocks
    
    def _is_article_header(self, text):
        """조항 헤더인지 확인 (제XX조)"""
        # 제1조, 제2조 등의 패턴 확인
        article_patterns = [
            r'제\s*\d+\s*조\s*\(',  # 제1조(제목)
            r'제\s*\d+\s*조\s*\（',  # 제1조（제목）
            r'제\s*\d+\s*조',       # 제1조
        ]
        
        for pattern in article_patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def _is_subsection_header(self, text):
        """소항목 헤더인지 확인 (①, ②, 가., 나. 등)"""
        # 항목 패턴 확인
        subsection_patterns = [
            r'^[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮]',  # 원문자 숫자로 시작
            r'^\d+\.\s+',           # 1. 으로 시작
            r'^[가-힣]\.\s+',        # 가. 으로 시작
        ]
        
        for pattern in subsection_patterns:
            if re.search(pattern, text.strip()):
                return True
        
        return False
    
    def _create_article_based_chunks(self, blocks):
        """조항 단위 청킹 알고리즘"""
        if not blocks:
            return []
        
        # 페이지 및 위치로 정렬
        blocks.sort(key=lambda x: (x.page_num, x.bbox[1] if x.bbox else 0))
        
        # 조항 기반 청크 생성
        chunks = []
        current_article = None
        current_article_text = ""
        current_article_blocks = []
        
        for i, block in enumerate(blocks):
            # 조항 헤더인지 확인
            is_article = self._is_article_header(block.text)
            
            # 새로운 조항 시작
            if is_article:
                # 이전 조항 저장
                if current_article and current_article_text:
                    chunks.append(TextChunk(
                        chunk_id=f"article_{current_article_blocks[0].page_num}_{uuid.uuid4().hex[:8]}",
                        page_num=current_article_blocks[0].page_num,
                        text=current_article_text.strip(),
                        bbox=None,
                        section_title=current_article
                    ))
                
                # 새 조항 시작
                current_article = block.text
                current_article_text = block.text + "\n"
                current_article_blocks = [block]
            
            # 현재 조항에 추가
            elif current_article:
                current_article_text += block.text + "\n"
                current_article_blocks.append(block)
            
            # 조항 외 블록 (도입부, 표, 기타 등)
            else:
                # 충분히 긴 텍스트면 별도 청크로 저장
                if len(block.text) > 50:
                    chunks.append(TextChunk(
                        chunk_id=f"nonart_{block.page_num}_{uuid.uuid4().hex[:8]}",
                        page_num=block.page_num,
                        text=block.text,
                        bbox=block.bbox
                    ))
        
        # 마지막 조항 처리
        if current_article and current_article_text:
            chunks.append(TextChunk(
                chunk_id=f"article_{current_article_blocks[0].page_num}_{uuid.uuid4().hex[:8]}",
                page_num=current_article_blocks[0].page_num,
                text=current_article_text.strip(),
                bbox=None,
                section_title=current_article
            ))
        
        # 너무 작은 청크 병합
        merged_chunks = self._merge_small_chunks(chunks)
        
        return merged_chunks
    
    def _merge_small_chunks(self, chunks, min_chars=100):
        """작은 청크 병합"""
        if len(chunks) <= 1:
            return chunks
        
        # 청크 병합
        merged = []
        temp_chunks = []
        temp_text = ""
        
        for chunk in chunks:
            # 조항 청크는 항상 별도로 유지
            if chunk.section_title and self._is_article_header(chunk.section_title):
                # 이전 임시 청크 처리
                if temp_text and len(temp_text) >= min_chars:
                    merged.append(TextChunk(
                        chunk_id=f"merged_{temp_chunks[0].page_num}_{uuid.uuid4().hex[:8]}",
                        page_num=temp_chunks[0].page_num,
                        text=temp_text.strip(),
                        bbox=None
                    ))
                    temp_chunks = []
                    temp_text = ""
                
                # 조항 청크 추가
                merged.append(chunk)
            
            # 작은 청크 병합
            elif len(chunk.text) < min_chars:
                if not temp_chunks or chunk.page_num == temp_chunks[-1].page_num:
                    temp_chunks.append(chunk)
                    temp_text += chunk.text + "\n"
                else:
                    # 페이지가 바뀌면 이전 임시 청크 처리
                    if temp_text:
                        merged.append(TextChunk(
                            chunk_id=f"merged_{temp_chunks[0].page_num}_{uuid.uuid4().hex[:8]}",
                            page_num=temp_chunks[0].page_num,
                            text=temp_text.strip(),
                            bbox=None
                        ))
                    
                    # 새 임시 청크 시작
                    temp_chunks = [chunk]
                    temp_text = chunk.text + "\n"
            
            # 충분히 큰 청크는 그대로 추가
            else:
                # 이전 임시 청크 처리
                if temp_text:
                    merged.append(TextChunk(
                        chunk_id=f"merged_{temp_chunks[0].page_num}_{uuid.uuid4().hex[:8]}",
                        page_num=temp_chunks[0].page_num,
                        text=temp_text.strip(),
                        bbox=None
                    ))
                    temp_chunks = []
                    temp_text = ""
                
                # 일반 청크 추가
                merged.append(chunk)
        
        # 마지막 임시 청크 처리
        if temp_text:
            merged.append(TextChunk(
                chunk_id=f"merged_{temp_chunks[0].page_num}_{uuid.uuid4().hex[:8]}",
                page_num=temp_chunks[0].page_num,
                text=temp_text.strip(),
                bbox=None
            ))
        
        return merged

    def process_directory(self, input_dir, output_dir=None):
        """디렉토리 내 모든 PDF 처리"""
        if not os.path.isdir(input_dir):
            raise NotADirectoryError(f"디렉토리가 아님: {input_dir}")
        
        # 출력 디렉토리 설정
        if not output_dir:
            output_dir = input_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # PDF 파일 목록 가져오기
        pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
        logger.info(f"{len(pdf_files)}개의 PDF 파일 발견")
        
        # 각 PDF 처리
        documents = []
        for pdf_file in pdf_files:
            pdf_path = os.path.join(input_dir, pdf_file)
            try:
                document = self.process_pdf(pdf_path, output_dir)
                documents.append(document)
            except Exception as e:
                logger.error(f"PDF 처리 중 오류 발생: {pdf_path}")
                logger.error(f"오류 내용: {str(e)}")
        
        return documents


# 간단한 사용 예시
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("사용법: python insurance_pdf_parser.py <pdf 파일 경로> [출력 디렉토리]")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    parser = InsurancePDFParser()
    
    if os.path.isdir(pdf_path):
        print(f"디렉토리 처리 중: {pdf_path}")
        documents = parser.process_directory(pdf_path, output_dir)
        print(f"{len(documents)}개의 PDF 문서 처리 완료")
    else:
        print(f"PDF 파일 처리 중: {pdf_path}")
        document = parser.process_pdf(pdf_path, output_dir)
        print(f"처리 완료: {document.title} - {len(document.chunks)}개 청크 생성")
