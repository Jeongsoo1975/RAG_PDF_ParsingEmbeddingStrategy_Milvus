#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Amnesty QA 데이터 생성 스크립트 (간소화된 버전)
- 외부 라이브러리 없이 간단한 amnesty 스타일 데이터 생성
- 기존 Document/TextChunk 구조로 변환
"""

import os
import json
import uuid
import time
from pathlib import Path

# 기본 amnesty 샘플 데이터
AMNESTY_SAMPLE_DATA = [
    {
        "question": "What are the basic principles of human rights?",
        "context": "Human rights are fundamental freedoms and protections that belong to every person. The basic principles include universality, indivisibility, interdependence, equality and non-discrimination. These rights are inherent to all human beings regardless of nationality, sex, national or ethnic origin, color, religion, language, or any other status.",
        "answer": "The basic principles of human rights include universality, indivisibility, interdependence, equality and non-discrimination.",
        "title": "Human Rights Principles"
    },
    {
        "question": "What is the role of international law in protecting human rights?",
        "context": "International human rights law establishes the obligations of governments to act in certain ways or to refrain from certain acts, in order to promote and protect human rights and fundamental freedoms of individuals or groups. International human rights law is primarily made up of treaties and customary international law.",
        "answer": "International law establishes government obligations to promote and protect human rights through treaties and customary international law.",
        "title": "International Human Rights Law"
    },
    {
        "question": "How does Amnesty International work to protect human rights?",
        "context": "Amnesty International is a global movement that campaigns for a world where human rights are enjoyed by all. We investigate and expose abuses, educate and mobilize the public, and help transform societies to create a safer, more just world. We received the Nobel Peace Prize for our life-saving work.",
        "answer": "Amnesty International investigates abuses, educates the public, and mobilizes communities to transform societies and protect human rights.",
        "title": "Amnesty International Mission"
    },
    {
        "question": "What are civil and political rights?",
        "context": "Civil and political rights are a class of rights that protect individuals' freedom from infringement by governments, social organizations, and private individuals. They ensure one's entitlement to participate in the civil and political life of society and the state without discrimination or repression. These include rights to life, liberty, freedom of expression, and fair trial.",
        "answer": "Civil and political rights protect individual freedom from government infringement and ensure participation in civil and political life, including rights to life, liberty, expression, and fair trial.",
        "title": "Civil and Political Rights"
    },
    {
        "question": "What are economic, social and cultural rights?",
        "context": "Economic, social and cultural rights include the rights to adequate food, housing, education, health, social security, and to take part in cultural life. These rights are enshrined in the International Covenant on Economic, Social and Cultural Rights. They are essential for human dignity and development.",
        "answer": "Economic, social and cultural rights include rights to adequate food, housing, education, health, social security, and cultural participation, essential for human dignity.",
        "title": "Economic Social Cultural Rights"
    },
    {
        "question": "How can individuals contribute to human rights protection?",
        "context": "Every individual can contribute to human rights protection by staying informed about human rights issues, supporting organizations that defend human rights, participating in peaceful demonstrations, writing to government officials, and speaking out against discrimination and injustice in their communities. Education and awareness are powerful tools for change.",
        "answer": "Individuals can protect human rights by staying informed, supporting rights organizations, participating in peaceful demonstrations, contacting officials, and speaking out against injustice.",
        "title": "Individual Human Rights Action"
    },
    {
        "question": "What is the Universal Declaration of Human Rights?",
        "context": "The Universal Declaration of Human Rights is a milestone document in the history of human rights. Drafted by representatives with different legal and cultural backgrounds from all regions of the world, it was proclaimed by the United Nations General Assembly in Paris on 10 December 1948 as a common standard of achievements for all peoples and all nations.",
        "answer": "The Universal Declaration of Human Rights is a UN document from 1948 that establishes a common standard of human rights achievements for all peoples and nations.",
        "title": "Universal Declaration"
    },
    {
        "question": "What are the main challenges facing human rights today?",
        "context": "Today's human rights challenges include increasing authoritarianism, restrictions on civil society, attacks on human rights defenders, discrimination against marginalized groups, and the impact of technology on privacy and freedom of expression. Climate change also poses new threats to human rights, particularly affecting vulnerable populations.",
        "answer": "Main human rights challenges today include authoritarianism, civil society restrictions, attacks on defenders, discrimination, technology impacts on privacy, and climate change threats.",
        "title": "Contemporary Human Rights Challenges"
    },
    {
        "question": "How does poverty relate to human rights?",
        "context": "Poverty is both a cause and consequence of human rights violations. It denies people access to basic necessities like food, shelter, education, and healthcare. The human rights approach to poverty reduction emphasizes empowerment, participation, and accountability, focusing on the most marginalized and excluded groups.",
        "answer": "Poverty both causes and results from human rights violations, denying access to basic necessities. Human rights approaches emphasize empowerment and focus on marginalized groups.",
        "title": "Poverty and Human Rights"
    },
    {
        "question": "What is the importance of human rights education?",
        "context": "Human rights education builds awareness, skills, and knowledge to promote respect for human rights. It empowers people to know and claim their rights and to respect the rights of others. Education helps create a culture of human rights where people understand their role in creating a just society.",
        "answer": "Human rights education builds awareness and skills, empowers people to claim their rights and respect others' rights, creating a culture of human rights and justice.",
        "title": "Human Rights Education"
    }
]

def create_text_chunks(text, doc_id, doc_title, chunk_size=300):
    """텍스트를 청크로 분할"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i + chunk_size]
        chunk_text = ' '.join(chunk_words)
        
        chunk_id = str(uuid.uuid4())
        chunk = {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "text": chunk_text,
            "page_num": (i // chunk_size) + 1,
            "chunk_type": "context",
            "metadata": {
                "source_file": "amnesty_qa_generated",
                "document_title": doc_title,
                "chunk_index": i // chunk_size,
                "word_count": len(chunk_words)
            }
        }
        chunks.append(chunk)
    
    return chunks

def create_amnesty_documents():
    """Amnesty 스타일 문서들 생성"""
    documents = []
    
    for i, sample in enumerate(AMNESTY_SAMPLE_DATA):
        doc_id = str(uuid.uuid4())
        
        # 텍스트 청킹
        chunks = create_text_chunks(sample["context"], doc_id, sample["title"])
        
        # Document 객체 생성
        document = {
            "doc_id": doc_id,
            "source": f"amnesty_qa_sample_{i}",
            "title": sample["title"],
            "chunks": chunks,
            "parent_chunks": None,
            "metadata": {
                "dataset": "amnesty_qa_generated",
                "question": sample["question"],
                "answer": sample["answer"],
                "context_length": len(sample["context"].split()),
                "sample_index": i
            }
        }
        
        documents.append(document)
    
    return documents

def create_evaluation_dataset():
    """평가용 데이터셋 생성"""
    evaluation_data = {
        "dataset_info": {
            "name": "amnesty_qa_evaluation",
            "total_questions": len(AMNESTY_SAMPLE_DATA),
            "creation_date": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "questions": [],
        "contexts": [],
        "ground_truths": [],
        "ids": [],
        "metadata": []
    }
    
    for i, sample in enumerate(AMNESTY_SAMPLE_DATA):
        evaluation_data["questions"].append(sample["question"])
        evaluation_data["contexts"].append([sample["context"]])
        evaluation_data["ground_truths"].append([sample["answer"]])
        evaluation_data["ids"].append(f"amnesty_qa_{i}")
        evaluation_data["metadata"].append({
            "title": sample["title"],
            "source": "amnesty_qa_generated",
            "index": i
        })
    
    return evaluation_data

def main():
    """메인 실행 함수"""
    # 출력 디렉토리 생성
    output_dir = Path("data/amnesty_qa")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Amnesty QA 데이터 생성 시작...")
    
    # 1. 문서 생성
    documents = create_amnesty_documents()
    
    # 2. 문서 데이터 저장
    documents_data = {
        "dataset_info": {
            "name": "amnesty_qa_generated",
            "version": "1.0",
            "total_documents": len(documents),
            "conversion_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": "generated_samples"
        },
        "documents": documents
    }
    
    documents_file = output_dir / "amnesty_qa_documents.json"
    with open(documents_file, 'w', encoding='utf-8') as f:
        json.dump(documents_data, f, ensure_ascii=False, indent=2)
    
    # 3. 평가 데이터셋 생성 및 저장
    evaluation_data = create_evaluation_dataset()
    
    evaluation_file = output_dir / "amnesty_qa_evaluation.json"
    with open(evaluation_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_data, f, ensure_ascii=False, indent=2)
    
    # 4. 결과 출력
    total_chunks = sum(len(doc["chunks"]) for doc in documents)
    
    print("\n=== Amnesty QA 데이터 생성 완료 ===")
    print(f"문서 수: {len(documents)}")
    print(f"질문 수: {len(AMNESTY_SAMPLE_DATA)}")
    print(f"청크 수: {total_chunks}")
    print(f"문서 파일: {documents_file}")
    print(f"평가 파일: {evaluation_file}")
    
    return str(documents_file), str(evaluation_file)

if __name__ == "__main__":
    main()
