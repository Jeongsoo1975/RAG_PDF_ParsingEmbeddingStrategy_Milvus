#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
오프라인 검색을 위한 샘플 데이터 모듈.
검색 테스트를 위한 더미 데이터를 제공합니다.
"""

# 보험 약관 관련 샘플 데이터
SAMPLE_INSURANCE_DATA = [
    {
        "id": "insurance-policy-1",
        "content": "① 이 보험의 보험기간은 다음에 정하는 \"제1보험기간\"과 \"제2보험기간\"으로 구분합니다. (이하 \"제1보험기간\"과 \"제2보험기간\"을 합하여 \"보험기간\"이라 합니다) 1. 제1보험기간은 계약일부터 80세 계약해당일(이하 \"제1보험기간 종료일\"이라 합니다)의 전일까지로 합니다. 2. 제2보험기간은 80세 계약해당일부터 종신까지로 합니다. ② 제1항의 나이는 피보험자의 보험나이를 기준으로 하며, 보험나이는 제9조(보험나이)에 따릅니다.",
        "similarity": 0.95,
        "collection": "insurance_policy",
        "metadata": {
            "source_file": "insurance_terms.pdf", 
            "page_num": 15, 
            "section_title": "제8조(보험기간)"
        }
    },
    {
        "id": "insurance-policy-2",
        "content": "① 주보험의 보험기간은 계약을 체결할 때 정한 기간(이하 '최초계약의 보험기간'이라 합니다)으로 하며, 최초계약의 보험기간은 최대 10년으로 합니다. ② 자동갱신 특약의 갱신계약의 보험기간은 갱신 전 계약의 보험기간과 동일하며, 갱신시 나이에 따른 갱신제한이 있을 수 있습니다.",
        "similarity": 0.89,
        "collection": "insurance_policy",
        "metadata": {
            "source_file": "insurance_terms.pdf", 
            "page_num": 16, 
            "section_title": "제4조(특약의 보험기간 및 보험료 납입)"
        }
    },
    {
        "id": "insurance-policy-3",
        "content": "① 피보험자가 제1보험기간 중 보장개시일 이후에 사망하거나 장해분류표 중 동일한 재해 또는 재해 이외의 동일한 원인으로 여러 신체부위의 장해지급률을 더하여 80% 이상인 장해상태가 된 경우, 재해로 장해분류표 중 여러 신체부위의 장해지급률을 더하여 50% 이상 80% 미만인 장해상태가 되고 계약일부터 10년이 지난 시점부터 제1보험기간 종료일까지 보험료 납입기간 중 피보험자가 보장개시일 이후에 암보장개시일 이후에 암(유방암, 대장암, 전립선암, 기타피부암 및 갑상선암 제외)으로 진단이 확정되었을 때에는, 차회 이후의 보험료 납입을 면제합니다.",
        "similarity": 0.87,
        "collection": "insurance_policy",
        "metadata": {
            "source_file": "insurance_terms.pdf", 
            "page_num": 20, 
            "section_title": "제7조(보험료 납입면제)"
        }
    },
    {
        "id": "insurance-policy-4",
        "content": "① 청약서상 계약 전 알릴 의무(중요한 사항에 한합니다)에 해당하는 질병으로 과거(청약서상 해당 질병의 고지대상 기간을 말합니다)에 진단 또는 치료를 받은 경우에는 제1보험기간 및 제2보험기간의 제10조(보험금의 지급사유) 제2호에서 정한 질병관련 보험금 중 해당 질병과 관련한 보험금을 지급하지 않습니다.",
        "similarity": 0.85,
        "collection": "insurance_policy",
        "metadata": {
            "source_file": "insurance_terms.pdf", 
            "page_num": 25, 
            "section_title": "제15조(보험금을 지급하지 않는 사유)"
        }
    },
    {
        "id": "insurance-policy-5",
        "content": "① 이 계약의 보험료 납입기간은 다음 각 호와 같이 정합니다. 1. 기본보험료의 납입기간: 계약을 체결할 때 계약자가 선택한 제1보험기간 내의 기간으로 합니다. 다만, 보험계약일로부터 최소 5년 이상으로 합니다. 2. 추가납입보험료: 수시납입하며, 제1보험기간 중 기본보험료 납입이 완료된 이후부터 납입할 수 있습니다.",
        "similarity": 0.82,
        "collection": "insurance_policy",
        "metadata": {
            "source_file": "insurance_terms.pdf", 
            "page_num": 18, 
            "section_title": "제14조(보험료의 납입기간)"
        }
    },
    {
        "id": "insurance-faq-1",
        "content": "Q: 제1보험기간과 제2보험기간은 어떻게 다른가요? A: 제1보험기간은 계약일부터 80세 계약해당일의 전일까지이며, 이 기간 동안 기본적인 사망보험금과 각종 특약에 따른 보장을 받습니다. 제2보험기간은 80세 계약해당일부터 종신(평생)까지로, 주로 사망보험금 중심의 보장이 제공됩니다. 두 보험기간에 따라 보장 내용과 보험금액이 달라질 수 있습니다.",
        "similarity": 0.92,
        "collection": "insurance_faq",
        "metadata": {
            "source_file": "insurance_faq.pdf", 
            "page_num": 3, 
            "section_title": "보험기간 관련 FAQ"
        }
    },
    {
        "id": "insurance-faq-2",
        "content": "Q: 보험료를 납입하지 않으면 어떻게 되나요? A: 보험료를 납입하지 않으면 일정 기간(보통 납입기일로부터 30일) 동안은 납입 유예 기간이 주어집니다. 이 기간 내에도 보험료를 납입하지 않으면 계약이 해지될 수 있습니다. 다만, 해지 전 해지환급금에서 월대체보험료를 충당하여 보험계약을 유지하는 '보험료 자동대출납입제도'를 신청한 경우에는 일정 기간 더 유지될 수 있습니다.",
        "similarity": 0.78,
        "collection": "insurance_faq",
        "metadata": {
            "source_file": "insurance_faq.pdf", 
            "page_num": 5, 
            "section_title": "보험료 납입 관련 FAQ"
        }
    },
    {
        "id": "insurance-faq-3",
        "content": "Q: 제1보험기간 중 납입면제 사유가 발생하면 제2보험기간의 보험료도 납입면제 되나요? A: 네, 제1보험기간 중 납입면제 사유(사망, 80% 이상 장해, 50% 이상 80% 미만 장해 후 10년 경과, 암 진단 등)가 발생하여 보험료 납입이 면제된 경우, 제2보험기간까지 모든 보험료 납입이 면제됩니다. 다만, 약관에 명시된 납입면제 사유에 해당하는 경우에만 적용됩니다.",
        "similarity": 0.88,
        "collection": "insurance_faq",
        "metadata": {
            "source_file": "insurance_faq.pdf", 
            "page_num": 7, 
            "section_title": "보험료 납입면제 관련 FAQ"
        }
    },
    {
        "id": "insurance-claim-1",
        "content": "보험금 청구권은 3년간 행사하지 않으면 소멸시효가 완성됩니다. 따라서 보험 사고 발생 후 3년 이내에 청구하셔야 합니다. 다만, 제1보험기간과 제2보험기간에 따라 보험금 청구 사유와 금액이 다를 수 있으니, 약관을 참고하시기 바랍니다.",
        "similarity": 0.75,
        "collection": "insurance_claim",
        "metadata": {
            "source_file": "claim_process.pdf", 
            "page_num": 7, 
            "section_title": "보험금 청구 시 유의사항"
        }
    }
]

# 테스트용 더미 데이터 가져오기
def get_sample_data(query: str = None, top_k: int = 5) -> list:
    """
    쿼리에 관련된 샘플 데이터를 반환합니다.
    
    Args:
        query: 검색 쿼리 문자열
        top_k: 반환할 결과 수
        
    Returns:
        관련 샘플 데이터 목록
    """
    if not query:
        return SAMPLE_INSURANCE_DATA[:top_k]
        
    # 쿼리와 관련성이 있는 결과만 필터링 (간단한 키워드 매칭)
    keywords = query.lower().split()
    filtered_results = []
    
    for result in SAMPLE_INSURANCE_DATA:
        content_lower = result["content"].lower()
        if any(keyword in content_lower for keyword in keywords):
            filtered_results.append(result)
    
    # 결과가 없으면 첫 번째 더미 결과 반환
    if not filtered_results and SAMPLE_INSURANCE_DATA:
        filtered_results = [SAMPLE_INSURANCE_DATA[0]]
        
    # 유사도 기준으로 정렬
    sorted_results = sorted(filtered_results, key=lambda x: x["similarity"], reverse=True)
    
    return sorted_results[:top_k]
