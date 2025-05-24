# offline_retrieve 함수 수정 사항

"""
오프라인 검색을 위해 기존의 하드코딩된 더미 데이터 대신 sample_data.py 모듈의 데이터를 사용하도록
retriever.py 파일의 offline_retrieve 함수를 수정합니다.

아래 코드를 retriever.py에서 offline_retrieve 함수에 적용하세요:
"""

def offline_retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    오프라인 모드에서 작동하는 검색 함수
    
    Args:
        query: 검색 쿼리
        top_k: 반환할 결과 수
        
    Returns:
        검색 결과 목록
    """
    self.logger.info(f"오프라인 모드로 검색: '{query}'")
    
    try:
        # 샘플 데이터 모듈에서 데이터 가져오기
        from src.utils.sample_data import get_sample_data
        sample_results = get_sample_data(query, top_k)
        
        if sample_results:
            self.logger.info(f"샘플 데이터에서 {len(sample_results)} 개의 결과를 찾았습니다.")
            return sample_results
    except ImportError:
        self.logger.warning("샘플 데이터 모듈을 로드할 수 없습니다. 기본 더미 결과를 사용합니다.")
    except Exception as e:
        self.logger.error(f"샘플 데이터 처리 중 오류 발생: {e}")
    
    # 기본 더미 결과 (모듈 불러오기 실패 시 대체용)
    dummy_results = [
        {
            "id": "dummy1",
            "content": "① 이 보험의 보험기간은 다음에 정하는 \"제1보험기간\"과 \"제2보험기간\"으로 구분합니다. (이하 \"제1보험기간\"과 \"제2보험기간\"을 합하여 \"보험기간\"이라 합니다) 1. 제1보험기간은 계약일부터 80세 계약해당일(이하 \"제1보험기간 종료일\"이라 합니다)의 전일까지로 합니다. 2. 제2보험기간은 80세 계약해당일부터 종신까지로 합니다.",
            "similarity": 0.95,
            "collection": "offline",
            "metadata": {"source": "insurance_policy.txt", "page_num": 15, "section_title": "제8조(보험기간)"}
        },
        {
            "id": "dummy2",
            "content": "보험료 자동이체 날짜를 변경할 수 있습니다. 보험회사 고객센터나 홈페이지를 통해 납입일 변경 신청을 하시면 됩니다.",
            "similarity": 0.82,
            "collection": "offline",
            "metadata": {"source": "payment_options.txt", "page_num": 3}
        },
        {
            "id": "dummy3",
            "content": "보험금 청구권은 3년간 행사하지 않으면 소멸시효가 완성됩니다. 따라서 보험 사고 발생 후 3년 이내에 청구하셔야 합니다.",
            "similarity": 0.75,
            "collection": "offline",
            "metadata": {"source": "claim_process.txt", "page_num": 7}
        }
    ]
    
    # 쿼리와 관련성이 있는 결과만 필터링 (간단한 키워드 매칭)
    keywords = query.lower().split()
    filtered_results = []
    
    for result in dummy_results:
        content_lower = result["content"].lower()
        if any(keyword in content_lower for keyword in keywords):
            filtered_results.append(result)
    
    # 결과가 없으면 첫 번째 더미 결과 반환
    if not filtered_results and dummy_results:
        filtered_results = [dummy_results[0]]
        
    return filtered_results[:top_k]
