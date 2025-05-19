import camelot
import fitz  # PyMuPDF
import re
from typing import List, Dict, Any, Tuple

def extract_tables_from_pdf(pdf_path: str, page_range: str = 'all') -> List[Dict[str, Any]]:
    """
    PDF에서 테이블 추출하는 함수
    
    Args:
        pdf_path: PDF 파일 경로
        page_range: 처리할 페이지 범위 (기본값: 'all')
        
    Returns:
        List[Dict]: 추출된 테이블 정보 목록
    """
    tables_data = []
    try:
        # lattice 방식으로 테이블 추출 시도
        tables = camelot.read_pdf(pdf_path, pages=page_range, flavor='lattice')
        
        # lattice 방식으로 발견된 테이블이 없으면 stream 방식 시도
        if len(tables) == 0:
            tables = camelot.read_pdf(pdf_path, pages=page_range, flavor='stream')
        
        # 추출된 테이블 처리
        for i, table in enumerate(tables):
            table_data = {
                "page": table.parsing_report['page'],
                "accuracy": table.parsing_report['accuracy'],
                "rows": table.df.shape[0],
                "columns": table.df.shape[1],
                "data": table.df.to_dict('records'),
                "table_number": i + 1
            }
            tables_data.append(table_data)
            
    except Exception as e:
        print(f"테이블 추출 중 오류: {e}")
    
    return tables_data

def clean_text(text: str) -> str:
    """추출된 텍스트를 정리하는 함수"""
    # 연속된 공백과 탭 제거
    text = re.sub(r'\s+', ' ', text)
    # 연속된 줄바꿈을 하나로 통합
    text = re.sub(r'\n+', '\n', text)
    # 줄 끝의 하이픈으로 나뉜 단어 결합 (영어 텍스트용)
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    # 줄바꿈 근처의 공백 정리
    text = re.sub(r' \n ', '\n', text)
    # 페이지 번호 패턴 정리 (예: "- 3 -")
    text = re.sub(r'- \d+ -', '', text)
    return text

def extract_text_and_tables(pdf_path: str) -> Tuple[List[Dict], List[Dict]]:
    """
    PDF에서 텍스트와 테이블을 모두 추출 (개선된 텍스트 추출)
    
    Args:
        pdf_path: PDF 파일 경로
    
    Returns:
        텍스트 페이지 목록과 테이블 목록의 튜플
    """
    # 텍스트 추출
    doc = fitz.open(pdf_path)
    pages_text = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # 방법 1: 블록 형식으로 텍스트 추출 (레이아웃 정보 포함)
        blocks = page.get_text("blocks")
        
        # 블록을 y 좌표에 따라 정렬 (위에서 아래로)
        blocks.sort(key=lambda b: b[1])  # y0 좌표로 정렬
        
        # 블록 텍스트 결합
        page_text = ""
        for block in blocks:
            block_text = block[4]
            if block_text.strip():
                page_text += block_text + "\n"
        
        # 텍스트 정리
        page_text = clean_text(page_text)
        
        # 또는 방법 2: HTML 형식으로 추출 후 처리도 고려할 수 있음
        # html = page.get_text("html")
        # 여기서 HTML 파싱 후 텍스트 추출 로직을 구현할 수 있음
        
        pages_text.append({
            "page": page_num + 1,
            "text": page_text
        })
    
    doc.close()
    
    # 테이블 추출
    tables = extract_tables_from_pdf(pdf_path)
    
    return pages_text, tables