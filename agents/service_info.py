from langchain_community.document_loaders import PyMuPDFLoader
from utils.vector_db import VectorDBManager
import os
from typing import List
import json
from datetime import datetime

def embed_pdf_documents(
    collection_name: str = "ethics_guidelines", 
    specific_files: List[str] = None,
    chunk_size: int = 500,
    chunk_overlap: int = 50
):
    """
    PDF 문서들을 임베딩하여 Vector DB에 저장
    
    Args:
        collection_name: 저장할 컬렉션 이름
        specific_files: 특정 파일들만 처리하고 싶을 경우 파일명 리스트
        chunk_size: 청크 사이즈 (기본값 500자)
        chunk_overlap: 청크 오버랩 (기본값 50자)
    """
    # VectorDBManager 인스턴스 생성
    db_manager = VectorDBManager()
    
    # data 디렉토리의 모든 PDF 파일 로드
    data_dir = "./data"
    
    if specific_files:
        pdf_files = specific_files
    else:
        pdf_files = [f for f in os.listdir(data_dir) if f.endswith('.pdf')]
    
    guideline_metadata = {}
    
    # 각 PDF 처리
    for pdf_file in pdf_files:
        pdf_path = os.path.join(data_dir, pdf_file)
        print(f"처리 중: {pdf_file}")
        
        try:
            # PyMuPDFLoader를 사용해 PDF 로드
            loader = PyMuPDFLoader(pdf_path)
            documents = loader.load()
            
            # 문서 정보 메타데이터 생성
            doc_metadata = {
                "source_type": "pdf",
                "file_name": pdf_file,
                "file_path": pdf_path,
                "type": "guideline",
                "timestamp": datetime.now().isoformat(),
                "total_pages": len(documents)
            }
            
            # 문서 특성에 따른 추가 메타데이터 설정
            if "UNESCO" in pdf_file or "유네스코" in pdf_file:
                doc_metadata["priority"] = 1
                doc_metadata["organization"] = "UNESCO"
            elif "OECD" in pdf_file:
                doc_metadata["priority"] = 2
                doc_metadata["organization"] = "OECD"
            else:
                doc_metadata["priority"] = 3
                doc_metadata["organization"] = "기타"
            
            # 전체 문서 내용 저장 (State에서 참조할 수 있도록)
            full_content = "\n\n".join([doc.page_content for doc in documents])
            doc_id = db_manager.add_document(
                collection_name=collection_name,
                content=full_content,
                metadata=doc_metadata
            )[0]  # 첫 번째 ID 반환
            
            # 각 페이지를 VectorDB에 추가 (청크 단위로)
            chunk_ids = []
            for i, doc in enumerate(documents):
                # 메타데이터 업데이트
                page_metadata = doc_metadata.copy()
                page_metadata.update({
                    "page_number": i + 1,
                    "doc_id": doc_id,
                    "content_type": "page"
                })
                
                # 페이지 내용이 테이블인지 확인하는 휴리스틱
                if contain_table(doc.page_content):
                    page_metadata["has_table"] = True
                    
                # 각 페이지를 VectorDB에 추가
                page_id = db_manager.add_document(
                    collection_name=collection_name,
                    content=doc.page_content,
                    metadata=page_metadata
                )[0]
                
                chunk_ids.append(page_id)
            
            # 가이드라인 메타데이터 정보 저장
            guideline_metadata[doc_id] = {
                "file_name": pdf_file,
                "chunk_ids": chunk_ids,
                "organization": doc_metadata["organization"],
                "priority": doc_metadata["priority"],
            }
            
            print(f"성공: {pdf_file} ({len(documents)} 페이지)")
            
        except Exception as e:
            print(f"오류 발생: {pdf_file} - {str(e)}")
    
    # 가이드라인 메타데이터 정보를 outputs에 저장
    os.makedirs("./outputs", exist_ok=True)
    with open("./outputs/guideline_metadata.json", "w", encoding="utf-8") as f:
        json.dump(guideline_metadata, f, ensure_ascii=False, indent=2)
    
    print("모든 PDF 문서의 임베딩이 완료되었습니다.")
    return guideline_metadata

def contain_table(text: str) -> bool:
    """
    텍스트 내용이 테이블을 포함하는지 휴리스틱하게 확인
    
    Args:
        text: 확인할 텍스트 내용
    
    Returns:
        bool: 테이블 포함 여부
    """
    # 테이블 특징 파악을 위한 휴리스틱 기준
    table_indicators = [
        # 행과 열 구분자가 일정 개수 이상 반복되는지
        text.count('|') > 5,
        text.count('\t') > 5,
        # 괄호나 대시 기호 등 테이블의 구분자 패턴
        '-----' in text or '=====' in text,
        # 여러 줄에 걸쳐 구분자가 반복되는 패턴
        text.count('\n') > 3 and (
            sum(line.count('|') > 2 for line in text.split('\n')) > 3
        )
    ]
    
    # 하나라도 True이면 테이블로 간주
    return any(table_indicators)

if __name__ == "__main__":
    # 특정 파일들만 처리하고 싶을 경우
    specific_files = [
        "[UNESCO] AI 윤리에 관한 권고.pdf",
        "인공지능 활용 원칙.pdf"
    ]
    
    embed_pdf_documents(specific_files=specific_files)