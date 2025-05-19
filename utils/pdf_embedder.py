from utils.vector_db import VectorDBManager
from utils.pdf_extractor import extract_text_and_tables
from langchain_huggingface import HuggingFaceEmbeddings
import os
from typing import List
import json
from datetime import datetime

def embed_pdf_documents(
    collection_name: str = "ethics_guidelines", 
    specific_files: List[str] = None,
    use_huggingface: bool = True,
    embedding_model: str = "nlpai-lab/KURE-v1",
    chunk_size: int = 500,
    chunk_overlap: int = 50
):
    """
    PDF 문서들을 임베딩하여 Vector DB에 저장
    
    Args:
        collection_name: 저장할 컬렉션 이름
        specific_files: 특정 파일들만 처리하고 싶을 경우 파일명 리스트
        use_huggingface: HuggingFace 임베딩 사용 여부
        embedding_model: 사용할 임베딩 모델명
        chunk_size: 청크 사이즈 (기본값 500자)
        chunk_overlap: 청크 오버랩 (기본값 50자)
    """
    # 임베딩 모델 설정
    if use_huggingface:
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            encode_kwargs={'normalize_embeddings': True}
        )
    else:
        embeddings = None  # 기본 OpenAI 임베딩 사용
    
    # VectorDBManager 인스턴스 생성 (임베딩 모델 지정)
    db_manager = VectorDBManager(embedding_function=embeddings)
    
    # data 디렉토리의 모든 PDF 파일 로드
    data_dir = "./data"
    
    if specific_files:
        pdf_files = specific_files
    else:
        pdf_files = [f for f in os.listdir(data_dir) if f.endswith('.pdf')]
    
    guideline_metadata = {}
    os.makedirs("./outputs/tables", exist_ok=True)
    
    # 텍스트를 청크로 나누는 헬퍼 함수
    def split_text_into_chunks(text, size, overlap):
        if not text:
            return []
        
        # 지정된 크기로 텍스트 분할 (오버랩 포함)
        chunks = []
        for i in range(0, len(text), size - overlap):
            chunk = text[i:i + size]
            if chunk:  # 빈 청크는 추가하지 않음
                chunks.append(chunk)
        
        return chunks
    
    # 각 PDF 처리
    for pdf_file in pdf_files:
        pdf_path = os.path.join(data_dir, pdf_file)
        print(f"처리 중: {pdf_file}")
        
        try:
            # 텍스트와 테이블 모두 추출 (pdf_extracter 모듈 사용)
            pages_text, tables = extract_text_and_tables(pdf_path)
            
            # 테이블 저장
            for i, table in enumerate(tables):
                table_file = f"./outputs/tables/table_{i+1}_page_{table['page']}_{pdf_file.replace('.pdf', '')}.json"
                with open(table_file, "w", encoding="utf-8") as f:
                    json.dump(table, f, ensure_ascii=False, indent=2)
            
            print(f"테이블 추출 완료: {len(tables)}개 테이블 발견")
            
            # 문서 정보 메타데이터 생성
            doc_metadata = {
                "source_type": "pdf",
                "file_name": pdf_file,
                "file_path": pdf_path,
                "type": "guideline",
                "timestamp": datetime.now().isoformat(),
                "total_pages": len(pages_text),
                "tables_count": len(tables)
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
            
            # 문서 요약 정보만 저장 (메타데이터와 함께) - 전체 문서 대신
            summary = f"{pdf_file} 문서 요약: 총 {len(pages_text)}페이지, {len(tables)}개 테이블"
            doc_id = db_manager.add_document(
                collection_name=collection_name,
                content=summary,
                metadata=doc_metadata
            )[0]
            
            # 각 페이지의 내용을 청크로 나누어 VectorDB에 추가
            chunk_ids = []
            for i, page in enumerate(pages_text):
                # 메타데이터 업데이트
                page_metadata = doc_metadata.copy()
                page_num = i + 1
                page_metadata.update({
                    "page_number": page_num,
                    "doc_id": doc_id,
                    "content_type": "page"
                })
                
                # 현재 페이지에 테이블이 있는지 확인
                page_tables = [t for t in tables if int(t["page"]) == page_num]
                if page_tables:
                    page_metadata["has_table"] = True
                    page_metadata["table_count"] = len(page_tables)
                    # 리스트를 쉼표로 구분된 문자열로 변환
                    page_metadata["table_ids"] = ",".join([f"table_{t['table_number']}_page_{t['page']}_{pdf_file.replace('.pdf', '')}" for t in page_tables])
                
                # 페이지 텍스트를 청크로 나누기
                page_text = page["text"]
                if len(page_text) > chunk_size:
                    # 청크로 나누기
                    text_chunks = split_text_into_chunks(page_text, chunk_size, chunk_overlap)
                    print(f"  페이지 {page_num}: 텍스트를 {len(text_chunks)}개 청크로 분할")
                    
                    for j, chunk in enumerate(text_chunks):
                        # 각 청크에 대한 메타데이터 추가
                        chunk_metadata = page_metadata.copy()
                        chunk_metadata.update({
                            "chunk_index": j,
                            "total_chunks": len(text_chunks),
                            "content_type": "chunk"
                        })
                        
                        # 청크를 VectorDB에 추가
                        chunk_id = db_manager.add_document(
                            collection_name=collection_name,
                            content=chunk,
                            metadata=chunk_metadata
                        )[0]
                        chunk_ids.append(chunk_id)
                else:
                    # 짧은 페이지는 그대로 추가
                    page_id = db_manager.add_document(
                        collection_name=collection_name,
                        content=page_text,
                        metadata=page_metadata
                    )[0]
                    chunk_ids.append(page_id)
            
            # 가이드라인 메타데이터 정보 저장
            guideline_metadata[doc_id] = {
                "file_name": pdf_file,
                "chunk_ids": chunk_ids,
                "organization": doc_metadata["organization"],
                "priority": doc_metadata["priority"],
                "tables": [t["page"] for t in tables]
            }
            
            print(f"성공: {pdf_file} ({len(pages_text)} 페이지, {len(chunk_ids)} 청크)")
            
        except Exception as e:
            print(f"오류 발생: {pdf_file} - {str(e)}")
    
    # 가이드라인 메타데이터 정보를 outputs에 저장
    os.makedirs("./outputs", exist_ok=True)
    with open("./outputs/guideline_metadata.json", "w", encoding="utf-8") as f:
        json.dump(guideline_metadata, f, ensure_ascii=False, indent=2)
    
    print("모든 PDF 문서의 임베딩이 완료되었습니다.")
    return guideline_metadata


if __name__ == "__main__":
    # 특정 파일들만 처리하고 싶을 경우
    specific_files = [
        "[UNESCO]AI 윤리에 관한 권고.pdf",
        "[OECD]인공지능 활용 원칙.pdf"
    ]
    
    # HuggingFace KURE-v1 임베딩 모델 사용
    embed_pdf_documents(
        specific_files=specific_files,
        use_huggingface=True,
        embedding_model="nlpai-lab/KURE-v1",
        chunk_size=1000,  # 청크 크기 조정
        chunk_overlap=100  # 오버랩 크기 조정
    )