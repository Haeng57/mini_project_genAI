"""
ChromaDB를 이용한 벡터 데이터베이스 관리 유틸리티
모든 에이전트에서 공통으로 사용할 수 있는 인터페이스 제공
"""

import os
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings.base import Embeddings
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

class VectorDBManager:
    """
    ChromaDB 기반 벡터 데이터베이스 관리자
    에이전트들이 공통으로 사용할 수 있는 문서 저장 및 검색 기능 제공
    """
    
    _instance = None
    
    # 싱글톤 패턴 구현 (선택사항)
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(VectorDBManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self,
        persist_directory: str = "./vector_store",
        openai_api_key: Optional[str] = None,
        embedding_function: Optional[Embeddings] = None,
    ):
        """
        VectorDB 매니저 초기화
        
        Args:
            persist_directory: ChromaDB 데이터 저장 경로
            openai_api_key: OpenAI API 키
            embedding_function: 사용자 지정 임베딩 함수 (선택사항)
        """
        # 이미 초기화된 경우 건너뜀 (싱글톤 패턴)
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self.persist_directory = persist_directory
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.collections = {}  # 컬렉션 캐싱
        
        # 저장 디렉토리 생성
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # 임베딩 함수 초기화 (사용자 지정 또는 기본값)
        self.embedding_function = embedding_function or OpenAIEmbeddings(
            openai_api_key=self.openai_api_key
        )
        
        self._initialized = True
    
    def get_collection(self, collection_name: str) -> Chroma:
        """
        지정된 이름의 컬렉션을 가져오거나 생성
        
        Args:
            collection_name: 컬렉션 이름
            
        Returns:
            Chroma: ChromaDB 컬렉션 객체
        """
        if collection_name not in self.collections:
            self.collections[collection_name] = Chroma(
                collection_name=collection_name,
                embedding_function=self.embedding_function,
                persist_directory=self.persist_directory
            )
        
        return self.collections[collection_name]
    
    def add_document(
        self, 
        collection_name: str, 
        content: Union[str, Dict],
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None
    ) -> List[str]:
        """
        문서를 컬렉션에 추가
        
        Args:
            collection_name: 컬렉션 이름
            content: 문서 내용 (문자열 또는 딕셔너리)
            metadata: 문서 메타데이터 (선택사항)
            doc_id: 문서 ID (선택사항)
            
        Returns:
            List[str]: 생성된 문서 ID 목록
        """
        # 컬렉션 가져오기
        collection = self.get_collection(collection_name)
        
        # 딕셔너리인 경우 JSON으로 변환
        if isinstance(content, dict):
            content = json.dumps(content, ensure_ascii=False)
        
        # 메타데이터가 없으면 기본값 생성
        if metadata is None:
            metadata = {}
        
        # doc_id가 제공되지 않은 경우 타임스탬프를 메타데이터에 추가
        if "timestamp" not in metadata:
            metadata["timestamp"] = datetime.now().isoformat()
        
        # 문서 객체 생성
        documents = [Document(
            page_content=content,
            metadata=metadata
        )]
        
        # 문서 추가 및 ID 반환
        return collection.add_documents(documents)
    
    def search(
        self, 
        collection_name: str, 
        query: str, 
        k: int = 3,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        컬렉션에서 쿼리와 유사한 문서 검색
        
        Args:
            collection_name: 컬렉션 이름
            query: 검색 쿼리
            k: 반환할 최대 문서 수
            filter: 필터링 조건 (선택사항)
            
        Returns:
            List[Document]: 검색된 문서 목록
        """
        collection = self.get_collection(collection_name)
        
        if filter:
            return collection.similarity_search(query, k=k, filter=filter)
        else:
            return collection.similarity_search(query, k=k)
    
    def get_by_metadata(
        self, 
        collection_name: str, 
        metadata_filter: Dict[str, Any],
        limit: int = 10
    ) -> List[Document]:
        """
        메타데이터로 문서 검색
        
        Args:
            collection_name: 컬렉션 이름
            metadata_filter: 메타데이터 필터 조건
            limit: 최대 결과 수
            
        Returns:
            List[Document]: 검색된 문서 목록
        """
        collection = self.get_collection(collection_name)
        return collection.get(where=metadata_filter, limit=limit)
    
    def delete_documents(
        self, 
        collection_name: str, 
        ids: List[str] = None,
        filter: Dict[str, Any] = None
    ) -> None:
        """
        문서 삭제
        
        Args:
            collection_name: 컬렉션 이름
            ids: 삭제할 문서 ID 목록 (선택사항)
            filter: 메타데이터 필터 (선택사항)
        """
        collection = self.get_collection(collection_name)
        
        if ids is not None:
            collection.delete(ids=ids)
        elif filter is not None:
            collection.delete(where=filter)