"""
서비스 분석 에이전트 - AI 서비스의 개요와 주요 기능을 분석하고 진단 범위를 확정
"""

import os
import sys
import json
from typing import Dict, Any, Optional
from datetime import datetime
from uuid import uuid4
from dotenv import load_dotenv

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# .env 파일 로드
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END

# 프롬프트 템플릿 가져오기
from prompts.service_info_prompts import SERVICE_ANALYSIS_TEMPLATE

# VectorDB 매니저 가져오기
from utils.vector_db import VectorDBManager

class ServiceInfoAgent:
    """
    AI 서비스 개요 분석 및 진단 범위 확정을 담당하는 에이전트
    """
    
    def __init__(
        self, 
        model_name: str = "gpt-4o-mini", 
        temperature: float = 0.2,
        openai_api_key: Optional[str] = None,
        tavily_api_key: Optional[str] = None
    ):
        """
        서비스 분석 에이전트 초기화
        
        Args:
            model_name: 사용할 OpenAI 모델명
            temperature: 생성 다양성 파라미터 (0.0~1.0)
            openai_api_key: OpenAI API 키
            tavily_api_key: Tavily 검색 API 키
        """
        self.model_name = model_name
        self.temperature = temperature
        
        # API 키 설정 (.env에서 자동 로드)
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.tavily_api_key = tavily_api_key or os.environ.get("TAVILY_API_KEY")
        
        # LLM 초기화
        self.llm = ChatOpenAI(
            model_name=self.model_name,
            temperature=self.temperature,
            openai_api_key=self.openai_api_key
        )
        
        # 검색 도구 초기화
        self.search = TavilySearchAPIWrapper()
        
        # 텍스트 분할기 초기화
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=300
        )
        
        # VectorDB 매니저 초기화
        self.vector_db = VectorDBManager(openai_api_key=self.openai_api_key)
        
        self.initialize_graph()
    
    def initialize_graph(self):
        """LangGraph 워크플로우 초기화"""
        # 노드 함수들 정의
        def search_service_info(state: Dict[str, Any]) -> Dict[str, Any]:
            """서비스에 관한 정보 검색"""
            service_name = state["service_name"]
            
            # Tavily API로 서비스 정보 검색
            search_results = self.search.results(
                f"{service_name} AI service overview features ethical considerations",
                max_results=5
            )
            
            # 검색 결과 텍스트 추출
            search_texts = []
            for result in search_results:
                if "content" in result:
                    search_texts.append(result["content"])
            
            combined_text = "\n\n".join(search_texts)
            
            # 대용량 텍스트일 경우 청크로 분할
            chunks = self.text_splitter.split_text(combined_text)
            
            return {
                **state,
                "search_results": search_results,
                "search_chunks": chunks
            }
        
        def analyze_service(state: Dict[str, Any]) -> Dict[str, Any]:
            """수집된 정보를 기반으로 서비스 분석"""
            service_name = state["service_name"]
            chunks = state["search_chunks"]
            
            # 청크 내용 결합
            context = "\n\n".join(chunks[:3]) if len(chunks) > 3 else "\n\n".join(chunks)
            
            # 서비스 분석 프롬프트
            service_analysis_prompt = PromptTemplate(
                template=SERVICE_ANALYSIS_TEMPLATE,
                input_variables=["service_name", "context"]
            )
            
            # LangChain을 통한 프롬프트 전송 및 결과 획득
            prompt = service_analysis_prompt.format(service_name=service_name, context=context)
            raw_response = self.llm.invoke(prompt)
            
            # JSON 추출을 위한 정규식 패턴
            import re
            json_pattern = r'```json\s*([\s\S]*?)\s*```'
            json_match = re.search(json_pattern, raw_response.content)
            
            # JSON 추출 실패 시 오류 처리
            if not json_match:
                error_msg = f"서비스 정보를 JSON 형식으로 추출하지 못했습니다: {service_name}"
                print(f"오류: {error_msg}")
                raise ValueError(error_msg)
            
            # JSON 파싱 시도
            json_str = json_match.group(1)
            try:
                analysis_result = json.loads(json_str)
                
                # 필수 필드 검증
                required_fields = ["service_name", "company", "service_summary", "diagnosis_scope"]
                missing_fields = [field for field in required_fields if field not in analysis_result]
                
                if missing_fields:
                    error_msg = f"분석 결과에 필수 필드가 누락되었습니다: {', '.join(missing_fields)}"
                    print(f"오류: {error_msg}")
                    raise ValueError(error_msg)
                    
            except json.JSONDecodeError as e:
                error_msg = f"JSON 파싱 오류: {str(e)}"
                print(f"오류: {error_msg}")
                raise ValueError(error_msg)
            
            # 고유 문서 ID 생성
            doc_id = f"service_info_{uuid4()}"
            
            return {
                **state,
                "analysis_result": analysis_result,
                "doc_id": doc_id
            }
                
        def store_to_db(state: Dict[str, Any]) -> Dict[str, Any]:
            """분석 결과를 VectorDB에 저장"""
            try:
                analysis_result = state["analysis_result"]
                
                # 메타데이터 준비 (get 메서드로 안전하게 접근)
                metadata = {
                    "doc_id": state["doc_id"],
                    "service_name": analysis_result.get("service_name", "알 수 없는 서비스"),
                    "company": analysis_result.get("company", "알 수 없는 회사"),
                    "timestamp": datetime.now().isoformat()
                }
                
                # VectorDB 매니저를 통해 문서 저장
                ids = self.vector_db.add_document(
                    collection_name="service_info",
                    content=analysis_result,
                    metadata=metadata
                )
                
                return {
                    **state,
                    "storage_status": "success",
                    "chunk_ids": ids
                }
                
            except Exception as e:
                print(f"VectorDB 저장 오류: {e}")
                return {
                    **state,
                    "storage_status": "failed",
                    "storage_error": str(e)
                }
            
        def prepare_output_state(state: Dict[str, Any]) -> Dict[str, Any]:
            """최종 State 구조 생성"""
            analysis_result = state["analysis_result"]
            
            # chunk_ids 가져오기 (저장 성공 시)
            chunk_ids = state.get("chunk_ids", [])
            
            # 진단 대상 서비스에 대한 정보 담기
            service_info = {
                "doc_id": state["doc_id"],
                "chunk_ids": chunk_ids,  # 저장된 청크 ID 추가
                "summary": analysis_result.get("service_summary", "요약 정보가 제공되지 않았습니다."),
                "service_name": analysis_result.get("service_name", "알 수 없는 서비스"),
                "company": analysis_result.get("company", "알 수 없는 회사"),
                "main_features": analysis_result.get("main_features", []),
                "diagnosis_scope": analysis_result.get("diagnosis_scope", [])
            }
            
            return {
                "SERVICE_INFO": service_info
            }

        # 워크플로우 그래프 구성
        workflow = StateGraph(input=Dict, output=Dict)
        
        # 노드 추가
        workflow.add_node("search_service_info", search_service_info)
        workflow.add_node("analyze_service", analyze_service)
        workflow.add_node("store_to_db", store_to_db)
        workflow.add_node("prepare_output", prepare_output_state)
        
        # 시작점 추가
        workflow.set_entry_point("search_service_info")
        
        # 엣지 추가
        workflow.add_edge("search_service_info", "analyze_service")
        workflow.add_edge("analyze_service", "store_to_db")
        workflow.add_edge("store_to_db", "prepare_output")
        workflow.add_edge("prepare_output", END)
        
        # 컴파일
        self.graph = workflow.compile()

    def analyze(self, service_name: str) -> Dict[str, Any]:
        """
        AI 서비스 분석 및 진단 범위 확정 실행
        
        Args:
            service_name: 분석할 AI 서비스명
            
        Returns:
            Dict[str, Any]: SERVICE_INFO 상태 객체
        """
        # 초기 상태 설정
        initial_state = {
            "service_name": service_name
        }
        
        # 그래프 실행
        result = self.graph.invoke(initial_state)
        
        return result

# 사용 예시
if __name__ == "__main__":
    # 환경변수에서 API 키를 가져오거나 직접 입력
    agent = ServiceInfoAgent()
    
    # Microsoft Azure AI Vision Face API 분석
    result = agent.analyze("Microsoft Azure AI Vision Face API")
    
    # 결과 출력
    print(json.dumps(result, indent=2, ensure_ascii=False))