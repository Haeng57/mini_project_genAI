import os
import sys
import json
from typing import Dict, List, Any
from datetime import datetime
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 상위 디렉토리를 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from utils.vector_db import VectorDBManager
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

# 상태 클래스 정의
class ServiceAnalysisState(BaseModel):
    # 입력
    service_name: str = Field(default="", description="분석 대상 AI 서비스의 이름")
    service_description: str = Field(default="", description="서비스에 대한 초기 설명")
    additional_data: Dict[str, Any] = Field(default_factory=dict, description="추가 정보 및 참고 자료")
    
    # 출력
    doc_id: str = Field(default="", description="서비스 개요 문서 고유 식별자")
    chunk_ids: List[str] = Field(default_factory=list, description="분할 청크 ID 목록")
    summary: Dict[str, Any] = Field(default_factory=dict, description="서비스 개요 요약")
    search_results: List[Dict[str, Any]] = Field(default_factory=list, description="검색 결과")
    
    # 제어
    status: str = Field(default="", description="분석 상태 (processing, completed, failed)")
    error_message: str = Field(default="", description="오류 발생 시 메시지")
    timestamp: str = Field(default="", description="분석 수행 시간")

# 서비스 정보 검색 함수
def search_service_info(service_name: str, service_description: str = "") -> Dict[str, Any]:
    """
    Tavily API를 사용하여 서비스에 대한 정보를 웹에서 검색합니다.
    """
    try:
        # Tavily API 키 확인
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            print("⚠️ Tavily API 키가 설정되지 않았습니다. 검색 기능을 사용할 수 없습니다.")
            return {"error": "Tavily API 키 없음"}
        
        # 검색 쿼리 구성
        search_query = f"{service_name} AI service features and technology"
        if service_description:
            search_query += f" {service_description}"
        
        # Tavily 검색 실행
        search = TavilySearchAPIWrapper()
        search_results = search.results(search_query, max_results=5)
        
        # 검색 결과에서 필요한 정보 추출
        extracted_info = {
            "description": "",
            "features": [],
            "target_users": "",
            "tech_stack": "",
            "search_results": search_results
        }
        
        # 검색 결과에서 주요 텍스트 통합
        combined_text = ""
        for result in search_results:
            if "content" in result:
                combined_text += result["content"] + "\n\n"
        
        extracted_info["description"] = combined_text[:500] + "..." if len(combined_text) > 500 else combined_text
        
        return extracted_info
        
    except Exception as e:
        print(f"⚠️ 서비스 정보 검색 중 오류 발생: {str(e)}")
        return {"error": str(e)}

# 에이전트 노드: 서비스 정보 분석
def analyze_service(state: ServiceAnalysisState) -> ServiceAnalysisState:
    """
    AI 서비스 정보를 분석하여 주요 특징, 대상 기능, 사용자 그룹 등을 정리합니다.
    """
    print(f"🔍 서비스 정보 분석 시작: {state.service_name}")
    timestamp = datetime.now().isoformat()
    
    try:
        # OpenAI API 키 가져오기
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OpenAI API 키가 설정되지 않았습니다.")
        
        # 서비스 정보 확인
        if not state.service_name:
            raise ValueError("분석할 서비스 이름이 지정되지 않았습니다.")
        
        # 서비스 정보 가져오기 - Tavily 검색 사용
        print(f"🌐 '{state.service_name}' 정보 검색 중...")
        service_info = search_service_info(state.service_name, state.service_description)
        
        # 검색 오류 확인
        if "error" in service_info:
            print(f"⚠️ 검색 오류: {service_info['error']}")
            # 기본 서비스 설명 사용
            if state.service_description:
                service_info = {"description": state.service_description}
            else:
                service_info = {"description": f"{state.service_name}에 대한 AI 서비스 정보"}
        
        # 추가 정보가 있으면 병합
        if state.additional_data:
            service_info.update(state.additional_data)
        
        # 검색 결과 저장
        search_results = service_info.pop("search_results", []) if "search_results" in service_info else []
        
        # LLM 초기화
        llm = ChatOpenAI(temperature=0, model="gpt-4o-mini", openai_api_key=openai_api_key)
        
        # 프롬프트 템플릿 정의
        template = """
        당신은 AI 서비스 분석 전문가입니다. 제공된 AI 서비스에 대한 정보를 분석하여 다음 형식으로 서비스 개요를 작성해 주세요.
        
        서비스 정보:
        - 이름: {service_name}
        - 설명: {service_description}
        - 검색된 정보: {search_results}
        
        다음 형식으로 정리된 JSON을 작성해주세요:
        ```json
        {{
            "service_name": "서비스 이름",
            "type": "서비스 유형(추천, 생성형, 분류, 예측 등)",
            "description": "250자 내외 서비스 개요",
            "primary_features": ["주요 기능 1", "주요 기능 2", "주요 기능 3"],
            "target_users": ["대상 사용자 그룹 1", "대상 사용자 그룹 2"],
            "data_sources": ["사용 데이터 소스 1", "사용 데이터 소스 2"],
            "technology": ["사용 기술 1", "사용 기술 2"],
            "ethical_concerns": ["잠재적 윤리 이슈 1", "잠재적 윤리 이슈 2", "잠재적 윤리 이슈 3"],
            "analysis_scope": ["진단 범위 항목 1", "진단 범위 항목 2", "진단 범위 항목 3"]
        }}
        ```
        
        특히 ethical_concerns와 analysis_scope는 해당 서비스의 특성을 고려하여 윤리적 진단이 필요한 항목들을 5개 이내로 정확히 작성해주세요.
        잠재적 윤리 이슈에는 편향성, 프라이버시, 투명성, 안전성 등의 관점에서 구체적으로 작성해주세요.
        검색 결과가 제한적이거나 불명확한 경우에는 서비스 이름과 일반적인 AI 서비스 특성을 기반으로 최대한 합리적인 추론을 통해 작성해주세요.
        """
        
        # 검색 결과를 텍스트로 변환
        search_results_text = ""
        for i, result in enumerate(search_results):
            search_results_text += f"\n[{i+1}] 제목: {result.get('title', 'No Title')}\n"
            search_results_text += f"내용: {result.get('content', 'No Content')[:500]}...\n"
            search_results_text += f"URL: {result.get('url', 'No URL')}\n"
        
        if not search_results_text:
            search_results_text = "검색 결과 없음"
        
        # 프롬프트에 필요한 값 준비
        prompt_values = {
            "service_name": state.service_name,
            "service_description": service_info.get("description", ""),
            "search_results": search_results_text
        }
        
        # 프롬프트 생성 및 LLM 호출
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm
        response = chain.invoke(prompt_values)
        
        # JSON 응답 파싱
        response_text = response.content
        json_start = response_text.find('```json')
        json_end = response_text.rfind('```')
        
        if json_start >= 0 and json_end > json_start:
            json_text = response_text[json_start+7:json_end].strip()
            service_summary = json.loads(json_text)
        else:
            service_summary = json.loads(response_text)
        
        # VectorDB에 저장
        db_manager = VectorDBManager()
        collection_name = "service_info"
        
        # 컬렉션 생성 (없는 경우)
        if not db_manager.collection_exists(collection_name):
            db_manager.create_collection(collection_name)
        
        # 서비스 정보 저장
        doc_content = json.dumps(service_summary, ensure_ascii=False, indent=2)
        metadata = {
            "service_name": state.service_name,
            "timestamp": timestamp,
            "content_type": "service_summary"
        }
        
        doc_id = db_manager.add_document(
            collection_name=collection_name,
            content=doc_content,
            metadata=metadata
        )[0]
        
        # 청크 처리 - 실제 상황에서는 필요시 서비스 문서를 청크로 나누어 저장
        chunk_ids = []
        
        # 결과 반환
        print(f"✅ 서비스 분석 완료: {state.service_name}")
        
        return ServiceAnalysisState(
            service_name=state.service_name,
            service_description=state.service_description,
            additional_data=state.additional_data,
            doc_id=doc_id,
            chunk_ids=chunk_ids,
            summary=service_summary,
            search_results=search_results,
            status="completed",
            timestamp=timestamp
        )
        
    except Exception as e:
        error_message = f"서비스 분석 중 오류 발생: {str(e)}"
        print(f"❌ {error_message}")
        
        return ServiceAnalysisState(
            service_name=state.service_name,
            service_description=state.service_description,
            additional_data=state.additional_data,
            status="failed",
            error_message=error_message,
            timestamp=timestamp
        )

# 에이전트 노드: 진단 범위 제안
def suggest_analysis_scope(state: ServiceAnalysisState) -> ServiceAnalysisState:
    """
    서비스 분석 결과를 바탕으로 윤리적 진단 범위를 확정합니다.
    """
    print(f"📋 진단 범위 제안 시작: {state.service_name}")
    
    # 이미 분석이 실패한 경우 처리하지 않음
    if state.status == "failed":
        return state
    
    try:
        # 이미 분석된 요약 정보에서 진단 범위 추출 (ethical_concerns, analysis_scope)
        if "ethical_concerns" in state.summary and "analysis_scope" in state.summary:
            # 이미 진단 범위가 결정되어 있으므로 추가 작업 없이 완료 처리
            print(f"✅ 진단 범위 확정 완료: {state.service_name}")
            return state
        else:
            # 요약 데이터에 진단 범위가 없는 경우 - 실제 구현에서는 LLM을 통해 추가 분석 가능
            print(f"⚠️ 서비스 요약에 진단 범위가 명시되지 않았습니다. 기본 범위를 사용합니다.")
            
            # 기본 진단 범위 설정
            state.summary.update({
                "ethical_concerns": [
                    "데이터 편향성",
                    "프라이버시 침해",
                    "투명성과 설명가능성",
                    "안전성과 신뢰성",
                    "책임성"
                ],
                "analysis_scope": [
                    "편향성 평가 및 완화 방안",
                    "개인정보 수집·이용·보호",
                    "의사결정 과정 투명성",
                    "시스템 안전성 검증",
                    "책임 소재 명확화"
                ]
            })
            
            # 업데이트된 요약 정보 다시 저장
            if state.doc_id:
                db_manager = VectorDBManager()
                doc_content = json.dumps(state.summary, ensure_ascii=False, indent=2)
                metadata = {
                    "service_name": state.service_name,
                    "timestamp": datetime.now().isoformat(),
                    "content_type": "service_summary_updated"
                }
                
                db_manager.update_document(
                    collection_name="service_info",
                    doc_id=state.doc_id,
                    content=doc_content,
                    metadata=metadata
                )
            
            return state
            
    except Exception as e:
        error_message = f"진단 범위 제안 중 오류 발생: {str(e)}"
        print(f"❌ {error_message}")
        
        return ServiceAnalysisState(
            service_name=state.service_name,
            service_description=state.service_description,
            additional_data=state.additional_data,
            doc_id=state.doc_id,
            chunk_ids=state.chunk_ids,
            summary=state.summary,
            search_results=state.search_results,
            status="failed",
            error_message=error_message,
            timestamp=datetime.now().isoformat()
        )

# 워크플로우 제어 함수
def router(state: ServiceAnalysisState) -> str:
    """상태에 따라 다음 단계를 결정합니다."""
    if state.status == "failed":
        return "end"  # 오류 발생 시 종료
    elif not state.summary:  # 아직 분석되지 않은 경우
        return "analyze"
    else:
        return "scope"  # 분석이 완료되어 범위 제안으로 이동

# 그래프 구성
def create_service_analysis_graph() -> StateGraph:
    """서비스 분석 에이전트 그래프 생성"""
    workflow = StateGraph(ServiceAnalysisState)
    
    # 노드 추가
    workflow.add_node("analyze", analyze_service)
    workflow.add_node("scope", suggest_analysis_scope)
    
    # 제어 흐름 설정
    workflow.add_conditional_edges(
        "analyze",  # 시작 노드
        router,     # 라우팅 함수
        {
            "analyze": "analyze",  # 분석 노드로 이동
            "scope": "scope",     # 범위 제안 노드로 이동
            "end": END           # 종료
        }
    )
    
    # 범위 제안 노드에서 종료로 가는 엣지
    workflow.add_edge("scope", END)
    
    # 시작점 설정
    workflow.set_entry_point("analyze")
    
    return workflow

# 에이전트 실행 함수
def run_service_analysis_agent(service_name: str, service_description: str = "", additional_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    서비스 분석 에이전트를 실행합니다.
    """
    print(f"🚀 서비스 분석 에이전트 시작: {service_name}")
    
    # 그래프 생성 및 컴파일
    graph = create_service_analysis_graph()
    app = graph.compile()
    
    # 초기 상태 설정
    initial_state = ServiceAnalysisState(
        service_name=service_name,
        service_description=service_description,
        additional_data=additional_data or {}
    )
    
    # 에이전트 실행
    try:
        result = app.invoke(initial_state)
        
        # 결과는 딕셔너리처럼 접근해야 함 (AddableValuesDict 타입)
        if "status" in result and result["status"] == "completed":
            print(f"✅ 서비스 분석 성공: {service_name}")
            return {
                "service_name": result["service_name"],
                "doc_id": result["doc_id"],
                "chunk_ids": result["chunk_ids"],
                "summary": result["summary"],
                "status": "completed",
                "timestamp": result["timestamp"]
            }
        else:
            error_msg = result.get("error_message", "알 수 없는 오류")
            print(f"❌ 서비스 분석 실패: {error_msg}")
            return {
                "service_name": service_name,
                "status": "failed",
                "error_message": error_msg,
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        error_message = f"에이전트 실행 중 오류 발생: {str(e)}"
        print(f"❌ {error_message}")
        return {
            "service_name": service_name,
            "status": "failed", 
            "error_message": error_message,
            "timestamp": datetime.now().isoformat()
        }

# 테스트 실행
if __name__ == "__main__":
    # Microsoft Azure AI Vision Face API 테스트
    service_name = "Microsoft Azure AI Vision Face API"
    service_description = "얼굴 감지, 식별, 감정 분석 등 얼굴 관련 컴퓨터 비전 기능을 제공하는 클라우드 API 서비스"
    
    # 테스트 시 API 키 확인
    print(f"OpenAI API 키 상태: {'설정됨' if os.getenv('OPENAI_API_KEY') else '설정되지 않음'}")
    print(f"Tavily API 키 상태: {'설정됨' if os.getenv('TAVILY_API_KEY') else '설정되지 않음'}")
    
    # 에이전트 실행 - 이제 직접 서비스 정보 전달
    result = run_service_analysis_agent(
        service_name=service_name,
        service_description=service_description
    )
    
    if result["status"] == "completed":
        print("\n===== 서비스 분석 결과 =====")
        for key, value in result["summary"].items():
            if isinstance(value, list):
                print(f"{key}:")
                for item in value:
                    print(f"  - {item}")
            else:
                print(f"{key}: {value}")
    else:
        print(f"테스트 실패: {result.get('error_message', '알 수 없는 오류')}")