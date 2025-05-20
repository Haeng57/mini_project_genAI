import os
import sys
import json
from typing import Dict, List, Any
from datetime import datetime
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# 상위 디렉토리를 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, END
from utils.vector_db import VectorDBManager
from langchain_openai import ChatOpenAI

# 상태 클래스 정의
class ScopeValidatorState(BaseModel):
    # 입력
    service_info: Dict[str, Any] = Field(default_factory=dict, description="서비스 분석 에이전트가 제공한 서비스 정보")
    
    # 중간 처리 결과
    guideline_references: List[Dict[str, Any]] = Field(default_factory=list, description="진단 범위 검증을 위해 참조한 가이드라인")
    validations: List[Dict[str, Any]] = Field(default_factory=list, description="범위 검증 결과")
    
    # 출력
    validated_scope: Dict[str, Any] = Field(default_factory=dict, description="검증 완료된 진단 범위")
    scope_updates: List[Dict[str, str]] = Field(default_factory=list, description="범위 갱신 사항 목록")
    validation_status: str = Field(default="", description="검증 상태 (completed, failed)")
    timestamp: str = Field(default="", description="검증 수행 시간")
    error_message: str = Field(default="", description="오류 메시지(있는 경우)")

# 에이전트 노드: GuidlineRetriever
def guideline_retriever(state: ScopeValidatorState) -> ScopeValidatorState:
    """
    서비스 정보를 기반으로 관련 윤리 가이드라인을 검색합니다.
    """
    print("🔍 서비스 관련 가이드라인 검색 중...")
    
    if not state.service_info:
        return ScopeValidatorState(
            service_info=state.service_info,
            validation_status="failed",
            error_message="서비스 정보가 제공되지 않았습니다."
        )
    
    # 서비스 키워드 추출
    service_title = state.service_info.get("title", "")
    service_features = state.service_info.get("features", [])
    service_domain = state.service_info.get("domain", "")
    service_summary = state.service_info.get("summary", "")
    
    search_keywords = [
        service_title,
        service_domain,
        *[feature.get("name", "") for feature in service_features],
        "윤리", "프라이버시", "투명성", "편향성", "공정성"
    ]
    
    # VectorDB 검색
    try:
        db_manager = VectorDBManager()
        collection_name = "ethics_guidelines"
        
        guideline_references = []
        for keyword in search_keywords:
            if not keyword.strip():
                continue
                
            results = db_manager.search(
                collection_name=collection_name,
                query=keyword,
                k=3,
                filter={"type": "guideline"}
            )
            
            for result in results:
                guideline_ref = {
                    "content": result.page_content[:500] + "...",
                    "metadata": result.metadata,
                    "relevance_to": keyword
                }
                guideline_references.append(guideline_ref)
        
        # 중복 제거
        unique_refs = []
        unique_ids = set()
        
        for ref in guideline_references:
            ref_id = ref["metadata"].get("doc_id", "")
            if ref_id not in unique_ids:
                unique_ids.add(ref_id)
                unique_refs.append(ref)
        
        print(f"✅ {len(unique_refs)}개의 관련 가이드라인 참조를 찾았습니다.")
        
        return ScopeValidatorState(
            service_info=state.service_info,
            guideline_references=unique_refs,
        )
    
    except Exception as e:
        error_message = f"가이드라인 검색 중 오류 발생: {str(e)}"
        print(f"❌ {error_message}")
        
        return ScopeValidatorState(
            service_info=state.service_info,
            validation_status="failed",
            error_message=error_message
        )

# 에이전트 노드: ScopeValidator
def scope_validator(state: ScopeValidatorState) -> ScopeValidatorState:
    """
    서비스 정보와 가이드라인을 비교하여 진단 범위를 검증합니다.
    """
    print("🔍 진단 범위 검증 중...")
    
    if not state.guideline_references:
        return ScopeValidatorState(
            service_info=state.service_info,
            guideline_references=state.guideline_references,
            validation_status="completed",
            validated_scope=state.service_info,
            timestamp=datetime.now().isoformat(),
            scope_updates=[{"update_type": "no_update", "reason": "가이드라인 참조 없음"}]
        )

    # OpenAI API 키 가져오기
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        return ScopeValidatorState(
            service_info=state.service_info,
            guideline_references=state.guideline_references,
            validation_status="failed",
            error_message="OpenAI API 키가 설정되지 않았습니다.",
            timestamp=datetime.now().isoformat()
        )
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=openai_api_key)
    
    # 서비스 정보와 가이드라인 텍스트 준비
    service_info_text = json.dumps(state.service_info, ensure_ascii=False, indent=2)
    guideline_texts = []
    
    for ref in state.guideline_references:
        org = ref["metadata"].get("organization", "Unknown")
        content = ref["content"]
        relevance = ref["relevance_to"]
        guideline_texts.append(f"출처({org}): {content}\n관련키워드: {relevance}")
    
    guideline_text = "\n\n".join(guideline_texts)
    
    # LLM으로 범위 검증
    validation_prompt = f"""
    당신은 AI 윤리 진단 전문가입니다. 제공된 AI 서비스 정보를 가이드라인과 비교하여 진단 범위를 검증하고 필요시 수정해주세요.
    
    ## AI 서비스 정보
    ```json
    {service_info_text}
    ```
    
    ## 관련 가이드라인
    {guideline_text}
    
    다음을 수행하세요:
    1. 서비스 정보가 윤리 진단에 충분한지 검토
    2. 가이드라인과 관련하여 추가해야 할 진단 범위가 있는지 확인
    3. 서비스 도메인에 특정된 윤리적 고려사항이 있는지 검토
    4. 진단 범위를 JSON 형식으로 반환 (기존 구조 유지, 필요시 필드 추가)
    5. 업데이트 내용 목록을 JSON 배열 형식으로 작성
    
    각 업데이트는 {{"update_type": "added" 또는 "modified" 또는 "removed", "field": "필드명", "reason": "사유"}}
    
    출력 형식:
    {{"validated_scope": [수정된 서비스 정보], "scope_updates": [업데이트 내역 목록]}}
    """
    
    try:
        response = llm.invoke(validation_prompt)
        validation_text = response.content
        
        # JSON 형식 추출 (텍스트에서 JSON 부분만 추출)
        import re
        json_match = re.search(r'\{.*\}', validation_text, re.DOTALL)
        
        if json_match:
            validation_data = json.loads(json_match.group(0))
            validated_scope = validation_data.get("validated_scope", state.service_info)
            scope_updates = validation_data.get("scope_updates", [])
        else:
            # JSON 파싱 실패 시 원본 데이터 유지
            validated_scope = state.service_info
            scope_updates = [{"update_type": "parsing_error", "reason": "검증 결과를 파싱할 수 없습니다."}]
        
        print(f"✅ 진단 범위 검증 완료: {len(scope_updates)}개 업데이트")
        
        return ScopeValidatorState(
            service_info=state.service_info,
            guideline_references=state.guideline_references,
            validated_scope=validated_scope,
            scope_updates=scope_updates,
            validation_status="completed",
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        error_message = f"범위 검증 중 오류 발생: {str(e)}"
        print(f"❌ {error_message}")
        
        # 오류 발생 시 원본 데이터 유지
        return ScopeValidatorState(
            service_info=state.service_info,
            guideline_references=state.guideline_references,
            validated_scope=state.service_info,  # 원본 유지
            validation_status="failed",
            error_message=error_message,
            timestamp=datetime.now().isoformat()
        )

# 그래프 구성
def create_scope_validator() -> StateGraph:
    """범위 검증 에이전트 그래프 생성"""
    workflow = StateGraph(ScopeValidatorState)
    
    # 노드 추가
    workflow.add_node("retrieve", guideline_retriever)
    workflow.add_node("validate", scope_validator)
    
    # 엣지 추가
    workflow.add_edge("retrieve", "validate")
    workflow.add_edge("validate", END)
    
    # 시작점 설정
    workflow.set_entry_point("retrieve")
    
    return workflow

# 에이전트 실행 함수
def run_scope_validator(service_info: Dict[str, Any]) -> Dict[str, Any]:
    """범위 검증 에이전트 실행"""
    print("🚀 범위 검증 에이전트 시작")
    
    # 그래프 생성 및 컴파일
    graph = create_scope_validator()
    app = graph.compile()
    
    # 초기 상태 설정 - model_dump() 사용하여 경고 제거
    initial_state = ScopeValidatorState(service_info=service_info)
    
    # 에이전트 실행
    try:
        result = app.invoke(initial_state.model_dump())  # dict() 대신 model_dump() 사용
        
        print(f"범위 검증 완료: 상태 = {result.get('validation_status', '알 수 없음')}")
        
        # 결과 반환 - 딕셔너리 접근 방식 사용
        return {
            "validated_scope": result.get("validated_scope", {}),
            "scope_updates": result.get("scope_updates", []),
            "validation_status": result.get("validation_status", ""),
            "timestamp": result.get("timestamp", datetime.now().isoformat()),
            "error_message": result.get("error_message") if result.get("error_message") else None
        }
    except Exception as e:
        error_message = f"범위 검증 에이전트 실행 중 오류 발생: {str(e)}"
        print(f"❌ {error_message}")
        return {
            "validation_status": "failed",
            "error_message": error_message,
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    # Microsoft Azure AI Vision Face API 테스트
    test_service_info = {
        "title": "Microsoft Azure AI Vision Face API",
        "domain": "컴퓨터 비전 / 얼굴 인식",
        "summary": "얼굴 감지, 식별, 감정 분석 등 얼굴 관련 컴퓨터 비전 기능을 제공하는 클라우드 API 서비스",
        "features": [
            {"name": "얼굴 감지", "description": "이미지에서 얼굴 위치 및 특징점 감지"},
            {"name": "얼굴 인식", "description": "개인 식별 및 유사도 분석"},
            {"name": "감정 분석", "description": "표정 기반 감정 상태 추정"},
            {"name": "속성 분석", "description": "나이, 성별 등 인구통계학적 속성 추정"}
        ]
    }
    
    # 에이전트 실행
    result = run_scope_validator(test_service_info)
    print(json.dumps(result, ensure_ascii=False, indent=2))