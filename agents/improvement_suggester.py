# filepath: /Users/lwh/SKALA/mini_project_genAI/agents/improvement_suggester.py
import os
import json
from typing import Dict, List
from datetime import datetime
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from utils.vector_db import VectorDBManager

# 상태 클래스 정의
class ImprovementSuggesterState(BaseModel):
    # 입력
    service_info: Dict = Field(default_factory=dict, description="서비스 정보")
    risk_assessment: Dict = Field(default_factory=dict, description="리스크 평가 결과")
    
    # 출력
    improvement_suggestion: Dict = Field(default_factory=dict, description="개선 권고안")
    error_message: str = Field(default="", description="오류 메시지(있는 경우)")
    timestamp: str = Field(default="", description="실행 시간")

# 에이전트 노드 함수들
def retrieve_best_practices(state: ImprovementSuggesterState) -> ImprovementSuggesterState:
    """
    관련 리스크에 대한 최선의 개선 방안 사례를 검색
    """
    print("🔍 모범 사례 검색 중...")
    
    risk_assessment = state.risk_assessment
    if not risk_assessment or "risk_items" not in risk_assessment:
        return ImprovementSuggesterState(
            service_info=state.service_info,
            risk_assessment=risk_assessment,
            error_message="리스크 평가 결과가 누락되었습니다",
            timestamp=datetime.now().isoformat()
        )
    
    try:
        # 심각한 리스크 항목들 추출
        severity_levels = risk_assessment.get("severity_levels", [])
        high_risk_items = [item for item in severity_levels 
                          if item.get("level") in ["높음", "심각"]]
        
        # 중간 리스크 항목들 추출
        medium_risk_items = [item for item in severity_levels 
                            if item.get("level") == "중간"]
        
        # 최대 3개의 심각한 리스크와 2개의 중간 리스크 선택
        selected_high = high_risk_items[:3]
        selected_medium = medium_risk_items[:2]
        
        selected_items = selected_high + selected_medium
        
        # 벡터 DB에서 관련 모범 사례 검색
        best_practices = {}
        db_manager = VectorDBManager()
        
        for item in selected_items:
            category = item.get("category", "")
            risk_item = item.get("risk_item", "")
            query = f"{category} {risk_item} best practices solutions"
            
            docs = db_manager.search(
                collection_name="ethics_guidelines",
                query=query,
                k=3
            )
            
            item_id = item.get("item_id", "unknown")
            best_practices[item_id] = {
                "item": item,
                "practices": [{
                    "source": doc.metadata.get("file_name", "알 수 없음"),
                    "content": doc.page_content
                } for doc in docs]
            }
        
        return ImprovementSuggesterState(
            service_info=state.service_info,
            risk_assessment=risk_assessment,
            improvement_suggestion={"best_practices": best_practices},
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        return ImprovementSuggesterState(
            service_info=state.service_info,
            risk_assessment=risk_assessment,
            error_message=f"모범 사례 검색 중 오류: {str(e)}",
            timestamp=datetime.now().isoformat()
        )

def generate_improvement_plan(state: ImprovementSuggesterState) -> ImprovementSuggesterState:
    """
    검색된 모범 사례를 기반으로 개선 계획 생성
    """
    print("📝 개선 계획 생성 중...")
    
    if state.error_message:
        return state
    
    try:
        best_practices = state.improvement_suggestion.get("best_practices", {})
        risk_assessment = state.risk_assessment
        service_info = state.service_info
        
        if not best_practices:
            # 모범사례가 없더라도 개선 계획 생성
            pass
        
        # LLM을 사용하여 개선 계획 생성
        llm = ChatOpenAI(model="gpt-4o-mini")
        
        # 프롬프트 템플릿
        template = """
        당신은 AI 윤리 및 리스크 완화 전문가입니다.
        다음 AI 서비스와 식별된 윤리적 리스크에 대한 구체적인 개선 방안을 제안해주세요.

        # 서비스 정보
        {service_info}

        # 리스크 평가 결과
        {risk_assessment}

        # 참고 모범 사례
        {best_practices}

        다음 기준에 따라 각 리스크 항목별 개선 방안을 제시하세요:
        1. 높음/심각 등급의 리스크부터 우선 처리
        2. 각 개선안의 기대 효과 및 구현 난이도 표시
        3. 단기(즉시 적용), 중기(3개월 내), 장기(6개월 이상) 로드맵 분류
        4. 국제 윤리 가이드라인 준수 여부 확인

        응답은 다음 JSON 형식으로 작성하세요:
        ```json
        {{
          "prioritized_improvements": [
            {{
              "risk_id": "리스크 ID",
              "category": "리스크 카테고리",
              "level": "리스크 등급",
              "risk_item": "리스크 항목명",
              "current_issue": "현재 문제점",
              "improvement_plan": "개선 방안 상세 설명",
              "expected_effects": "기대 효과",
              "implementation_difficulty": "구현 난이도(상/중/하)",
              "timeline": "단기/중기/장기",
              "guideline_compliance": ["준수하는 가이드라인 목록"]
            }},
            ...
          ],
          "general_recommendations": "전체적인 윤리성 강화를 위한 일반 권고사항",
          "monitoring_plan": "지속적인 모니터링 방안"
        }}
        ```
        """
        
        # LLM 호출
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm
        
        service_info_str = json.dumps(service_info, ensure_ascii=False, indent=2)
        risk_assessment_str = json.dumps(risk_assessment, ensure_ascii=False, indent=2)
        best_practices_str = json.dumps(best_practices, ensure_ascii=False, indent=2)
        
        response = chain.invoke({
            "service_info": service_info_str,
            "risk_assessment": risk_assessment_str,
            "best_practices": best_practices_str
        })
        
        # JSON 응답 파싱
        content = response.content
        json_start = content.find("```json") + 7 if "```json" in content else content.find("{")
        json_end = content.find("```", json_start) if "```" in content[json_start:] else len(content)
        json_str = content[json_start:json_end].strip()
        
        improvement_plan = json.loads(json_str)
        
        # 결과 저장
        doc_id = f"improvement_plan_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # ChromaDB에 저장
        db_manager = VectorDBManager()
        saved_id = db_manager.add_document(
            collection_name="improvement_plans",
            content=json.dumps(improvement_plan, ensure_ascii=False),
            metadata={
                "type": "improvement_plan",
                "service_name": service_info.get("service_name", "unknown"),
                "timestamp": datetime.now().isoformat()
            },
            doc_id=doc_id
        )[0]
        
        return ImprovementSuggesterState(
            service_info=service_info,
            risk_assessment=risk_assessment,
            improvement_suggestion={
                "doc_id": saved_id,
                "best_practices": best_practices,
                "improvement_plan": improvement_plan,
                "suggestions": improvement_plan.get("prioritized_improvements", [])
            },
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        current_improvement = state.improvement_suggestion.copy()
        current_improvement["error"] = str(e)
        
        return ImprovementSuggesterState(
            service_info=state.service_info,
            risk_assessment=state.risk_assessment,
            improvement_suggestion=current_improvement,
            error_message=f"개선 계획 생성 중 오류: {str(e)}",
            timestamp=datetime.now().isoformat()
        )

# 그래프 구성
def create_improvement_suggester_agent() -> StateGraph:
    """개선안 제안 에이전트 그래프 생성"""
    workflow = StateGraph(ImprovementSuggesterState)
    
    # 노드 추가
    workflow.add_node("retrieve", retrieve_best_practices)
    workflow.add_node("generate", generate_improvement_plan)
    
    # 엣지 추가
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    # 시작점 설정
    workflow.set_entry_point("retrieve")
    
    return workflow

# 에이전트 실행 함수
def run_improvement_suggester(service_info: Dict, risk_assessment: Dict) -> Dict:
    """개선안 제안 에이전트 실행"""
    print("🚀 개선안 제안 에이전트 시작")
    
    # 그래프 생성 및 컴파일
    graph = create_improvement_suggester_agent()
    app = graph.compile()
    
    # 에이전트 실행
    initial_state = ImprovementSuggesterState(
        service_info=service_info,
        risk_assessment=risk_assessment
    )
    
    result = app.invoke(initial_state)
    
    # 결과 출력
    if result.error_message:
        print(f"❌ 개선안 제안 실패: {result.error_message}")
    else:
        suggestions_count = len(result.improvement_suggestion.get("suggestions", []))
        print(f"✅ 개선안 제안 완료: {suggestions_count}개 개선안 제안됨")
    
    # 개선 제안 결과 반환
    return result.improvement_suggestion

if __name__ == "__main__":
    # 테스트용 데이터
    test_service_info = {
        "service_name": "AI 영상 분석 서비스",
        "company": "테스트회사",
        "service_category": "영상분석",
        "features": ["얼굴 인식", "행동 분석", "감정 인식"],
        "summary": "이 서비스는 CCTV 영상에서 얼굴을 인식하고 행동과 감정을 분석하는 AI 기반 서비스입니다."
    }
    
    test_risk_assessment = {
        "doc_id": "test_risk_assessment",
        "risk_items": [
            {
                "id": "privacy_1",
                "category": "프라이버시",
                "risk_item": "비식별화 처리 미흡",
                "severity_level": "높음"
            },
            {
                "id": "bias_1",
                "category": "편향성",
                "risk_item": "특정 인종 인식률 불균형",
                "severity_level": "중간"
            }
        ],
        "severity_levels": [
            {
                "item_id": "privacy_1",
                "category": "프라이버시",
                "risk_item": "비식별화 처리 미흡",
                "level": "높음",
                "weighted_score": 15.5
            },
            {
                "item_id": "bias_1",
                "category": "편향성",
                "risk_item": "특정 인종 인식률 불균형",
                "level": "중간",
                "weighted_score": 10.2
            }
        ]
    }
    
    run_improvement_suggester(test_service_info, test_risk_assessment)