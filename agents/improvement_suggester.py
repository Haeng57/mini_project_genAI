# filepath: /Users/lwh/SKALA/mini_project_genAI/agents/improvement_suggester.py
import os
import json
from typing import Dict
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
    if not risk_assessment:
        return ImprovementSuggesterState(
            service_info=state.service_info,
            risk_assessment=risk_assessment,
            error_message="리스크 평가 결과가 누락되었습니다",
            timestamp=datetime.now().isoformat()
        )
    
    try:
        # 리스크 평가 구조 확인 - risk_items 또는 risk_assessments 확인
        risk_items = []
        
        # risk_assessment.py에서 반환한 risk_assessments 필드 확인
        if "risk_assessments" in risk_assessment:
            # risk_assessments 데이터로 작업
            risk_assessments = risk_assessment.get("risk_assessments", [])
            
            # 각 카테고리의 리스크 항목 추출
            for assessment in risk_assessments:
                risks = assessment.get("risks", [])
                dimension = assessment.get("dimension", "unknown")
                for i, risk in enumerate(risks):
                    risk_items.append({
                        "item_id": f"{dimension}_{i}",
                        "category": dimension,
                        "risk_item": risk.get("title", ""),
                        "level": risk.get("severity", "중간")
                    })
        
        # 기존 구조도 확인
        elif "risk_items" in risk_assessment:
            risk_items = risk_assessment.get("risk_items", [])
        
        # severity_levels가 있으면 사용
        elif "severity_levels" in risk_assessment:
            risk_items = risk_assessment.get("severity_levels", [])
        
        # 리스크 항목이 없는 경우
        if not risk_items:
            return ImprovementSuggesterState(
                service_info=state.service_info,
                risk_assessment=risk_assessment,
                error_message="식별된 리스크 항목이 없습니다",
                timestamp=datetime.now().isoformat()
            )
        
        # 심각한 리스크 항목들 추출
        high_risk_items = [item for item in risk_items 
                          if item.get("level", "").lower() in ["높음", "심각", "high", "severe"]]
        
        # 중간 리스크 항목들 추출
        medium_risk_items = [item for item in risk_items 
                            if item.get("level", "").lower() in ["중간", "medium"]]
        
        # 최대 3개의 심각한 리스크와 2개의 중간 리스크 선택
        selected_high = high_risk_items[:3]
        selected_medium = medium_risk_items[:2]
        
        selected_items = selected_high + selected_medium
        
        # 선택된 항목이 없으면 모든 항목 사용
        if not selected_items and risk_items:
            selected_items = risk_items[:5]
        
        # 벡터 DB에서 관련 모범 사례 검색
        best_practices = {}
        db_manager = VectorDBManager()
        
        for item in selected_items:
            category = item.get("category", "")
            risk_item = item.get("risk_item", "")
            query = f"{category} {risk_item} best practices solutions"
            
            try:
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
            except Exception as e:
                print(f"  ⚠️ 모범 사례 검색 중 오류: {str(e)}")
        
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
    
    # 결과 출력 - 딕셔너리 접근 방식으로 수정
    if result.get("error_message"):
        print(f"❌ 개선안 제안 실패: {result.get('error_message')}")
    else:
        suggestions_count = len(result.get("improvement_suggestion", {}).get("suggestions", []))
        print(f"✅ 개선안 제안 완료: {suggestions_count}개 개선안 제안됨")
    
    # 개선 제안 결과 반환
    return result.get("improvement_suggestion", {})

if __name__ == "__main__":
    # Microsoft Azure AI Vision Face API 테스트
    test_service_info = {
        "title": "Microsoft Azure AI Vision Face API",
        "domain": "컴퓨터 비전 / 얼굴 인식",
        "summary": "얼굴 감지, 식별, 감정 분석 등 얼굴 관련 컴퓨터 비전 기능을 제공하는 클라우드 API 서비스"
    }
    
    test_risk_assessment = {
        "doc_id": "test_risk_assessment",
        "risk_items": [
            {
                "id": "bias_1",
                "category": "편향성",
                "risk_item": "인구통계학적 편향",
                "severity_level": "높음"
            },
            {
                "id": "privacy_1",
                "category": "프라이버시",
                "risk_item": "얼굴 데이터 수집 및 저장",
                "severity_level": "심각"
            }
        ],
        "severity_levels": [
            {
                "item_id": "bias_1",
                "category": "편향성",
                "risk_item": "인구통계학적 편향",
                "level": "높음",
                "weighted_score": 4.2
            },
            {
                "item_id": "privacy_1",
                "category": "프라이버시",
                "risk_item": "얼굴 데이터 수집 및 저장",
                "level": "심각",
                "weighted_score": 4.8
            }
        ]
    }
    
    # 에이전트 실행
    result = run_improvement_suggester(test_service_info, test_risk_assessment)
    if result.get("error_message"):
        print(f"❌ 개선안 제안 실패: {result['error_message']}")
    else:
        print(f"✅ 개선안 제안 완료")
        print(json.dumps(result.get("suggestions", []), ensure_ascii=False, indent=2))