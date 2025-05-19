import os
import sys
import json
from typing import Dict, List, Any
from datetime import datetime
from pydantic import BaseModel, Field

# 상위 디렉토리를 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, END
from utils.vector_db import VectorDBManager
from langchain_openai import ChatOpenAI

# 상태 클래스 정의
class EthicalRiskState(BaseModel):
    # 입력
    service_info: Dict[str, Any] = Field(default_factory=dict, description="검증된 서비스 정보")
    
    # 중간 처리 결과
    guideline_summary: Dict[str, Any] = Field(default_factory=dict, description="가이드라인 요약 정보")
    risk_categories: List[str] = Field(default_factory=list, description="분석할 리스크 카테고리")
    
    # 출력
    risk_assessments: List[Dict[str, Any]] = Field(default_factory=list, description="리스크 평가 결과")
    assessment_status: str = Field(default="", description="평가 상태 (completed, failed)")
    timestamp: str = Field(default="", description="평가 수행 시간")
    error_message: str = Field(default="", description="오류 메시지(있는 경우)")

# 에이전트 노드: GuidelineRetriever
def guideline_retriever(state: EthicalRiskState) -> EthicalRiskState:
    """
    윤리 가이드라인을 검색하고 요약합니다.
    """
    print("📚 가이드라인 검색 중...")
    
    if not state.service_info:
        return EthicalRiskState(
            service_info=state.service_info,
            assessment_status="failed",
            error_message="서비스 정보가 제공되지 않았습니다."
        )
    
    # VectorDB 검색
    try:
        db_manager = VectorDBManager()
        collection_name = "ethics_guidelines"
        
        # 주요 윤리적 주제별로 가이드라인 검색
        categories = ["편향성", "프라이버시", "투명성", "안전성", "책임성"]
        guideline_summary = {}
        
        for category in categories:
            # 카테고리별 가이드라인 검색
            results = db_manager.search(
                collection_name=collection_name,
                query=f"{category} 관련 윤리 가이드라인",
                k=3,
                filter={"type": "guideline"}
            )
            
            # 결과에서 우선순위(UNESCO > OECD > 기타)를 고려하여 정렬
            sorted_results = sorted(results, key=lambda x: x.metadata.get("priority", 999))
            
            category_items = []
            for result in sorted_results:
                org = result.metadata.get("organization", "기타")
                content = result.page_content
                
                # 페이지 번호와 문서명 추출
                page_num = result.metadata.get("page_number", "")
                file_name = result.metadata.get("file_name", "").replace(".pdf", "")
                
                category_items.append({
                    "content": content,
                    "source": f"{org} ({file_name}, p.{page_num})"
                })
            
            guideline_summary[category] = category_items
        
        print(f"✅ {len(categories)}개 카테고리의 가이드라인 검색 완료")
        
        return EthicalRiskState(
            service_info=state.service_info,
            guideline_summary=guideline_summary,
            risk_categories=categories
        )
    
    except Exception as e:
        error_message = f"가이드라인 검색 중 오류 발생: {str(e)}"
        print(f"❌ {error_message}")
        
        return EthicalRiskState(
            service_info=state.service_info,
            assessment_status="failed",
            error_message=error_message
        )

# 에이전트 노드: RiskAssessor
def risk_assessor(state: EthicalRiskState) -> EthicalRiskState:
    """
    가이드라인을 기반으로 서비스의 윤리적 리스크를 평가합니다.
    """
    print("🔍 윤리적 리스크 평가 중...")
    
    if not state.guideline_summary or not state.risk_categories:
        return EthicalRiskState(
            service_info=state.service_info,
            guideline_summary=state.guideline_summary,
            assessment_status="failed",
            error_message="가이드라인 정보가 부족합니다."
        )
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    
    service_info_text = json.dumps(state.service_info, ensure_ascii=False, indent=2)
    risk_assessments = []
    
    for category in state.risk_categories:
        print(f"  - {category} 카테고리 평가 중...")
        
        # 카테고리 관련 가이드라인 추출
        category_guidelines = state.guideline_summary.get(category, [])
        if not category_guidelines:
            continue
            
        guidelines_text = "\n\n".join([
            f"출처: {item['source']}\n내용: {item['content']}" 
            for item in category_guidelines
        ])
        
        # LLM으로 리스크 평가
        assessment_prompt = f"""
        당신은 AI 윤리 전문가입니다. 주어진 AI 서비스에 대해 "{category}" 측면의 윤리적 리스크를 평가해주세요.
        
        ## AI 서비스 정보
        ```json
        {service_info_text}
        ```
        
        ## 관련 윤리 가이드라인
        {guidelines_text}
        
        다음 구조로 평가 결과를 작성해주세요:
        1. 리스크 수준: "높음", "중간", "낮음" 중 하나로 평가
        2. 주요 리스크 요인: 최대 3개까지 구체적으로 기술
        3. 근거: 각 리스크 요인이 가이드라인의 어떤 부분을 위반하는지 설명
        4. 기준 문서: 판단의 근거가 된 주요 문서 참조
        
        출력 형식:
        {{
          "category": "{category}",
          "risk_level": "높음|중간|낮음",
          "risk_factors": [
            {{
              "name": "리스크 요인명",
              "description": "상세 설명",
              "guideline_reference": "관련 가이드라인 조항"
            }},
            ...
          ],
          "evidence": "종합적인 평가 근거",
          "reference_documents": ["참조 문서1", ...]
        }}
        """
        
        try:
            response = llm.invoke(assessment_prompt)
            assessment_text = response.content
            
            # JSON 형식 추출
            import re
            json_match = re.search(r'\{.*\}', assessment_text, re.DOTALL)
            
            if json_match:
                assessment = json.loads(json_match.group(0))
                risk_assessments.append(assessment)
            else:
                print(f"  ⚠️ {category} 평가 결과를 JSON으로 파싱할 수 없습니다.")
                
        except Exception as e:
            print(f"  ⚠️ {category} 평가 중 오류: {str(e)}")
    
    if not risk_assessments:
        return EthicalRiskState(
            service_info=state.service_info,
            guideline_summary=state.guideline_summary,
            risk_categories=state.risk_categories,
            assessment_status="failed",
            error_message="리스크 평가를 완료할 수 없습니다.",
            timestamp=datetime.now().isoformat()
        )
    
    print(f"✅ {len(risk_assessments)}개 카테고리 평가 완료")
    
    return EthicalRiskState(
        service_info=state.service_info,
        guideline_summary=state.guideline_summary,
        risk_categories=state.risk_categories,
        risk_assessments=risk_assessments,
        assessment_status="completed",
        timestamp=datetime.now().isoformat()
    )

# 그래프 구성
def create_ethical_risk_agent() -> StateGraph:
    """윤리 리스크 진단 에이전트 그래프 생성"""
    workflow = StateGraph(EthicalRiskState)
    
    # 노드 추가
    workflow.add_node("retrieve", guideline_retriever)
    workflow.add_node("assess", risk_assessor)
    
    # 엣지 추가
    workflow.add_edge("retrieve", "assess")
    workflow.add_edge("assess", END)
    
    # 시작점 설정
    workflow.set_entry_point("retrieve")
    
    return workflow

# 에이전트 실행 함수
def run_ethical_risk_agent(service_info: Dict[str, Any]) -> Dict[str, Any]:
    """윤리 리스크 진단 에이전트 실행"""
    print("🚀 윤리 리스크 진단 에이전트 시작")
    
    # 그래프 생성 및 컴파일
    graph = create_ethical_risk_agent()
    app = graph.compile()
    
    # 초기 상태 설정
    initial_state = EthicalRiskState(service_info=service_info)
    
    # 에이전트 실행
    result = app.invoke(initial_state.dict())
    
    print(f"윤리 리스크 진단 완료: 상태 = {result.assessment_status}")
    
    # 결과 반환
    return {
        "service_info": result.service_info,
        "risk_assessments": result.risk_assessments,
        "assessment_status": result.assessment_status,
        "timestamp": result.timestamp,
        "error_message": result.error_message if result.error_message else None
    }

if __name__ == "__main__":
    # 테스트용 서비스 정보
    test_service_info = {
        "title": "AI 이미지 생성 서비스",
        "domain": "창작 도구",
        "summary": "사용자가 텍스트 프롬프트를 입력하면 AI가 관련 이미지를 생성하는 서비스입니다.",
        "features": [
            {"name": "텍스트-이미지 변환", "description": "텍스트 설명을 바탕으로 이미지 생성"},
            {"name": "이미지 편집", "description": "생성된 이미지 스타일 변경 및 편집"}
        ]
    }
    
    # 에이전트 실행
    result = run_ethical_risk_agent(test_service_info)
    print(json.dumps(result, ensure_ascii=False, indent=2))