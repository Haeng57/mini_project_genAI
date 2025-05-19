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
class ImprovementState(BaseModel):
    # 입력
    service_info: Dict[str, Any] = Field(default_factory=dict, description="서비스 정보")
    risk_assessments: List[Dict[str, Any]] = Field(default_factory=list, description="리스크 평가 결과")
    
    # 중간 처리 결과
    best_practices: List[Dict[str, Any]] = Field(default_factory=list, description="관련 모범 사례")
    
    # 출력
    improvement_suggestions: List[Dict[str, Any]] = Field(default_factory=list, description="개선 권고사항")
    improvement_status: str = Field(default="", description="개선안 도출 상태 (completed, failed)")
    timestamp: str = Field(default="", description="개선안 도출 시간")
    error_message: str = Field(default="", description="오류 메시지(있는 경우)")

# 에이전트 노드: BestPracticeRetriever
def best_practice_retriever(state: ImprovementState) -> ImprovementState:
    """
    리스크 유형별 모범 사례를 검색합니다.
    """
    print("📚 윤리적 모범 사례 검색 중...")
    
    if not state.risk_assessments:
        return ImprovementState(
            service_info=state.service_info,
            risk_assessments=state.risk_assessments,
            improvement_status="failed",
            error_message="리스크 평가 결과가 제공되지 않았습니다.",
            timestamp=datetime.now().isoformat()
        )
    
    # 리스크 카테고리와 요인 추출
    categories = []
    for assessment in state.risk_assessments:
        categories.append(assessment.get("category", ""))
    
    # VectorDB 검색
    try:
        db_manager = VectorDBManager()
        collection_name = "ethics_guidelines"
        
        best_practices = []
        for category in categories:
            # 카테고리별 모범 사례 검색
            query = f"{category} 개선 방안 모범 사례"
            results = db_manager.search(
                collection_name=collection_name,
                query=query,
                k=2
            )
            
            for result in results:
                practice = {
                    "category": category,
                    "content": result.page_content,
                    "source": result.metadata.get("file_name", "").replace(".pdf", ""),
                    "page": result.metadata.get("page_number", "")
                }
                best_practices.append(practice)
        
        print(f"✅ {len(best_practices)}개의 모범 사례를 찾았습니다.")
        
        return ImprovementState(
            service_info=state.service_info,
            risk_assessments=state.risk_assessments,
            best_practices=best_practices
        )
    
    except Exception as e:
        error_message = f"모범 사례 검색 중 오류 발생: {str(e)}"
        print(f"❌ {error_message}")
        
        # 실패 시에도 다음 단계 진행 가능하도록 빈 best_practices와 함께 반환
        return ImprovementState(
            service_info=state.service_info,
            risk_assessments=state.risk_assessments,
            best_practices=[]
        )

# 에이전트 노드: ImprovementGenerator
def improvement_generator(state: ImprovementState) -> ImprovementState:
    """
    리스크 평가 결과와 모범 사례를 기반으로 개선 권고안을 생성합니다.
    """
    print("💡 개선 권고안 생성 중...")
    
    if not state.risk_assessments:
        return ImprovementState(
            service_info=state.service_info,
            risk_assessments=state.risk_assessments,
            best_practices=state.best_practices,
            improvement_status="failed",
            error_message="리스크 평가 결과가 제공되지 않았습니다.",
            timestamp=datetime.now().isoformat()
        )
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    service_info_text = json.dumps(state.service_info, ensure_ascii=False, indent=2)
    improvement_suggestions = []
    
    for assessment in state.risk_assessments:
        category = assessment.get("category", "")
        risk_level = assessment.get("risk_level", "")
        risk_factors = assessment.get("risk_factors", [])
        
        if not category or not risk_factors:
            continue
            
        # 해당 카테고리의 모범 사례 검색
        category_practices = []
        for practice in state.best_practices:
            if practice.get("category") == category:
                category_practices.append(practice)
        
        practices_text = "\n\n".join([
            f"출처: {practice.get('source')} (p.{practice.get('page')})\n내용: {practice.get('content')}" 
            for practice in category_practices
        ]) if category_practices else "관련 모범 사례를 찾을 수 없습니다."
        
        # 리스크 요인 텍스트 생성
        risk_factors_text = "\n".join([
            f"- {factor.get('name')}: {factor.get('description')}"
            for factor in risk_factors
        ])
        
        # LLM으로 개선 권고안 생성
        improvement_prompt = f"""
        당신은 AI 윤리 전문가입니다. 주어진 AI 서비스의 "{category}" 측면에서 발견된 윤리적 리스크를 개선하기 위한 구체적인 권고안을 제시해주세요.
        
        ## AI 서비스 정보
        ```json
        {service_info_text}
        ```
        
        ## 발견된 리스크 (수준: {risk_level})
        {risk_factors_text}
        
        ## 관련 모범 사례
        {practices_text}
        
        다음 구조로 개선 권고안을 작성해주세요:
        1. 개선 제목: 간결하고 명확한 제목
        2. 개선 우선순위: "높음", "중간", "낮음" 중 하나로 평가
        3. 개선 권고사항: 최대 3개의 구체적인 개선 방안
        4. 이행 난이도: "쉬움", "보통", "어려움" 중 하나로 평가
        5. 기대 효과: 개선 시 예상되는 긍정적 효과
        
        출력 형식:
        {{
          "category": "{category}",
          "title": "개선 제목",
          "priority": "높음|중간|낮음",
          "recommendations": [
            {{
              "action": "개선 행동",
              "detail": "구체적 방법",
              "rationale": "개선 근거"
            }},
            ...
          ],
          "implementation_difficulty": "쉬움|보통|어려움",
          "expected_benefits": "기대되는 긍정적 효과"
        }}
        """
        
        try:
            response = llm.invoke(improvement_prompt)
            improvement_text = response.content
            
            # JSON 형식 추출
            import re
            json_match = re.search(r'\{.*\}', improvement_text, re.DOTALL)
            
            if json_match:
                suggestion = json.loads(json_match.group(0))
                improvement_suggestions.append(suggestion)
            else:
                print(f"  ⚠️ {category} 개선안을 JSON으로 파싱할 수 없습니다.")
                
        except Exception as e:
            print(f"  ⚠️ {category} 개선안 생성 중 오류: {str(e)}")
    
    if not improvement_suggestions:
        return ImprovementState(
            service_info=state.service_info,
            risk_assessments=state.risk_assessments,
            best_practices=state.best_practices,
            improvement_status="failed",
            error_message="개선 권고안을 생성할 수 없습니다.",
            timestamp=datetime.now().isoformat()
        )
    
    print(f"✅ {len(improvement_suggestions)}개 카테고리의 개선 권고안 생성 완료")
    
    return ImprovementState(
        service_info=state.service_info,
        risk_assessments=state.risk_assessments,
        best_practices=state.best_practices,
        improvement_suggestions=improvement_suggestions,
        improvement_status="completed",
        timestamp=datetime.now().isoformat()
    )

# 그래프 구성
def create_improvement_agent() -> StateGraph:
    """개선안 제안 에이전트 그래프 생성"""
    workflow = StateGraph(ImprovementState)
    
    # 노드 추가
    workflow.add_node("retrieve", best_practice_retriever)
    workflow.add_node("generate", improvement_generator)
    
    # 엣지 추가
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    # 시작점 설정
    workflow.set_entry_point("retrieve")
    
    return workflow

# 에이전트 실행 함수
def run_improvement_agent(service_info: Dict[str, Any], risk_assessments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """개선안 제안 에이전트 실행"""
    print("🚀 개선안 제안 에이전트 시작")
    
    # 그래프 생성 및 컴파일
    graph = create_improvement_agent()
    app = graph.compile()
    
    # 초기 상태 설정
    initial_state = ImprovementState(
        service_info=service_info,
        risk_assessments=risk_assessments
    )
    
    # 에이전트 실행
    result = app.invoke(initial_state.dict())
    
    print(f"개선안 제안 완료: 상태 = {result.improvement_status}")
    
    # 결과 반환
    return {
        "service_info": result.service_info,
        "improvement_suggestions": result.improvement_suggestions,
        "improvement_status": result.improvement_status,
        "timestamp": result.timestamp,
        "error_message": result.error_message if result.error_message else None
    }

if __name__ == "__main__":
    # 테스트용 데이터
    test_service_info = {
        "title": "AI 이미지 생성 서비스",
        "domain": "창작 도구",
        "summary": "사용자가 텍스트 프롬프트를 입력하면 AI가 관련 이미지를 생성하는 서비스입니다."
    }
    
    test_risk_assessments = [
        {
            "category": "편향성",
            "risk_level": "높음",
            "risk_factors": [
                {
                    "name": "성별 편향",
                    "description": "특정 성별을 고정관념에 따라 묘사하는 이미지를 생성함",
                    "guideline_reference": "UNESCO AI 윤리 권고 42항"
                }
            ]
        }
    ]
    
    # 에이전트 실행
    result = run_improvement_agent(test_service_info, test_risk_assessments)
    print(json.dumps(result, ensure_ascii=False, indent=2))