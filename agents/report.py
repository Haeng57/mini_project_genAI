import os
import sys
import json
from typing import Dict, List, Any
from datetime import datetime
from pydantic import BaseModel, Field

# 상위 디렉토리를 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

# 상태 클래스 정의
class ReportState(BaseModel):
    # 입력
    service_info: Dict[str, Any] = Field(default_factory=dict, description="서비스 정보")
    risk_assessments: List[Dict[str, Any]] = Field(default_factory=list, description="리스크 평가 결과")
    improvement_suggestions: List[Dict[str, Any]] = Field(default_factory=list, description="개선 권고사항")
    
    # 중간 처리 결과
    report_sections: Dict[str, Any] = Field(default_factory=dict, description="보고서 섹션별 내용")
    
    # 출력
    final_report: Dict[str, Any] = Field(default_factory=dict, description="최종 보고서")
    report_status: str = Field(default="", description="보고서 작성 상태 (completed, failed)")
    timestamp: str = Field(default="", description="보고서 작성 시간")
    error_message: str = Field(default="", description="오류 메시지(있는 경우)")

# 에이전트 노드: ReportDrafter
def report_drafter(state: ReportState) -> ReportState:
    """
    진단 결과를 바탕으로 보고서의 각 섹션을 작성합니다.
    """
    print("📝 보고서 초안 작성 중...")
    
    if not state.service_info or not state.risk_assessments:
        return ReportState(
            service_info=state.service_info,
            risk_assessments=state.risk_assessments,
            improvement_suggestions=state.improvement_suggestions,
            report_status="failed",
            error_message="서비스 정보 또는 리스크 평가 결과가 부족합니다.",
            timestamp=datetime.now().isoformat()
        )
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    service_info_text = json.dumps(state.service_info, ensure_ascii=False, indent=2)
    
    # 각 섹션 작성
    report_sections = {}
    
    # 1. 개요 섹션 작성
    overview_prompt = f"""
    당신은 AI 윤리성 진단 전문가입니다. 다음 AI 서비스에 대한 진단 보고서의 '개요' 섹션을 작성해주세요.
    
    ## AI 서비스 정보
    ```json
    {service_info_text}
    ```
    
    개요 섹션에는 다음 내용을 포함하세요:
    1. 서비스 소개: 주요 기능과 목적
    2. 진단 범위: 평가 대상이 된 주요 기능 영역
    3. 진단 방법론: 국제 가이드라인(UNESCO, OECD) 기반 평가 방식
    
    출력은 마크다운 형식으로 작성하되, 표나 목록을 활용하여 가독성을 높여주세요.
    ```markdown
    # 개요
    (내용)
    ```
    """
    
    try:
        response = llm.invoke(overview_prompt)
        report_sections["overview"] = response.content
    except Exception as e:
        print(f"⚠️ 개요 섹션 작성 중 오류: {str(e)}")
        report_sections["overview"] = "# 개요\n*섹션 생성 중 오류 발생*"
    
    # 2. 주요 발견사항 섹션 작성
    risk_assessments_text = json.dumps(state.risk_assessments, ensure_ascii=False, indent=2)
    
    findings_prompt = f"""
    당신은 AI 윤리성 진단 전문가입니다. 다음 리스크 평가 결과를 바탕으로 진단 보고서의 '주요 발견사항' 섹션을 작성해주세요.
    
    ## 리스크 평가 결과
    ```json
    {risk_assessments_text}
    ```
    
    주요 발견사항 섹션에는 다음 내용을 포함하세요:
    1. 리스크 영역별 주요 이슈 요약
    2. 가장 심각한 상위 3가지 리스크 하이라이트
    3. 리스크 수준별 분포 (높음, 중간, 낮음)
    
    출력은 마크다운 형식으로 작성하되, 표나 목록을 활용하여 가독성을 높여주세요.
    ```markdown
    # 주요 발견사항
    (내용)
    ```
    """
    
    try:
        response = llm.invoke(findings_prompt)
        report_sections["findings"] = response.content
    except Exception as e:
        print(f"⚠️ 주요 발견사항 섹션 작성 중 오류: {str(e)}")
        report_sections["findings"] = "# 주요 발견사항\n*섹션 생성 중 오류 발생*"
    
    # 3. 개선 권고사항 섹션 작성
    if state.improvement_suggestions:
        improvements_text = json.dumps(state.improvement_suggestions, ensure_ascii=False, indent=2)
        
        recommendations_prompt = f"""
        당신은 AI 윤리성 진단 전문가입니다. 다음 개선 제안을 바탕으로 진단 보고서의 '개선 권고사항' 섹션을 작성해주세요.
        
        ## 개선 제안
        ```json
        {improvements_text}
        ```
        
        개선 권고사항 섹션에는 다음 내용을 포함하세요:
        1. 우선순위별 주요 개선 권고사항 요약
        2. 단기/중기/장기 개선 로드맵
        3. 이행 난이도와 기대 효과 비교
        
        출력은 마크다운 형식으로 작성하되, 표나 목록을 활용하여 가독성을 높여주세요.
        ```markdown
        # 개선 권고사항
        (내용)
        ```
        """
        
        try:
            response = llm.invoke(recommendations_prompt)
            report_sections["recommendations"] = response.content
        except Exception as e:
            print(f"⚠️ 개선 권고사항 섹션 작성 중 오류: {str(e)}")
            report_sections["recommendations"] = "# 개선 권고사항\n*섹션 생성 중 오류 발생*"
    else:
        report_sections["recommendations"] = "# 개선 권고사항\n개선 제안 정보가 제공되지 않았습니다."
    
    print(f"✅ {len(report_sections)}개 섹션 작성 완료")
    
    return ReportState(
        service_info=state.service_info,
        risk_assessments=state.risk_assessments,
        improvement_suggestions=state.improvement_suggestions,
        report_sections=report_sections
    )

# 에이전트 노드: ReportFinalizer
def report_finalizer(state: ReportState) -> ReportState:
    """
    각 섹션을 조합하고 최종 보고서를 생성합니다.
    """
    print("📊 최종 보고서 생성 중...")
    
    if not state.report_sections:
        return ReportState(
            service_info=state.service_info,
            risk_assessments=state.risk_assessments,
            improvement_suggestions=state.improvement_suggestions,
            report_sections={},
            report_status="failed",
            error_message="보고서 섹션이 준비되지 않았습니다.",
            timestamp=datetime.now().isoformat()
        )
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    
    # 각 섹션 텍스트 결합
    sections_text = "\n\n".join([
        state.report_sections.get("overview", "# 개요\n섹션 없음"),
        state.report_sections.get("findings", "# 주요 발견사항\n섹션 없음"),
        state.report_sections.get("recommendations", "# 개선 권고사항\n섹션 없음")
    ])
    
    service_name = state.service_info.get("title", "AI 서비스")
    
    # 보고서 최종화
    finalize_prompt = f"""
    당신은 AI 윤리성 진단 전문가입니다. 다음 섹션들을 바탕으로 "{service_name}에 대한 AI 윤리성 진단 보고서"를 최종화해주세요.
    
    ## 보고서 섹션
    {sections_text}
    
    다음 작업을 수행해주세요:
    1. 모든 섹션을 일관된 형식과 톤으로 통합
    2. 요약문(Executive Summary) 섹션 추가
    3. 결론 섹션 추가
    4. 적절한 표, 차트 위치 표시 (실제 차트는 생성하지 않고 [차트: 내용] 형식으로 표시)
    
    최종 보고서는 다음 구조를 따라야 합니다:
    1. 제목
    2. 요약문(Executive Summary)
    3. 개요
    4. 주요 발견사항
    5. 개선 권고사항
    6. 결론
    
    출력은 마크다운 형식으로 작성하되, 표나 목록을 활용하여 가독성을 높여주세요.
    """
    
    try:
        response = llm.invoke(finalize_prompt)
        report_content = response.content
        
        # 보고서 메타데이터 생성
        report_metadata = {
            "title": f"{service_name} AI 윤리성 진단 보고서",
            "created_at": datetime.now().isoformat(),
            "service_name": service_name,
            "risk_categories": len(state.risk_assessments),
            "improvement_count": len(state.improvement_suggestions if state.improvement_suggestions else [])
        }
        
        final_report = {
            "metadata": report_metadata,
            "content": report_content
        }
        
        # 보고서 파일 저장
        os.makedirs("./outputs/reports", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"./outputs/reports/ethics_report_{timestamp}.md"
        
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write(report_content)
            
        print(f"✅ 최종 보고서 생성 완료: {report_filename}")
        
        return ReportState(
            service_info=state.service_info,
            risk_assessments=state.risk_assessments,
            improvement_suggestions=state.improvement_suggestions,
            report_sections=state.report_sections,
            final_report=final_report,
            report_status="completed",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        error_message = f"최종 보고서 생성 중 오류 발생: {str(e)}"
        print(f"❌ {error_message}")
        
        return ReportState(
            service_info=state.service_info,
            risk_assessments=state.risk_assessments,
            improvement_suggestions=state.improvement_suggestions,
            report_sections=state.report_sections,
            report_status="failed",
            error_message=error_message,
            timestamp=datetime.now().isoformat()
        )

# 그래프 구성
def create_report_agent() -> StateGraph:
    """리포트 작성 에이전트 그래프 생성"""
    workflow = StateGraph(ReportState)
    
    # 노드 추가
    workflow.add_node("draft", report_drafter)
    workflow.add_node("finalize", report_finalizer)
    
    # 엣지 추가
    workflow.add_edge("draft", "finalize")
    workflow.add_edge("finalize", END)
    
    # 시작점 설정
    workflow.set_entry_point("draft")
    
    return workflow

# 에이전트 실행 함수
def run_report_agent(
    service_info: Dict[str, Any], 
    risk_assessments: List[Dict[str, Any]], 
    improvement_suggestions: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """리포트 작성 에이전트 실행"""
    print("🚀 리포트 작성 에이전트 시작")
    
    # 그래프 생성 및 컴파일
    graph = create_report_agent()
    app = graph.compile()
    
    # 초기 상태 설정
    initial_state = ReportState(
        service_info=service_info,
        risk_assessments=risk_assessments,
        improvement_suggestions=improvement_suggestions or []
    )
    
    # 에이전트 실행
    result = app.invoke(initial_state.dict())
    
    print(f"리포트 작성 완료: 상태 = {result.report_status}")
    
    # 결과 반환
    return {
        "report_metadata": result.final_report.get("metadata", {}),
        "report_content": result.final_report.get("content", ""),
        "report_status": result.report_status,
        "timestamp": result.timestamp,
        "error_message": result.error_message if result.error_message else None
    }

if __name__ == "__main__":
    # 테스트용 데이터
    test_service_info = {
        "title": "AI 이미지 생성 서비스",
        "domain": "창작 도구",
        "summary": "사용자가 텍스트 프롬프트를 입력하면 AI가 관련 이미지를 생성하는 서비스입니다.",
        "features": [
            {"name": "텍스트-이미지 변환", "description": "텍스트 설명을 바탕으로 이미지 생성"}
        ]
    }
    
    test_risk_assessments = [
        {
            "category": "편향성",
            "risk_level": "높음",
            "risk_factors": [
                {
                    "name": "성별 편향",
                    "description": "특정 성별을 고정관념에 따라 묘사하는 이미지를 생성함"
                }
            ]
        }
    ]
    
    test_improvements = [
        {
            "category": "편향성",
            "title": "생성 모델의 편향성 완화",
            "priority": "높음",
            "recommendations": [
                {
                    "action": "학습 데이터 다양화",
                    "detail": "다양한 문화, 성별, 인종을 포함한 데이터셋으로 모델 재학습"
                }
            ]
        }
    ]
    
    # 에이전트 실행
    result = run_report_agent(test_service_info, test_risk_assessments, test_improvements)
    print(f"보고서 생성 완료: {result.get('report_status')}")
    print(f"보고서 파일: outputs/reports/ethics_report_*.md")