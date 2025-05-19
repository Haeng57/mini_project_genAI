import os
import json
import time
from datetime import datetime
from typing import Dict, Any
from pydantic import BaseModel, Field
import sys

# 현재 디렉토리를 포함하도록 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langgraph.graph import StateGraph, END

# 에이전트 모듈 임포트
from agents.guideline_embedder import run_embedding_agent
from agents.service_info import run_service_analysis_agent
from agents.scope_validator import run_scope_validator
from agents.ethical_risk import run_ethical_risk_agent
from agents.improvement import run_improvement_agent
from agents.report import run_report_agent

# 전체 시스템 상태 정의
class SystemState(BaseModel):
    # 임베딩 상태
    guideline_embedding: Dict[str, Any] = Field(default_factory=dict)
    
    # 서비스 정보
    service_info: Dict[str, Any] = Field(default_factory=dict)
    
    # 범위 검증 결과
    scope_update: Dict[str, Any] = Field(default_factory=dict)
    
    # 윤리 가이드라인
    ethics_guideline: Dict[str, Any] = Field(default_factory=dict)
    
    # 리스크 평가 결과
    risk_assessment: Dict[str, Any] = Field(default_factory=dict)
    
    # 개선 제안
    improvement_suggestion: Dict[str, Any] = Field(default_factory=dict)
    
    # 최종 보고서
    report: Dict[str, Any] = Field(default_factory=dict)
    
    # 워크플로우 제어
    workflow_control: Dict[str, Any] = Field(
        default_factory=lambda: {
            "current_step": "guideline_embedding",
            "retry_counts": {"risk_assessment": 0, "improvement": 0, "scope": 0},
            "error_messages": []
        }
    )

# 노드 1: 가이드라인 임베딩 에이전트
def guideline_embedding_node(state: SystemState) -> SystemState:
    """가이드라인 임베딩 에이전트 실행"""
    print("\n🚀 Step 1: 가이드라인 임베딩 에이전트 실행 중...")
    
    try:
        result = run_embedding_agent()
        state.guideline_embedding = result
        state.workflow_control["current_step"] = "service_analysis"
    except Exception as e:
        state.workflow_control["error_messages"].append(f"가이드라인 임베딩 에러: {str(e)}")
        state.workflow_control["current_step"] = "end"  # 에러 발생 시 종료
    
    return state

# 노드 2: 서비스 분석 에이전트
def service_analysis_node(state: SystemState) -> SystemState:
    """서비스 분석 에이전트 실행"""
    print("\n🚀 Step 2: 서비스 분석 에이전트 실행 중...")
    
    # 테스트용 서비스 이름
    service_name = "AI 이미지 생성 서비스"
    
    try:
        # 해당 함수 존재 가정 (없으면 구현 필요)
        result = run_service_analysis_agent(service_name)
        state.service_info = result
        state.workflow_control["current_step"] = "scope_validation"
    except Exception as e:
        state.workflow_control["error_messages"].append(f"서비스 분석 에러: {str(e)}")
        state.workflow_control["current_step"] = "end"
    
    return state

# 노드 3: 범위 검증 에이전트
def scope_validation_node(state: SystemState) -> SystemState:
    """범위 검증 에이전트 실행"""
    print("\n🚀 Step 3: 범위 검증 에이전트 실행 중...")
    
    try:
        result = run_scope_validator(state.service_info)
        
        # 범위 재조정 필요 여부 확인
        scope_needs_update = any(update.get("update_type") == "major_change" 
                             for update in result.get("scope_updates", []))
        
        if scope_needs_update and state.workflow_control["retry_counts"]["scope"] < 2:
            # 범위 재조정이 필요하면 서비스 분석으로 돌아감
            state.scope_update = result
            state.workflow_control["retry_counts"]["scope"] += 1
            state.workflow_control["current_step"] = "service_analysis"
            print("⚠️ 범위 재조정 필요: 서비스 분석 단계로 돌아갑니다.")
        else:
            # 정상 진행
            state.scope_update = result
            state.service_info = result["validated_scope"]  # 검증된 서비스 정보로 업데이트
            state.workflow_control["current_step"] = "ethical_risk"
    except Exception as e:
        state.workflow_control["error_messages"].append(f"범위 검증 에러: {str(e)}")
        state.workflow_control["current_step"] = "end"
    
    return state

# 노드 4: 윤리 리스크 진단 에이전트
def ethical_risk_node(state: SystemState) -> SystemState:
    """윤리 리스크 진단 에이전트 실행"""
    print("\n🚀 Step 4: 윤리 리스크 진단 에이전트 실행 중...")
    
    try:
        result = run_ethical_risk_agent(state.service_info)
        state.risk_assessment = result
        
        # 높은 리스크가 있는지 확인
        has_high_risk = any(
            assessment.get("risk_level") in ["높음", "심각"]
            for assessment in result.get("risk_assessments", [])
        )
        
        # 재시도 횟수 확인
        retry_count = state.workflow_control["retry_counts"]["risk_assessment"]
        
        if has_high_risk and retry_count < 3:
            # 높은 리스크가 있고 재시도 횟수가 3 미만이면 개선안 제안 후 재진단
            state.workflow_control["retry_counts"]["risk_assessment"] += 1
            state.workflow_control["current_step"] = "improvement"
        else:
            # 그렇지 않으면 정상적으로 개선안 제안으로 이동
            state.workflow_control["current_step"] = "improvement"
        
    except Exception as e:
        state.workflow_control["error_messages"].append(f"윤리 리스크 진단 에러: {str(e)}")
        state.workflow_control["current_step"] = "end"
    
    return state

# 노드 5: 개선안 제안 에이전트
def improvement_node(state: SystemState) -> SystemState:
    """개선안 제안 에이전트 실행"""
    print("\n🚀 Step 5: 개선안 제안 에이전트 실행 중...")
    
    try:
        result = run_improvement_agent(
            state.service_info,
            state.risk_assessment.get("risk_assessments", [])
        )
        state.improvement_suggestion = result
        
        # 진단 재시도 여부 확인
        retry_count = state.workflow_control["retry_counts"]["risk_assessment"]
        if retry_count > 0 and retry_count < 3:
            # 재진단 모드였다면 다시 윤리 리스크 진단으로
            state.workflow_control["current_step"] = "ethical_risk"
            print(f"⚠️ 높은 리스크 발견: 개선안 적용 후 재진단 ({retry_count}/3)")
        else:
            # 아니면 정상적으로 리포트 작성으로 이동
            state.workflow_control["current_step"] = "report"
    except Exception as e:
        state.workflow_control["error_messages"].append(f"개선안 제안 에러: {str(e)}")
        state.workflow_control["current_step"] = "end"
    
    return state

# 노드 6: 리포트 작성 에이전트
def report_node(state: SystemState) -> SystemState:
    """리포트 작성 에이전트 실행"""
    print("\n🚀 Step 6: 리포트 작성 에이전트 실행 중...")
    
    try:
        result = run_report_agent(
            state.service_info,
            state.risk_assessment.get("risk_assessments", []),
            state.improvement_suggestion.get("improvement_suggestions", [])
        )
        state.report = result
        
        # 보고서 검토 미흡 여부 (예: 보고서 품질이 일정 기준 미달)
        needs_improvement = False  # 실제로는 보고서 품질 평가 로직이 필요
        
        if needs_improvement:
            state.workflow_control["retry_counts"]["improvement"] += 1
            state.workflow_control["current_step"] = "improvement"
            print("⚠️ 보고서 검토 미흡: 개선안 재검토 필요")
        else:
            state.workflow_control["current_step"] = "end"  # 성공적으로 종료
            print("\n✅ AI 윤리성 리스크 진단 완료!")
            print(f"📊 결과 보고서: {os.path.basename(result.get('report_path', ''))}")
    except Exception as e:
        state.workflow_control["error_messages"].append(f"리포트 작성 에러: {str(e)}")
        state.workflow_control["current_step"] = "end"
    
    return state

# 워크플로우 제어 함수
def router(state: SystemState) -> str:
    """현재 상태에 따라 다음 노드를 결정합니다"""
    return state.workflow_control["current_step"]

# 그래프 생성
def create_workflow_graph() -> StateGraph:
    """전체 워크플로우 그래프 생성"""
    workflow = StateGraph(SystemState)
    
    # 노드 추가
    workflow.add_node("guideline_embedding", guideline_embedding_node)
    workflow.add_node("service_analysis", service_analysis_node)
    workflow.add_node("scope_validation", scope_validation_node)
    workflow.add_node("ethical_risk", ethical_risk_node)
    workflow.add_node("improvement", improvement_node)
    workflow.add_node("report", report_node)
    workflow.add_node("end", lambda x: x)  # 종료 노드
    
    # 라우터 설정
    workflow.set_conditional_edges(
        router,
        {
            "guideline_embedding": "guideline_embedding",
            "service_analysis": "service_analysis",
            "scope_validation": "scope_validation",
            "ethical_risk": "ethical_risk",
            "improvement": "improvement",
            "report": "report",
            "end": END
        }
    )
    
    # 시작점 설정
    workflow.set_entry_point("guideline_embedding")
    
    return workflow

def main():
    """메인 함수: 전체 워크플로우 실행"""
    print("=" * 70)
    print("🤖 AI 윤리성 리스크 진단 멀티에이전트 시스템 시작")
    print("=" * 70)
    
    # 출력 디렉토리 생성
    os.makedirs("./outputs/reports", exist_ok=True)
    
    # 그래프 생성 및 컴파일
    graph = create_workflow_graph()
    app = graph.compile()
    
    # 초기 상태 설정
    initial_state = SystemState()
    
    # 워크플로우 실행
    start_time = time.time()
    final_state = app.invoke(initial_state)
    duration = time.time() - start_time
    
    # 결과 출력
    print("\n" + "=" * 70)
    print(f"🏁 AI 윤리성 리스크 진단 완료! (소요 시간: {duration:.2f}초)")
    
    if final_state.workflow_control["error_messages"]:
        print("\n⚠️ 실행 중 발생한 오류:")
        for error in final_state.workflow_control["error_messages"]:
            print(f" - {error}")
    
    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = f"./outputs/system_result_{timestamp}.json"
    
    with open(result_path, "w", encoding="utf-8") as f:
        # 객체를 직렬화 가능한 딕셔너리로 변환
        result_dict = {
            "guideline_embedding": final_state.guideline_embedding,
            "service_info": final_state.service_info,
            "scope_update": final_state.scope_update,
            "risk_assessment": final_state.risk_assessment,
            "improvement_suggestion": final_state.improvement_suggestion,
            "report": final_state.report,
            "workflow_control": final_state.workflow_control
        }
        json.dump(result_dict, f, ensure_ascii=False, indent=2)
    
    print(f"📄 시스템 실행 결과가 저장되었습니다: {result_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()