import os
from datetime import datetime
import json
from dotenv import load_dotenv

# 에이전트 모듈 임포트
from agents.guideline_embedder import run_embedding_agent
from agents.service_info import run_service_analysis_agent 
from agents.scope_validator import run_scope_validator
from agents.risk_assessment import run_ethical_risk_agent
from agents.improvement_suggester import run_improvement_suggester
from agents.report import run_report_agent

# 환경변수 로드
load_dotenv()

def run_pipeline(service_name: str, service_description: str = ""):
    """
    전체 AI 윤리성 리스크 진단 파이프라인을 실행합니다.
    """
    print(f"🚀 {service_name}에 대한 AI 윤리성 리스크 진단 시작")
    start_time = datetime.now()
    
    # 1. 가이드라인 임베딩 에이전트 실행
    print("\n===== 1단계: 가이드라인 임베딩 =====")
    embedding_result = run_embedding_agent()
    
    # 임베딩 실패 시 중단
    if embedding_result.get("embedding_status") == "failed":
        print(f"❌ 가이드라인 임베딩 실패: {embedding_result.get('error_message')}")
        return {"status": "failed", "error": embedding_result.get("error_message")}
    
    # 2. 서비스 분석 에이전트 실행
    print("\n===== 2단계: 서비스 분석 =====")
    service_result = run_service_analysis_agent(service_name, service_description)
    
    # 서비스 분석 실패 시 중단
    if service_result.get("status") == "failed":
        print(f"❌ 서비스 분석 실패: {service_result.get('error_message', '알 수 없는 오류')}")
        return {"status": "failed", "error": service_result.get("error_message")}
    
    # 3. 범위 검증 에이전트 실행
    print("\n===== 3단계: 진단 범위 검증 =====")
    scope_result = run_scope_validator(service_result.get("summary", {}))
    
    # 범위 검증 실패 시 중단
    if scope_result.get("validation_status") == "failed":
        print(f"❌ 범위 검증 실패: {scope_result.get('error_message')}")
        return {"status": "failed", "error": scope_result.get("error_message")}
    
    # 4. 윤리 리스크 진단 에이전트 실행
    print("\n===== 4단계: 윤리 리스크 진단 =====")
    risk_result = run_ethical_risk_agent(scope_result.get("validated_scope", {}))
    
    # 리스크 진단 실패 시 중단
    if risk_result.get("assessment_status") == "failed":
        print(f"❌ 리스크 진단 실패: {risk_result.get('error_message')}")
        return {"status": "failed", "error": risk_result.get("error_message")}
    
    # 5. 개선안 제안 에이전트 실행
    print("\n===== 5단계: 개선안 제안 =====")
    improvement_result = run_improvement_suggester(
        service_info=scope_result.get("validated_scope", {}),
        risk_assessment=risk_result
    )
    
    # 개선안 제안 실패 여부 확인
    if improvement_result.get("error_message"):
        print(f"❌ 개선안 제안 실패: {improvement_result.get('error_message')}")
        return {"status": "failed", "error": improvement_result.get("error_message")}
    
    # 6. 리포트 작성 에이전트 실행
    print("\n===== 6단계: 최종 보고서 작성 =====")
    report_result = run_report_agent(
        service_info=scope_result.get("validated_scope", {}),
        risk_assessments=risk_result.get("risk_assessments", []),
        improvement_suggestions=improvement_result.get("suggestions", [])
    )
    
    # 보고서 작성 실패 시 메시지 출력
    if report_result.get("report_status") == "failed":
        print(f"❌ 보고서 작성 실패: {report_result.get('error_message')}")
        return {"status": "failed", "error": report_result.get("error_message")}
    
    # 전체 파이프라인 실행 시간
    end_time = datetime.now()
    execution_time = end_time - start_time
    
    print(f"\n✅ {service_name}에 대한 AI 윤리성 리스크 진단 완료")
    print(f"🕒 총 실행 시간: {execution_time}")
    
    # 결과 반환
    return {
        "status": "completed",
        "service_name": service_name,
        "report_path": f"outputs/reports/ethics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        "execution_time": str(execution_time)
    }

if __name__ == "__main__":
    # Microsoft Azure AI Vision Face API 테스트
    service_name = "Microsoft Azure AI Vision Face API"
    service_description = "얼굴 감지, 식별, 감정 분석 등 얼굴 관련 컴퓨터 비전 기능을 제공하는 클라우드 API 서비스"
    
    # 환경 변수 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        exit(1)
    
    # 파이프라인 실행
    result = run_pipeline(service_name, service_description)
    
    # 결과 출력
    if result["status"] == "completed":
        print(f"📄 최종 보고서 위치: {result['report_path']}")
    else:
        print(f"❌ 파이프라인 실행 실패: {result.get('error', '알 수 없는 오류')}")