# filepath: /Users/lwh/SKALA/mini_project_genAI/agents/risk_assessment.py
import os
import json
from typing import Dict, List, Any, Literal
from datetime import datetime
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from utils.vector_db import VectorDBManager

# 상태 클래스 정의
class RiskAssessmentState(BaseModel):
    # 입력 데이터
    ethics_guideline: Dict = Field(default_factory=dict, description="적용할 윤리 가이드라인")
    service_info: Dict = Field(default_factory=dict, description="서비스 정보")
    scope_update: Dict = Field(default_factory=dict, description="검증된 진단 범위")
    
    # 중간 처리 데이터
    guideline_summary: str = Field(default="", description="가이드라인 요약")
    risk_items: List[Dict] = Field(default_factory=list, description="추출된 리스크 항목 목록")
    current_risk_item: Dict = Field(default_factory=dict, description="현재 평가 중인 리스크 항목")
    current_index: int = Field(default=0, description="현재 평가 중인 리스크 항목 인덱스")
    
    # 점수 관련
    score_P: int = Field(default=0, description="발생 가능성 점수 (1-5)")
    score_S: int = Field(default=0, description="심각도 점수 (1-5)")
    score_D: int = Field(default=0, description="탐지 용이성 점수 (1-5)")
    score_M: int = Field(default=0, description="완화 난이도 점수 (1-5)")
    rationale: str = Field(default="", description="점수 산정 근거")
    
    # 계산된 리스크 점수
    risk_scores: Dict = Field(default_factory=lambda: {"basic": 0, "weighted": 0}, description="계산된 리스크 점수")
    
    # 리스크 등급
    severity_level: Dict = Field(default_factory=dict, description="리스크 등급 정보")
    severity_levels: List[Dict] = Field(default_factory=list, description="모든 항목의 리스크 등급 목록")
    
    # 컨트롤 정보
    retry_count: int = Field(default=0, description="재진단 시도 횟수")
    next_node: str = Field(default="", description="다음 노드")
    
    # 출력
    assessment_result: Dict = Field(default_factory=dict, description="최종 리스크 평가 결과")
    error_message: str = Field(default="", description="오류 메시지(있는 경우)")
    timestamp: str = Field(default="", description="실행 시간")

# 에이전트 노드 함수 구현
def guideline_retriever(state: RiskAssessmentState) -> RiskAssessmentState:
    """
    윤리 가이드라인을 검색하고 요약하는 노드
    """
    print("📚 가이드라인 검색 및 요약 중...")
    
    try:
        doc_id = state.ethics_guideline.get("doc_id", "")
        if not doc_id:
            # 문서 ID가 없으면 가이드라인 문서 검색
            db_manager = VectorDBManager()
            docs = db_manager.search(
                collection_name="ethics_guidelines",
                query="UNESCO AI Ethics Recommendations OECD AI Principles",
                k=2,
                filter={"type": "guideline"}
            )
            
            if docs:
                doc_id = docs[0].metadata.get("doc_id", "")
            else:
                raise ValueError("가이드라인 문서를 찾을 수 없습니다")
        
        # LLM을 사용하여 가이드라인 요약
        llm = ChatOpenAI(model="gpt-4o-mini")
        
        # 문서 내용 가져오기
        db_manager = VectorDBManager()
        guideline_docs = db_manager.get_by_metadata(
            collection_name="ethics_guidelines",
            metadata_filter={"doc_id": doc_id}
        )
        
        if not guideline_docs:
            raise ValueError(f"문서 ID '{doc_id}'에 해당하는 가이드라인을 찾을 수 없습니다")
        
        # 가이드라인 내용
        guideline_content = "\n\n".join([doc.page_content for doc in guideline_docs[:5]])  # 처음 5개 청크만 사용
        
        # 프롬프트 템플릿
        template = """
        당신은 AI 윤리 가이드라인 전문가입니다.
        아래 가이드라인 내용에서 **편향성**, **프라이버시**, **투명성** 관련 조항을 
        우선순위(UNESCO > OECD > 기타)에 따라 요약해주세요.
        
        각 항목별로 조항 번호와 제목을 포함하여 표 형식으로 작성하세요.
        
        # 가이드라인 내용
        {guideline_content}
        
        # 요약 형식
        ## 1. 편향성(Bias) 관련 조항
        | 출처 | 조항 번호 | 제목 | 주요 내용 |
        |------|-----------|------|-----------|
        | UNESCO | 조항 x | 제목 | 내용 요약 |
        | OECD | 원칙 y | 제목 | 내용 요약 |
        
        ## 2. 프라이버시(Privacy) 관련 조항
        (동일한 표 형식)
        
        ## 3. 투명성(Transparency) 관련 조항
        (동일한 표 형식)
        
        ## 4. 기타 중요 윤리 원칙
        (동일한 표 형식)
        """
        
        # LLM 호출
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm
        
        response = chain.invoke({"guideline_content": guideline_content})
        guideline_summary = response.content
        
        return RiskAssessmentState(
            **state.model_dump(),
            guideline_summary=guideline_summary,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        return RiskAssessmentState(
            **state.model_dump(),
            error_message=f"가이드라인 검색 중 오류: {str(e)}",
            timestamp=datetime.now().isoformat()
        )

def risk_item_extractor(state: RiskAssessmentState) -> RiskAssessmentState:
    """
    서비스 정보와 검증된 범위를 바탕으로 리스크 항목을 추출
    """
    print("🔍 리스크 항목 추출 중...")
    
    if state.error_message:
        return state
    
    try:
        service_summary = state.service_info.get("summary", "")
        scope_update = state.scope_update
        
        if not service_summary:
            raise ValueError("서비스 요약 정보가 누락되었습니다")
        
        # LLM을 사용하여 리스크 항목 추출
        llm = ChatOpenAI(model="gpt-4o-mini")
        
        # 프롬프트 템플릿
        template = """
        당신은 AI 서비스 윤리 평가 전문가입니다.
        다음 서비스 요약과 검증된 범위 정보를 바탕으로 주요 윤리 항목별로 
        잠재 리스크 항목을 5~7개씩 추출하고, 간단한 설명을 덧붙여 주세요.
        
        # 서비스 요약
        {service_summary}
        
        # 검증된 범위 정보
        {scope_update}
        
        다음 카테고리별로 리스크 항목을 작성하세요:
        1. 편향성(Bias) 리스크
        2. 프라이버시(Privacy) 리스크
        3. 설명가능성(Explainability) 리스크
        
        응답은 다음 JSON 형식으로 작성하세요:
        ```json
        [
          {{
            "category": "편향성",
            "id": "bias_1",
            "risk_item": "리스크 항목 제목",
            "description": "리스크 설명(1-2문장)"
          }},
          {{
            "category": "프라이버시",
            "id": "privacy_1",
            "risk_item": "리스크 항목 제목",
            "description": "리스크 설명(1-2문장)"
          }},
          ...
        ]
        ```
        """
        
        # LLM 호출
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm
        
        scope_update_str = json.dumps(scope_update, ensure_ascii=False)
        response = chain.invoke({
            "service_summary": service_summary,
            "scope_update": scope_update_str
        })
        
        # JSON 응답 파싱
        content = response.content
        json_start = content.find("```json") + 7 if "```json" in content else content.find("[")
        json_end = content.find("```", json_start) if "```" in content[json_start:] else len(content)
        json_str = content[json_start:json_end].strip()
        
        risk_items = json.loads(json_str)
        
        # 첫 번째 리스크 항목을 현재 항목으로 설정
        current_risk_item = risk_items[0] if risk_items else {}
        
        return RiskAssessmentState(
            **state.model_dump(),
            risk_items=risk_items,
            current_risk_item=current_risk_item,
            current_index=0,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        return RiskAssessmentState(
            **state.model_dump(),
            error_message=f"리스크 항목 추출 중 오류: {str(e)}",
            timestamp=datetime.now().isoformat()
        )

def score_predictor(state: RiskAssessmentState) -> RiskAssessmentState:
    """
    현재 리스크 항목에 대한 점수 예측
    """
    current_index = state.current_index
    risk_items = state.risk_items
    
    # 모든 항목을 처리했으면 종료
    if current_index >= len(risk_items):
        return state
    
    current_risk_item = risk_items[current_index]
    print(f"🧮 리스크 항목 [{current_index+1}/{len(risk_items)}] 점수 예측 중: {current_risk_item.get('risk_item', '')}")
    
    try:
        # LLM을 사용하여 리스크 점수 예측
        llm = ChatOpenAI(model="gpt-4o-mini")
        
        # 프롬프트 템플릿
        template = """
        당신은 AI 윤리 리스크 평가 전문가입니다.
        다음 항목의 윤리 리스크를 평가하세요.
        
        # 리스크 항목
        {risk_item}
        
        # 가이드라인 요약
        {guideline_summary}
        
        다음 4가지 기준으로 1~5점을 부여하고(1=매우 낮음, 5=매우 높음), 각 점수에 대한 근거를 2문장 이내로 설명하세요:
        
        1. 발생 가능성(P): 해당 리스크가 발생할 확률
        2. 심각도(S): 발생 시 미치는 영향의 심각성
        3. 탐지 용이성(D): 리스크 발생을 얼마나 쉽게 탐지할 수 있는지 (1=매우 쉬움, 5=매우 어려움)
        4. 완화 난이도(M): 리스크를 완화하기 위한 어려움 정도 (1=매우 쉬움, 5=매우 어려움)
        
        응답은 다음 JSON 형식으로 작성하세요:
        ```json
        {{
          "P": 점수(1-5),
          "S": 점수(1-5),
          "D": 점수(1-5),
          "M": 점수(1-5),
          "rationale": "점수 산정 근거 설명"
        }}
        ```
        """
        
        # LLM 호출
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm
        
        risk_item_str = json.dumps(current_risk_item, ensure_ascii=False)
        response = chain.invoke({
            "risk_item": risk_item_str,
            "guideline_summary": state.guideline_summary
        })
        
        # JSON 응답 파싱
        content = response.content
        json_start = content.find("```json") + 7 if "```json" in content else content.find("{")
        json_end = content.find("```", json_start) if "```" in content[json_start:] else len(content)
        json_str = content[json_start:json_end].strip()
        
        scores = json.loads(json_str)
        
        return RiskAssessmentState(
            **state.model_dump(),
            current_risk_item=current_risk_item,
            score_P=scores.get("P", 0),
            score_S=scores.get("S", 0),
            score_D=scores.get("D", 0),
            score_M=scores.get("M", 0),
            rationale=scores.get("rationale", "")
        )
        
    except Exception as e:
        return RiskAssessmentState(
            **state.model_dump(),
            error_message=f"점수 예측 중 오류: {str(e)}",
            current_risk_item=current_risk_item
        )

def score_calculator(state: RiskAssessmentState) -> RiskAssessmentState:
    """
    리스크 점수 계산
    """
    P = state.score_P
    S = state.score_S
    D = state.score_D
    M = state.score_M
    
    # 기본 점수 계산: P × S
    basic_score = P * S
    
    # 가중합 계산: 0.4×P + 0.4×S + 0.1×D + 0.1×M
    weighted_score = 0.4 * P + 0.4 * S + 0.1 * D + 0.1 * M
    
    print(f"🧮 리스크 점수 계산 완료: 기본={basic_score}, 가중합={weighted_score}")
    
    return RiskAssessmentState(
        **state.model_dump(),
        risk_scores={
            "basic": basic_score,
            "weighted": weighted_score
        }
    )

def severity_classifier(state: RiskAssessmentState) -> RiskAssessmentState:
    """
    리스크 등급 분류
    """
    weighted_score = state.risk_scores.get("weighted", 0)
    
    # 등급 결정
    if weighted_score <= 6:
        level = "낮음"
    elif weighted_score <= 12:
        level = "중간"
    elif weighted_score <= 18:
        level = "높음"
    else:
        level = "심각"
    
    print(f"🔍 리스크 등급 분류: {level} (점수: {weighted_score})")
    
    severity = {
        "level": level,
        "thresholds": [
            {"range": "1-6", "level": "낮음"},
            {"range": "7-12", "level": "중간"},
            {"range": "13-18", "level": "높음"},
            {"range": "19-25", "level": "심각"}
        ]
    }
    
    # 현재 리스크 항목에 대한 결과 저장
    current_item = state.current_risk_item.copy()
    current_item.update({
        "scores": {
            "P": state.score_P,
            "S": state.score_S,
            "D": state.score_D,
            "M": state.score_M
        },
        "risk_scores": state.risk_scores,
        "severity_level": level,
        "rationale": state.rationale
    })
    
    # 처리된 항목 추가
    severity_levels = state.severity_levels.copy()
    severity_levels.append({
        "item_id": current_item.get("id", f"item_{state.current_index}"),
        "category": current_item.get("category", ""),
        "risk_item": current_item.get("risk_item", ""),
        "level": level,
        "weighted_score": weighted_score
    })
    
    # 리스크 항목 배열 업데이트
    updated_risk_items = state.risk_items.copy()
    updated_risk_items[state.current_index] = current_item
    
    # 다음 인덱스로 이동
    next_index = state.current_index + 1
    next_item = {}
    if next_index < len(updated_risk_items):
        next_item = updated_risk_items[next_index]
    
    return RiskAssessmentState(
        **state.model_dump(),
        risk_items=updated_risk_items,
        severity_level=severity,
        severity_levels=severity_levels,
        current_index=next_index,
        current_risk_item=next_item
    )

def loop_controller(state: RiskAssessmentState) -> RiskAssessmentState:
    """
    진단 루프 제어
    """
    # 모든 항목 처리 완료 확인
    if state.current_index >= len(state.risk_items):
        # 결과 저장
        doc_id = f"risk_assessment_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # ChromaDB에 저장
        assessment_result = {
            "service_name": state.service_info.get("service_name", ""),
            "risk_items": state.risk_items,
            "severity_levels": state.severity_levels,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            db_manager = VectorDBManager()
            saved_id = db_manager.add_document(
                collection_name="risk_assessments",
                content=json.dumps(assessment_result, ensure_ascii=False),
                metadata={
                    "type": "risk_assessment",
                    "service_name": state.service_info.get("service_name", ""),
                    "timestamp": datetime.now().isoformat()
                },
                doc_id=doc_id
            )[0]
            
            # 높음 또는 심각 등급 확인
            high_risks = [item for item in state.severity_levels 
                         if item.get("level") in ["높음", "심각"]]
            
            # 다음 노드 결정
            if state.retry_count < 3 and high_risks:
                next_node = "ScorePredictor"
                retry_count = state.retry_count + 1
                print(f"⚠️ 높은 리스크 항목 발견: 재진단 시도 ({retry_count}/3)")
            else:
                next_node = "ImprovementAgent"
                retry_count = state.retry_count
                print("✅ 리스크 진단 완료: 개선안 제안 단계로 이동")
            
            return RiskAssessmentState(
                **state.model_dump(),
                assessment_result={
                    "doc_id": saved_id,
                    "risk_items": state.risk_items,
                    "severity_levels": state.severity_levels
                },
                next_node=next_node,
                retry_count=retry_count
            )
            
        except Exception as e:
            return RiskAssessmentState(
                **state.model_dump(),
                error_message=f"결과 저장 중 오류: {str(e)}",
                next_node="ImprovementAgent"  # 오류 발생해도 다음 단계로 진행
            )
    else:
        # 다음 항목 처리를 위해 ScorePredictor로 돌아감
        return RiskAssessmentState(
            **state.model_dump(),
            next_node="ScorePredictor"
        )

# 워크플로우 제어 함수
def determine_next_step(state: RiskAssessmentState) -> Literal["process_next", "finalize"]:
    """모든 리스크 항목 처리 완료 여부 확인"""
    if state.current_index < len(state.risk_items):
        return "process_next"
    return "finalize"

def determine_agent_path(state: RiskAssessmentState) -> str:
    """다음 에이전트 경로 결정"""
    return state.next_node if state.next_node else "ImprovementAgent"

# 그래프 구성
def create_risk_assessment_agent() -> StateGraph:
    """윤리 리스크 진단 에이전트 그래프 생성"""
    workflow = StateGraph(RiskAssessmentState)
    
    # 노드 추가
    workflow.add_node("GuidelineRetriever", guideline_retriever)
    workflow.add_node("RiskItemExtractor", risk_item_extractor)
    workflow.add_node("ScorePredictor", score_predictor)
    workflow.add_node("ScoreCalculator", score_calculator)
    workflow.add_node("SeverityClassifier", severity_classifier)
    workflow.add_node("LoopController", loop_controller)
    
    # 기본 플로우
    workflow.add_edge("GuidelineRetriever", "RiskItemExtractor")
    workflow.add_edge("RiskItemExtractor", "ScorePredictor")
    workflow.add_edge("ScorePredictor", "ScoreCalculator")
    workflow.add_edge("ScoreCalculator", "SeverityClassifier")
    workflow.add_edge("SeverityClassifier", "LoopController")
    
    # 조건부 분기
    workflow.add_conditional_edges(
        "LoopController",
        determine_next_step,
        {
            "process_next": "ScorePredictor",
            "finalize": END
        }
    )
    
    # 시작점 설정
    workflow.set_entry_point("GuidelineRetriever")
    
    return workflow

# 에이전트 실행 함수
def run_risk_assessment(service_info: Dict, scope_update: Dict, ethics_guideline: Dict = None) -> Dict:
    """윤리 리스크 진단 에이전트 실행"""
    print("🚀 윤리 리스크 진단 에이전트 시작")
    
    # 그래프 생성 및 컴파일
    graph = create_risk_assessment_agent()
    app = graph.compile()
    
    # 에이전트 실행
    initial_state = RiskAssessmentState(
        service_info=service_info,
        scope_update=scope_update,
        ethics_guideline=ethics_guideline or {}
    )
    
    result = app.invoke(initial_state)
    
    # 결과 출력
    if result.error_message:
        print(f"❌ 리스크 진단 실패: {result.error_message}")
    else:
        print(f"✅ 리스크 진단 완료: {len(result.risk_items)} 항목 평가됨")
        
        # 등급별 항목 수 계산
        severity_counts = {}
        for level in result.severity_levels:
            category = level.get("level", "알 수 없음")
            severity_counts[category] = severity_counts.get(category, 0) + 1
        
        print("📊 등급별 항목 수:")
        for level, count in severity_counts.items():
            print(f"  - {level}: {count}개")
    
    # 평가 결과 반환
    return {
        "assessment_result": result.assessment_result,
        "next_node": result.next_node,
        "retry_count": result.retry_count
    }

if __name__ == "__main__":
    # 테스트용 데이터
    test_service_info = {
        "service_name": "AI 영상 분석 서비스",
        "company": "테스트회사",
        "service_category": "영상분석",
        "features": ["얼굴 인식", "행동 분석", "감정 인식"],
        "summary": "이 서비스는 CCTV 영상에서 얼굴을 인식하고 행동과 감정을 분석하는 AI 기반 서비스입니다."
    }
    
    test_scope_update = {
        "validated_scope": {
            "included_features": ["얼굴 인식", "행동 분석", "감정 인식"],
            "priority_areas": ["프라이버시", "편향성"]
        }
    }
    
    run_risk_assessment(test_service_info, test_scope_update)