from typing import Dict, List, Any, Optional
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END
import json
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# 상태 정의
class EthicalRiskState(BaseModel):
    # 입력 정보
    service_info: Dict[str, Any] = Field(default_factory=dict)
    
    # 가이드라인 관련 정보
    guideline_summary: Dict[str, List[Dict[str, str]]] = Field(default_factory=dict)
    
    # 리스크 관련 정보
    risk_items: List[Dict[str, Any]] = Field(default_factory=list)
    scores: Dict[str, Any] = Field(default_factory=dict)
    risk_scores: Dict[str, Any] = Field(default_factory=dict)
    severity_levels: List[Dict[str, str]] = Field(default_factory=list)
    rationale: Dict[str, str] = Field(default_factory=dict)
    
    # 워크플로우 제어 정보
    assessment_status: str = "pending"
    retry_count: int = 0
    next_node: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    error_message: Optional[str] = None
    
    # ChromaDB 연결 정보
    risk_assessments: List[Dict[str, Any]] = Field(default_factory=list)
    
    def dict(self) -> Dict[str, Any]:
        """상태 객체를 딕셔너리로 변환"""
        return {
            "service_info": self.service_info,
            "guideline_summary": self.guideline_summary,
            "risk_items": self.risk_items,
            "scores": self.scores,
            "risk_scores": self.risk_scores,
            "severity_levels": self.severity_levels,
            "rationale": self.rationale,
            "assessment_status": self.assessment_status,
            "retry_count": self.retry_count,
            "next_node": self.next_node,
            "timestamp": self.timestamp,
            "error_message": self.error_message,
            "risk_assessments": self.risk_assessments
        }


# LLM 모델 설정
llm = ChatOpenAI(
    model="gpt-4o-mini",  # README에 명시된 모델
    temperature=0.2
)

# ChromaDB 연결 설정
def get_vector_store():
    """ChromaDB 벡터 스토어 연결"""
    embedding_function = HuggingFaceEmbeddings(model_name="nlpai-lab/KURE-v1")
    vector_store = Chroma(
        persist_directory="./vector_store",
        embedding_function=embedding_function,
        collection_name="ethics_guidelines"  # 콜렉션명 명시
    )
    return vector_store


# 가이드라인 검색 노드
def guideline_retriever(state: EthicalRiskState) -> EthicalRiskState:
    """
    AI 윤리 가이드라인을 검색하여 5대 윤리 차원별로 요약합니다.
    """
    try:
        # 벡터 스토어 연결
        vector_store = get_vector_store()
        
        # 5대 윤리 차원
        ethic_dimensions = [
            "공정성", "프라이버시", "투명성", "책임성", "안전성"
        ]
        
        guideline_summary = {}
        
        # 각 차원별로 관련 가이드라인 검색
        for dimension in ethic_dimensions:
            # 검색 쿼리 생성
            query = f"AI {dimension} 윤리 가이드라인"
            
            # 벡터 검색 실행 (최대 5개 문서) - 필터 제거 또는 수정
            results = vector_store.similarity_search(
                query=query,
                k=5
            )
            
            # 검색 결과 처리
            guidelines = []
            for doc in results:
                guidelines.append({
                    "source": doc.metadata.get("file_name", "Unknown"),
                    "content": doc.page_content,
                    "page": doc.metadata.get("page_number", 0)
                })
            
            guideline_summary[dimension] = guidelines
        
        # 가이드라인 요약 생성
        system_prompt = """
        당신은 AI 윤리 가이드라인 전문가입니다. 각 윤리 차원(공정성, 프라이버시, 투명성, 책임성, 안전성)에 대한
        가이드라인을 요약하고, 각 차원의 1-5점 척도 평가 기준을 표로 작성해주세요.
        """
        
        human_prompt = f"""
        다음은 5대 윤리 차원별로 검색된 가이드라인 내용입니다:
        
        {json.dumps(guideline_summary, ensure_ascii=False, indent=2)}
        
        이를 바탕으로 각 윤리 차원(공정성, 프라이버시, 투명성, 책임성, 안전성)의 주요 평가 기준을 요약하고,
        각 차원의 1-5점 척도 평가 기준을 표로 작성해주세요.
        """
        
        # LLM으로 가이드라인 요약
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        response = llm.invoke(messages)
        
        # 요약 결과 저장
        state.guideline_summary = guideline_summary
        
        # 벡터 스토어에 요약 결과 저장
        vector_store.add_texts(
            texts=[response.content],
            metadatas=[{
                "type": "guideline_summary",
                "timestamp": datetime.now().isoformat(),
                "dimensions": ",".join(ethic_dimensions)
            }]
        )
        
        return state
        
    except Exception as e:
        state.error_message = f"가이드라인 검색 중 오류 발생: {str(e)}"
        state.assessment_status = "failed"
        return state


# 리스크 항목 추출 노드
def risk_item_extractor(state: EthicalRiskState) -> EthicalRiskState:
    """
    서비스 정보로부터 5대 윤리 차원에 따른 리스크 항목을 추출합니다.
    """
    try:
        # 서비스 정보 추출
        service_info_text = json.dumps(state.service_info, ensure_ascii=False, indent=2)
        
        # 프롬프트 템플릿 정의
        system_prompt = """
        당신은 AI 윤리 전문가입니다. 제공된 AI 서비스 정보를 분석하여, 
        5대 윤리 차원(공정성/프라이버시/투명성/책임성/안전성) 기준으로 
        5~7개의 잠재적 리스크 항목을 추출해주세요.
        
        출력은 다음 JSON 형식을 따라야 합니다:
        ```json
        [
            {
                "id": "risk_1",
                "dimension": "공정성",
                "title": "리스크 제목",
                "description": "리스크 설명"
            },
            ...
        ]
        ```
        """
        
        human_prompt = f"""
        ## AI 서비스 정보
        ```json
        {service_info_text}
        ```
        
        위 서비스에서 발생할 수 있는 5대 윤리 차원별 잠재 리스크를 5~7개 추출하고, 
        각 항목이 어느 차원(공정성/프라이버시/투명성/책임성/안전성)에 해당하는지 표시해주세요.
        """
        
        # JSON 파서 설정
        parser = JsonOutputParser()
        
        # 메시지 생성
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        # LLM으로 리스크 추출
        response = llm.invoke(messages)
        
        # JSON 추출
        try:
            # 응답에서 JSON 부분만 추출
            json_content = response.content
            if "```json" in json_content:
                json_content = json_content.split("```json")[1].split("```")[0].strip()
            elif "```" in json_content:
                json_content = json_content.split("```")[1].split("```")[0].strip()
            
            risk_items = json.loads(json_content)
            
            # 상태 업데이트
            state.risk_items = risk_items
            
        except Exception as e:
            state.error_message = f"리스크 항목 JSON 파싱 오류: {str(e)}"
            state.assessment_status = "failed"
        
        return state
        
    except Exception as e:
        state.error_message = f"리스크 항목 추출 중 오류 발생: {str(e)}"
        state.assessment_status = "failed"
        return state


# 점수 예측 노드
def score_predictor(state: EthicalRiskState) -> EthicalRiskState:
    """
    각 리스크 항목에 대해 1~5점 척도로 점수를 평가하고 근거를 제시합니다.
    """
    try:
        # 리스크 항목 및 가이드라인 요약 추출
        risk_items = state.risk_items
        guideline_summary = state.guideline_summary
        
        # 결과 저장용 딕셔너리
        scores = {}
        rationale = {}
        
        # 프롬프트 템플릿 정의
        system_prompt = """
        당신은 AI 윤리 평가 전문가입니다. 제공된 리스크 항목에 대해 해당 차원의 1~5점 척도에 따라 
        점수를 평가하고, 2문장 이내의 간결한 근거를 작성해주세요.

        각 윤리 차원별 평가 기준:
        1) 공정성(Fairness): 성별·연령·장애·지역·인종·종교 등 개인 특성에 따른 편향과 차별 최소화
        - 1점: 편향 여부 평가 프로세스 전혀 없음
        - 2점: 기본적 인식 있으나 실질적 조치 부재
        - 3점: 일부 데이터 검증 있으나 체계적이지 않음
        - 4점: 체계적 편향 평가와 일부 집단 간 성능 차이 모니터링
        - 5점: 데이터·모델 평가 시 주요 집단 간 성능 차이 2% 미만

        2) 프라이버시(Privacy): 개인정보 보호를 위한 사전 프라이버시 영향평가(PIA) 및 암호화·익명화 조치 적용
        - 1점: PIA 미실시 및 비식별화 절차 부재
        - 2점: 기초적 개인정보 식별 조치만 존재
        - 3점: 부분적 PIA 및 일부 암호화 조치
        - 4점: 체계적 PIA와 대부분의 데이터 암호화
        - 5점: 전수 PIA 수행 및 암호화·접근 통제 체계 완전 구축

        3) 투명성(Transparency): 의사결정 근거와 처리 과정을 이해관계자가 확인할 수 있는 설명 가능성 보장
        - 1점: 결과의 근거를 전혀 제공하지 않음
        - 2점: 최소한의 결과 설명만 제공
        - 3점: 부분적 설명 및 일부 의사결정 과정 공개
        - 4점: 상세한 설명과 주요 의사결정 과정 공개
        - 5점: 모델 로직·데이터 출처 문서화로 사용자 질의 응답 가능

        4) 책임성(Accountability): 윤리적 문제에 대한 책임 부담 및 독립 감사·보고 체계 마련
        - 1점: 책임 주체 및 절차 전무
        - 2점: 기본적 담당자 지정만 있음
        - 3점: 부분적 책임 체계와 간헐적 검토
        - 4점: 명확한 책임 체계와 정기적 내부 검토
        - 5점: 정기적 윤리영향평가·외부 감사를 통한 거버넌스 완전 작동

        5) 안전성(Safety & Robustness): 예기치 않은 오류·공격으로부터 안정성 유지를 위한 취약점 분석과 대응 절차 구축
        - 1점: 취약점 진단·모니터링 전무
        - 2점: 기본적 보안 점검만 시행
        - 3점: 주기적 취약점 분석 및 기본 대응책
        - 4점: 포괄적 취약점 분석 및 체계적 대응 절차
        - 5점: 위협 시나리오 테스트, 실시간 모니터링, 자동 대응 체계 완비

        출력은 다음 JSON 형식을 따라야 합니다:
        ```json
        {
            "score": 3,
            "rationale": "평가 근거를 간결하게 작성"
        }
        ```
        """
        
        # 각 리스크 항목에 대해 점수 평가
        for risk in risk_items:
            risk_id = risk["id"]
            dimension = risk["dimension"]
            title = risk["title"]
            description = risk["description"]
            
            # 해당 차원의 가이드라인 추출
            dimension_guidelines = guideline_summary.get(dimension, [])
            guidelines_text = "\n\n".join([
                f"출처: {item['source']}\n내용: {item['content']}" 
                for item in dimension_guidelines
            ])
            
            human_prompt = f"""
            ## 리스크 항목
            - ID: {risk_id}
            - 차원: {dimension}
            - 제목: {title}
            - 설명: {description}
            
            ## 관련 가이드라인
            {guidelines_text}
            
            위 리스크 항목에 대해 1~5점 척도로 점수를 평가하고 근거를 제시해주세요.
            """
            
            # 메시지 생성
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            # LLM으로 점수 예측
            response = llm.invoke(messages)
            
            # JSON 추출
            try:
                # 응답에서 JSON 부분만 추출
                json_content = response.content
                if "```json" in json_content:
                    json_content = json_content.split("```json")[1].split("```")[0].strip()
                elif "```" in json_content:
                    json_content = json_content.split("```")[1].split("```")[0].strip()
                
                result = json.loads(json_content)
                
                # 결과 저장
                scores[risk_id] = result["score"]
                rationale[risk_id] = result["rationale"]
                
            except Exception as e:
                state.error_message = f"점수 예측 JSON 파싱 오류: {str(e)}"
        
        # 상태 업데이트
        state.scores = scores
        state.rationale = rationale
        
        return state
        
    except Exception as e:
        state.error_message = f"점수 예측 중 오류 발생: {str(e)}"
        state.assessment_status = "failed"
        return state


# 점수 계산 노드
def score_calculator(state: EthicalRiskState) -> EthicalRiskState:
    """
    기본 점수와 가중 점수를 계산합니다.
    """
    try:
        # 점수 추출
        scores = state.scores
        risk_items = state.risk_items
        
        # 차원별 점수 집계
        dimension_scores = {
            "공정성": [],
            "프라이버시": [],
            "투명성": [],
            "책임성": [],
            "안전성": []
        }
        
        # 각 리스크 항목을 차원별로 분류
        for risk in risk_items:
            risk_id = risk["id"]
            dimension = risk["dimension"]
            if risk_id in scores:
                dimension_scores[dimension].append(scores[risk_id])
        
        # 차원별 평균 점수 계산
        avg_scores = {}
        for dimension, score_list in dimension_scores.items():
            if score_list:
                avg_scores[dimension] = sum(score_list) / len(score_list)
            else:
                avg_scores[dimension] = 0
        
        # 기본 점수 계산 (전체 평균)
        all_scores = list(scores.values())
        basic_score = sum(all_scores) / len(all_scores) if all_scores else 0
        
        # 가중 점수 계산
        # 가중치: 공정성(0.25), 프라이버시(0.25), 투명성(0.2), 책임성(0.15), 안전성(0.15)
        weights = {
            "공정성": 0.25,
            "프라이버시": 0.25,
            "투명성": 0.2,
            "책임성": 0.15,
            "안전성": 0.15
        }
        
        weighted_score = 0
        for dimension, weight in weights.items():
            weighted_score += avg_scores.get(dimension, 0) * weight
        
        # 상태 업데이트
        state.risk_scores = {
            "basic": basic_score,
            "weighted": weighted_score,
            "dimension_averages": avg_scores
        }
        
        return state
        
    except Exception as e:
        state.error_message = f"점수 계산 중 오류 발생: {str(e)}"
        state.assessment_status = "failed"
        return state


# 위험 등급 분류 노드
def severity_classifier(state: EthicalRiskState) -> EthicalRiskState:
    """
    가중 점수를 기반으로 위험 등급을 분류합니다.
    """
    try:
        # 가중 점수 추출
        weighted_score = state.risk_scores.get("weighted", 0)
        risk_items = state.risk_items
        scores = state.scores
        
        # 위험 등급 기준
        thresholds = [
            {"range": [1, 2], "level": "낮음"},
            {"range": [2.1, 3], "level": "중간"},
            {"range": [3.1, 4], "level": "높음"},
            {"range": [4.1, 5], "level": "심각"}
        ]
        
        # 전체 위험 등급 결정
        overall_level = "낮음"  # 기본값
        for threshold in thresholds:
            min_val, max_val = threshold["range"]
            if min_val <= weighted_score <= max_val:
                overall_level = threshold["level"]
                break
        
        # 각 리스크 항목의 위험 등급 결정
        severity_levels = []
        for risk in risk_items:
            risk_id = risk["id"]
            if risk_id in scores:
                score = scores[risk_id]
                
                # 점수에 따른 등급 결정
                level = "낮음"  # 기본값
                for threshold in thresholds:
                    min_val, max_val = threshold["range"]
                    if min_val <= score <= max_val:
                        level = threshold["level"]
                        break
                
                severity_levels.append({
                    "item_id": risk_id,
                    "level": level,
                    "score": score
                })
        
        # 상태 업데이트
        state.severity_levels = severity_levels
        
        # 전체 위험 등급 추가
        state.risk_scores["overall_level"] = overall_level
        
        return state
        
    except Exception as e:
        state.error_message = f"위험 등급 분류 중 오류 발생: {str(e)}"
        state.assessment_status = "failed"
        return state


# 루프 컨트롤러 노드
def loop_controller(state: EthicalRiskState) -> EthicalRiskState:
    """
    리스크 심각도에 따라 재진단 여부를 결정합니다.
    """
    try:
        # 위험 등급 및 재시도 횟수 확인
        severity_levels = state.severity_levels
        retry_count = state.retry_count
        
        # 고위험 항목 확인
        high_risk_exists = any(item["level"] in ["높음", "심각"] for item in severity_levels)
        
        # 다음 노드 결정
        if high_risk_exists and retry_count < 3:
            # 고위험 항목이 있고 최대 재시도 횟수를 초과하지 않은 경우 재진단
            state.next_node = "ScorePredictor"
            state.retry_count += 1
        else:
            # 고위험 항목이 없거나 최대 재시도 횟수를 초과한 경우 개선 제안으로 이동
            state.next_node = "ImprovementAgent"
            state.assessment_status = "completed"
        
        # 리스크 평가 결과 요약
        risk_summary = {
            "service_title": state.service_info.get("title", "Unknown Service"),
            "risk_scores": state.risk_scores,
            "severity_levels": state.severity_levels,
            "retry_count": state.retry_count,
            "next_node": state.next_node,
            "timestamp": datetime.now().isoformat()
        }
        
        # 상태 업데이트
        state.risk_assessments.append(risk_summary)
        
        # ChromaDB에 결과 저장
        vector_store = get_vector_store()
        vector_store.add_texts(
            texts=[json.dumps(risk_summary, ensure_ascii=False, indent=2)],
            metadatas=[{
                "type": "risk_assessment",
                "service_id": state.service_info.get("id", "unknown"),
                "timestamp": datetime.now().isoformat(),
                "overall_level": state.risk_scores.get("overall_level", "unknown")
            }]
        )
        
        return state
        
    except Exception as e:
        state.error_message = f"루프 제어 중 오류 발생: {str(e)}"
        state.assessment_status = "failed"
        return state


# 윤리 리스크 진단 에이전트
def risk_assessor(state: EthicalRiskState) -> EthicalRiskState:
    """
    가이드라인을 기반으로 서비스의 윤리적 리스크를 평가합니다.
    """
    try:
        print("🔍 윤리적 리스크 평가 시작...")
        
        # 서비스 정보 추출
        service_info_text = json.dumps(state.service_info, ensure_ascii=False, indent=2)
        
        # 결과 저장용 리스트
        risk_assessments = []
        
        # 각 윤리 차원별 평가
        for category in ["공정성", "프라이버시", "투명성", "책임성", "안전성"]:
            print(f"  - {category} 카테고리 평가 중...")
            
            # 해당 차원의 가이드라인 추출
            category_guidelines = state.guideline_summary.get(category, [])
            
            # 가이드라인 텍스트 생성 (없으면 기본값 사용)
            if category_guidelines:
                guidelines_text = "\n\n".join([
                    f"출처: {item['source']}\n내용: {item['content']}" 
                    for item in category_guidelines
                ])
            else:
                # 가이드라인이 없을 경우 기본 가이드라인 텍스트 제공
                print(f"  ⚠️ {category} 가이드라인 없음, 기본값 사용")
                guidelines_text = f"{category}에 관한 AI 윤리 가이드라인의 일반적 원칙을 적용하세요."
            
            # LLM으로 리스크 평가
            assessment_prompt = f"""
                        당신은 AI 윤리 전문가입니다. 주어진 AI 서비스에 대해 "{category}" 측면의 윤리적 리스크를 평가해주세요.
                        
                        ## AI 서비스 정보
                        ```json
                        {service_info_text}
                        ```
                        
                        ## 관련 윤리 가이드라인
                        {guidelines_text}
                        
                        ## 지시사항
                        1. "{category}" 측면에서 이 서비스의 주요 윤리적 리스크를 3-5가지 식별하세요.
                        2. 각 리스크의 심각도를 '낮음', '중간', '높음', '심각' 중 하나로 평가하세요.
                        3. 각 리스크에 대한 근거와 예방/완화 방안을 제안하세요.
                        4. 1-5점 척도로 이 차원의 전반적인 윤리적 위험 점수를 매기세요.
                        
                        다음 JSON 형식으로 결과를 반환하세요:
                        ```json
                        {{
                            "dimension": "{category}",
                            "risks": [
                                {{
                                    "title": "리스크 제목",
                                    "severity": "중간",
                                    "description": "리스크 설명",
                                    "evidence": "근거",
                                    "mitigation": "완화 방안"
                                }}
                            ],
                            "overall_score": 3,
                            "rationale": "전반적인 점수에 대한 근거"
                        }}
                        ```
                        """
            
            # 메시지 생성
            messages = [
                SystemMessage(content="당신은 AI 윤리 전문가입니다. 주어진 지시에 따라 AI 서비스의 윤리적 리스크를 평가하세요."),
                HumanMessage(content=assessment_prompt)
            ]
            
            try:
                # LLM으로 리스크 평가
                response = llm.invoke(messages)
                
                # JSON 추출
                try:
                    # 응답에서 JSON 부분만 추출
                    json_content = response.content
                    if "```json" in json_content:
                        json_content = json_content.split("```json")[1].split("```")[0].strip()
                    elif "```" in json_content:
                        json_content = json_content.split("```")[1].split("```")[0].strip()
                    
                    assessment = json.loads(json_content)
                    risk_assessments.append(assessment)
                    print(f"  ✓ {category} 평가 완료")
                    
                except Exception as e:
                    print(f"  ⚠️ {category} 평가 JSON 파싱 오류: {str(e)}")
                    # JSON 파싱 실패 시 기본 결과 생성
                    fallback_assessment = {
                        "dimension": category,
                        "risks": [
                            {
                                "title": f"{category} 관련 리스크",
                                "severity": "중간",
                                "description": "자동 생성된 기본 리스크 항목",
                                "evidence": "JSON 파싱 오류로 인한 기본 항목",
                                "mitigation": "상세 리스크 평가 필요"
                            }
                        ],
                        "overall_score": 3,
                        "rationale": "JSON 파싱 오류로 인한 기본 평가"
                    }
                    risk_assessments.append(fallback_assessment)
                    
            except Exception as e:
                print(f"  ⚠️ {category} 평가 중 오류 발생: {str(e)}")
        
        # 상태 업데이트
        print(f"✅ 총 {len(risk_assessments)}개 카테고리 평가 완료")
        state.risk_assessments = risk_assessments
        state.assessment_status = "completed"
        
        return state
        
    except Exception as e:
        state.error_message = f"윤리 리스크 평가 중 오류 발생: {str(e)}"
        state.assessment_status = "failed"
        print(f"❌ 윤리 리스크 평가 실패: {str(e)}")
        return state
        
    except Exception as e:
        state.error_message = f"윤리 리스크 평가 중 오류 발생: {str(e)}"
        state.assessment_status = "failed"
        return state


# 그래프 구성
def create_ethical_risk_agent() -> StateGraph:
    """윤리 리스크 진단 에이전트 그래프 생성"""
    workflow = StateGraph(EthicalRiskState)
    
    # 노드 추가
    workflow.add_node("retrieve", guideline_retriever)
    workflow.add_node("extract", risk_item_extractor)
    workflow.add_node("predict", score_predictor)
    workflow.add_node("calculate", score_calculator)
    workflow.add_node("classify", severity_classifier)
    workflow.add_node("control", loop_controller)
    workflow.add_node("assess", risk_assessor)
    
    # 엣지 추가
    workflow.add_edge("retrieve", "extract")
    workflow.add_edge("extract", "predict")
    workflow.add_edge("predict", "calculate")
    workflow.add_edge("calculate", "classify")
    workflow.add_edge("classify", "control")
    
    # 수정: 조건부 엣지 수정 - 항상 assess 노드로 이동하도록 설정
    workflow.add_conditional_edges(
        "control",
        lambda state: state.next_node,
        {
            "ScorePredictor": "predict",
            "ImprovementAgent": "assess"
        }
    )
    
    # 최종 엣지 추가 
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
    
    # 디버깅 정보 추가
    print(f"윤리 리스크 진단 완료: 상태 = {result['assessment_status']}")
    print(f"리스크 평가 결과 수: {len(result['risk_assessments'])}")
    
    # 결과 반환
    return {
        "service_info": result["service_info"],
        "risk_assessments": result["risk_assessments"],
        "assessment_status": result["assessment_status"],
        "timestamp": result["timestamp"],
        "error_message": result.get("error_message")
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
        ],
        "target_users": [
            "보안 시스템 개발자", "마케팅 분석가", "UX 연구원", "접근성 개발자"
        ],
        "data_sources": [
            "사용자 업로드 이미지", "비디오 프레임", "저장된 얼굴 템플릿"
        ],
        "algorithms": [
            "딥러닝 CNN", "얼굴 임베딩", "특징점 추출"
        ]
    }
    
    # 에이전트 실행
    result = run_ethical_risk_agent(test_service_info)
    print(json.dumps(result, ensure_ascii=False, indent=2))