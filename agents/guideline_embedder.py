import os
import json
from typing import Dict, List
from datetime import datetime
from pydantic import BaseModel, Field
import sys

# 상위 디렉토리를 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, END

from utils.pdf_embedder import embed_pdf_documents
from utils.vector_db import VectorDBManager

# 상태 클래스 정의
class EmbeddingAgentState(BaseModel):
    # 입력
    need_embedding: List[str] = Field(default_factory=list, description="임베딩이 필요한 파일 목록")
    
    # 출력
    embedded_files: List[str] = Field(default_factory=list, description="임베딩 완료된 파일 목록")
    embedding_status: str = Field(default="", description="임베딩 상태 (completed, failed, skipped)")
    timestamp: str = Field(default="", description="최근 임베딩 수행 시간")
    error_message: str = Field(default="", description="오류 메시지(있는 경우)")

# 체크포인트 파일 경로
CHECKPOINT_FILE = "./outputs/embedding_checkpoint.json"

# 에이전트 노드: EmbeddingChecker
def embedding_checker(state: EmbeddingAgentState) -> EmbeddingAgentState:
    """
    벡터 데이터베이스에서 가이드라인 문서의 임베딩 상태를 확인하고
    아직 임베딩되지 않은 파일 목록을 반환합니다.
    """
    print("🔎 임베딩 상태 확인 중...")
    
    # 기본 가이드라인 파일 목록
    target_files = [
        "[UNESCO]AI 윤리에 관한 권고.pdf",
        "[OECD]인공지능 활용 원칙.pdf"
    ]
    
    # 체크포인트 파일 확인
    embedded_files = []
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                checkpoint_data = json.load(f)
                embedded_files = checkpoint_data.get("embedded_files", [])
                print(f"체크포인트 파일에서 {len(embedded_files)}개의 임베딩된 파일 정보를 로드했습니다.")
        except Exception as e:
            print(f"체크포인트 파일을 읽는 중 오류 발생: {e}")
    
    # VectorDB에서 직접 확인
    try:
        db_manager = VectorDBManager()
        collection_name = "ethics_guidelines"
        
        if db_manager.collection_exists(collection_name):
            collection = db_manager.get_collection(collection_name)
            results = collection.get()
            
            # 결과에서 파일명 추출하여 임베딩 상태 확인
            if results and "metadatas" in results:
                for metadata in results["metadatas"]:
                    if metadata.get("content_type") == "page" and metadata.get("file_name") not in embedded_files:
                        embedded_files.append(metadata.get("file_name"))
            
            print(f"데이터베이스에서 {len(embedded_files)}개의 임베딩된 파일을 확인했습니다.")
    except Exception as e:
        print(f"데이터베이스 확인 중 오류 발생: {e}")
    
    # 임베딩 필요한 파일 찾기
    need_embedding = []
    for file in target_files:
        if file not in embedded_files:
            data_path = os.path.join("./data", file)
            if os.path.exists(data_path):
                need_embedding.append(file)
                print(f"임베딩 필요: {file}")
            else:
                print(f"파일 없음: {file}")
    
    if not need_embedding:
        print("모든 가이드라인 파일이 이미 임베딩되어 있습니다.")
        return EmbeddingAgentState(
            embedded_files=embedded_files,
            need_embedding=[],
            embedding_status="skipped",
            timestamp=datetime.now().isoformat()
        )
    
    # 결과 반환
    return EmbeddingAgentState(
        embedded_files=embedded_files,
        need_embedding=need_embedding,
        timestamp=datetime.now().isoformat()
    )

# 에이전트 노드: GuidelineEmbedder
def guideline_embedder(state: EmbeddingAgentState) -> EmbeddingAgentState:
    """
    필요한 가이드라인 문서를 임베딩하고 결과를 반환합니다.
    """
    need_embedding = state.need_embedding
    if not need_embedding:
        return state
    
    print(f"📚 가이드라인 임베딩 시작: {need_embedding}")
    timestamp = datetime.now().isoformat()
    
    try:
        # pdf_embedder.py의 함수 활용
        embedding_result = embed_pdf_documents(
            collection_name="ethics_guidelines",
            specific_files=need_embedding,
            use_huggingface=True,
            embedding_model="nlpai-lab/KURE-v1",
            chunk_size=500,  # 청크 크기를 충분히 크게 설정
            chunk_overlap=50  # 오버랩도 적절히 설정
        )
        
        # 체크포인트 저장
        os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
        
        # 기존 임베딩 파일 목록에 새로 임베딩한 파일 추가
        all_embedded = state.embedded_files + need_embedding
        all_embedded = list(set(all_embedded))  # 중복 제거
        
        checkpoint_data = {
            "embedded_files": all_embedded,
            "last_embedding": need_embedding,
            "timestamp": timestamp
        }
        
        with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 임베딩 완료: {need_embedding}")
        
        return EmbeddingAgentState(
            embedded_files=all_embedded,
            need_embedding=[],  # 이미 처리했으므로 비움
            embedding_status="completed",
            timestamp=timestamp
        )
        
    except Exception as e:
        error_message = f"임베딩 중 오류 발생: {str(e)}"
        print(f"❌ {error_message}")
        
        return EmbeddingAgentState(
            embedded_files=state.embedded_files,  # 기존 목록 유지
            need_embedding=need_embedding,  # 처리 못한 항목 유지
            embedding_status="failed",
            timestamp=timestamp,
            error_message=error_message
        )

# 워크플로우 제어 함수
def should_embed(state: EmbeddingAgentState) -> str:
    """임베딩이 필요한지 확인하는 워크플로우 제어 함수"""
    if not state.need_embedding:
        return "end"  # 임베딩 필요 없음
    return "embed"  # 임베딩 필요

# 그래프 구성
def create_embedding_agent() -> StateGraph:
    """가이드라인 임베딩 에이전트 그래프 생성"""
    workflow = StateGraph(EmbeddingAgentState)
    
    # 노드 추가
    workflow.add_node("check", embedding_checker)
    workflow.add_node("embed", guideline_embedder)
    
    # 조건부 엣지 추가 (수정된 부분)
    workflow.add_conditional_edges(
        "check",  # 시작 노드
        should_embed,  # 조건 함수
        {
            "embed": "embed",  # should_embed가 "embed" 반환 시 "embed" 노드로
            "end": END  # should_embed가 "end" 반환 시 종료
        }
    )
    
    # embed 노드에서 종료로 가는 직접 엣지
    workflow.add_edge("embed", END)
    
    # 시작점 설정
    workflow.set_entry_point("check")
    
    return workflow

# 에이전트 실행 함수
def run_embedding_agent() -> Dict:
    """가이드라인 임베딩 에이전트 실행"""
    print("🚀 가이드라인 임베딩 에이전트 시작")
    
    # 그래프 생성 및 컴파일
    graph = create_embedding_agent()
    app = graph.compile()
    
    # 임베딩 시도할 파일 기록
    attempted_files = []
    
    # 에이전트 실행
    result = app.invoke({})
    
    # 결과 값을 딕셔너리로 추출
    embedding_status = result.get("embedding_status", "")
    
    # 결과 출력
    if embedding_status == "completed":
        print(f"✅ 임베딩 성공: {', '.join(result.get('embedded_files', []))}")
    elif embedding_status == "skipped":
        print("🔄 임베딩 생략: 모든 파일이 이미 임베딩되어 있습니다")
    else:
        print(f"❌ 임베딩 실패: {result.get('error_message', '')}")
    
    # 현재 임베딩 상태 요약 반환
    return {
        "embedded_files": result.get("embedded_files", []),
        "embedding_status": embedding_status,
        "timestamp": result.get("timestamp", ""),
        "error_message": result.get("error_message", None)
    }

if __name__ == "__main__":
    run_embedding_agent()