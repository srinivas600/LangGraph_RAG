import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
import uuid
from datetime import datetime

from database.models import (
    create_tables, get_db, TelemetryLog, FeedbackLog,
    ChatSession, ChatHistory, EvaluationMetrics
)
from agents.langgraph_agent import AgenticRAGBot
from utils.embeddings import ChromaVectorStore
from evaluation.evaluator import evaluate_response_sync
from config.settings import Config

# Pydantic models for API
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    transaction_id: str
    session_id: str
    retrieved_documents: List[dict]
    route_taken: str

class FeedbackRequest(BaseModel):
    transaction_id: str
    feedback: str
    feedback_text: Optional[str] = None

class DocumentRequest(BaseModel):
    content: str
    metadata: Optional[dict] = None

class SessionResponse(BaseModel):
    session_id: str
    session_name: str
    created_at: datetime
    message_count: int

class ChatHistoryResponse(BaseModel):
    message_type: str
    message: str
    timestamp: datetime

class ChatbotAPI:
    def __init__(self):
        self.app = FastAPI(title="Agentic RAG Chatbot API", version="1.0.0")
        self.rag_bot = AgenticRAGBot()
        self.vector_store = ChromaVectorStore()
        self._configure_middleware()
        self._register_routes()
        create_tables()

    def _configure_middleware(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _register_routes(self):
        self.app.get("/")(self.root)
        self.app.post("/chat", response_model=ChatResponse)(self.chat)
        self.app.post("/feedback")(self.submit_feedback)
        self.app.get("/sessions", response_model=List[SessionResponse])(self.get_sessions)
        self.app.get("/sessions/{session_id}/history", response_model=List[ChatHistoryResponse])(self.get_chat_history)
        self.app.post("/sessions")(self.create_session)
        self.app.delete("/sessions/{session_id}")(self.delete_session)
        self.app.post("/documents")(self.add_document)
        self.app.get("/documents/count")(self.get_document_count)
        self.app.get("/evaluation/{transaction_id}")(self.get_evaluation_metrics)

    async def root(self):
        return {"message": "Agentic RAG Chatbot API is running!"}

    async def chat(self, request: ChatRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
        try:
            session = self._get_or_create_session(db, request.session_id)
            request.session_id = session.session_id

            result = self.rag_bot.process_query(request.message, request.session_id)
            self._store_chat_history(db, request.session_id, request.message, result["response"])

            background_tasks.add_task(
                self.evaluate_response_async,
                request.message,
                result["response"],
                result["retrieved_documents"],
                result["transaction_id"]
            )

            return ChatResponse(**result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def submit_feedback(self, request: FeedbackRequest, db: Session = Depends(get_db)):
        try:
            telemetry = db.query(TelemetryLog).filter(TelemetryLog.transaction_id == request.transaction_id).first()
            if not telemetry:
                raise HTTPException(status_code=404, detail="Transaction not found")

            feedback_log = FeedbackLog(
                transaction_id=request.transaction_id,
                query=telemetry.query,
                llm_response=telemetry.response,
                feedback=request.feedback,
                feedback_text=request.feedback_text
            )
            db.add(feedback_log)
            db.commit()
            return {"message": "Feedback submitted successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def get_sessions(self, db: Session = Depends(get_db)):
        try:
            sessions = db.query(ChatSession).order_by(ChatSession.updated_at.desc()).all()
            return [SessionResponse(
                session_id=s.session_id,
                session_name=s.session_name,
                created_at=s.created_at,
                message_count=db.query(ChatHistory).filter(ChatHistory.session_id == s.session_id).count()
            ) for s in sessions]
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def get_chat_history(self, session_id: str, db: Session = Depends(get_db)):
        try:
            history = db.query(ChatHistory).filter(ChatHistory.session_id == session_id).order_by(ChatHistory.timestamp.desc()).all()
            history.reverse()
            # Manually construct the response to avoid issues with SQLAlchemy's __dict__
            return [
                ChatHistoryResponse(
                    message_type=msg.message_type,
                    message=msg.message,
                    timestamp=msg.timestamp
                ) for msg in history
            ]
        except Exception as e:
            # It's good practice to log the actual error on the server
            print(f"Error fetching chat history for session {session_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def create_session(self, db: Session = Depends(get_db)):
        try:
            session_id = str(uuid.uuid4())
            session = ChatSession(session_id=session_id, session_name=f"New Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            db.add(session)
            db.commit()
            return {"session_id": session_id, "message": "Session created successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def delete_session(self, session_id: str, db: Session = Depends(get_db)):
        try:
            db.query(ChatHistory).filter(ChatHistory.session_id == session_id).delete()
            db.query(ChatSession).filter(ChatSession.session_id == session_id).delete()
            db.commit()
            return {"message": "Session deleted successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def add_document(self, request: DocumentRequest):
        try:
            doc_id = str(uuid.uuid4())
            self.vector_store.add_documents(documents=[request.content], metadatas=[request.metadata or {}], ids=[doc_id])
            return {"document_id": doc_id, "message": "Document added successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def get_document_count(self):
        try:
            return {"document_count": self.vector_store.get_collection_count()}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def get_evaluation_metrics(self, transaction_id: str, db: Session = Depends(get_db)):
        try:
            metrics = db.query(EvaluationMetrics).filter(EvaluationMetrics.transaction_id == transaction_id).first()
            if not metrics:
                raise HTTPException(status_code=404, detail="Evaluation metrics not found")
            return metrics
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def evaluate_response_async(self, query, response, retrieved_contexts, transaction_id):
        try:
            evaluate_response_sync(query=query, response=response, retrieved_contexts=retrieved_contexts, transaction_id=transaction_id)
        except Exception as e:
            print(f"Error in background evaluation: {e}")

    def _get_or_create_session(self, db: Session, session_id: Optional[str]) -> ChatSession:
        if session_id:
            session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
            return session
        else:
            new_session_id = str(uuid.uuid4())
            session = ChatSession(session_id=new_session_id, session_name=f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            db.add(session)
            db.commit()
            db.refresh(session)
            return session

    def _store_chat_history(self, db: Session, session_id: str, user_message: str, assistant_response: str):
        db.add(ChatHistory(session_id=session_id, message_type="user", message=user_message))
        db.add(ChatHistory(session_id=session_id, message_type="assistant", message=assistant_response))
        db.commit()

# Main application instance
chatbot_api = ChatbotAPI()
app = chatbot_api.app

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=True
    )