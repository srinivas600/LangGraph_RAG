
from sqlalchemy import create_engine, Column, String, Text, DateTime, Integer, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import uuid

Base = declarative_base()

class TelemetryLog(Base):
    __tablename__ = "telemetry_logs"

    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(String, unique=True, index=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, index=True)
    query = Column(Text)
    retrieved_documents = Column(JSON)  # Top 5 retrieved documents
    complete_prompt = Column(Text)
    response = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

class FeedbackLog(Base):
    __tablename__ = "feedback_logs"

    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(String, index=True)  # References TelemetryLog.transaction_id
    query = Column(Text)
    llm_response = Column(Text)
    feedback = Column(String)  # positive/negative or rating
    feedback_text = Column(Text)  # optional feedback text
    timestamp = Column(DateTime, default=datetime.utcnow)

class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True, default=lambda: str(uuid.uuid4()))
    session_name = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    message_type = Column(String)  # 'user' or 'assistant'
    message = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

class EvaluationMetrics(Base):
    __tablename__ = "evaluation_metrics"

    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(String, index=True)
    answer_relevancy = Column(Float)
    faithfulness = Column(Float)
    context_precision = Column(Float)
    context_recall = Column(Float)
    context_relevancy = Column(Float)
    hallucination_score = Column(Float)
    bleu_score = Column(Float)
    rouge_score = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

# Database connection
from settings import Config

engine = create_engine(Config.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_tables():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

create_tables()