# app/database_knowledge.py
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()

KNOWLEDGE_DATABASE_URL = os.getenv(
    "KNOWLEDGE_DATABASE_URL", 
    "sqlite:///./medical_knowledge.db"
)

knowledge_engine = create_engine(
    KNOWLEDGE_DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in KNOWLEDGE_DATABASE_URL else {}
)

KnowledgeSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=knowledge_engine)

KnowledgeBase = declarative_base()

def get_knowledge_db():
    db = KnowledgeSessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_knowledge_db():
    print("Creating Medical Knowledge Database tables...")
    KnowledgeBase.metadata.create_all(bind=knowledge_engine)
    print("Medical Knowledge Database created successfully!")