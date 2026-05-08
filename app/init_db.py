from app.core.database_user      import engine as user_engine
from app.core.database_knowledge import knowledge_engine

from app.models.models_user      import Base as UserBase
from app.models.models_knowledge import KnowledgeBase


def init():
    UserBase.metadata.create_all(bind=user_engine)

    KnowledgeBase.metadata.create_all(bind=knowledge_engine)

    print("Done.")

if __name__ == "__main__":
    init()