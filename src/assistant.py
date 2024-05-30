from typing import Optional
from phi.assistant import Assistant
from phi.knowledge import AssistantKnowledge
from phi.llm.ollama import Ollama
from phi.embedder.ollama import OllamaEmbedder
from phi.vectordb.pgvector import PgVector2
from phi.storage.assistant.postgres import PgAssistantStorage

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

def get_rag_assistant(
    llm_model: str = "llama3",
    embeddings_model: str = "nomic-embed-text",
    user_id: Optional[str] = None,
    run_id: Optional[str] = None,
    debug_mode: bool = True,
) -> Assistant:
    """Get a Local RAG Assistant."""

    # Define the embedder based on the embeddings model
    embedder = OllamaEmbedder(model=embeddings_model, dimensions=4096)
    embeddings_model_clean = embeddings_model.replace("-", "_")
    if embeddings_model == "nomic-embed-text":
        embedder = OllamaEmbedder(model=embeddings_model, dimensions=768)
    elif embeddings_model == "phi3":
        embedder = OllamaEmbedder(model=embeddings_model, dimensions=3072)
    knowledge = AssistantKnowledge(
        vector_db=PgVector2(
            db_url=db_url,
            collection=f"local_rag_documents_{embeddings_model_clean}",
            embedder=embedder,
        ),
        num_documents=3,
    )

    return Assistant(
        name="local_rag_assistant",
        run_id=run_id,
        user_id=user_id,
        llm=Ollama(model=llm_model),
        storage=PgAssistantStorage(table_name="local_rag_assistant", db_url=db_url),
        knowledge_base=knowledge,
        description="You are an AI called 'Cappy RAG' and your task is to answer questions using the provided information",
        instructions=[
            "When a user asks a question, you will be provided with information about the question.",
            "Carefully read this information and provide a clear and concise answer to the user.",
            "Do not use phrases like 'based on my knowledge' or 'depending on the information'.",
        ],
        add_references_to_prompt=True,
        markdown=True,
        add_datetime_to_instructions=True,
        debug_mode=debug_mode,
    )
