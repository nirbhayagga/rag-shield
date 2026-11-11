"""LangChain retriever: pip install "rag-shield[langchain]"

    from ragshield.integrations.langchain import ShieldRetriever
    retriever = ShieldRetriever(shield=shield, k=5)   # use it anywhere LangChain takes a retriever
    docs = retriever.invoke('How do contractors get VPN access?')
"""
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from . import trust_metadata


class ShieldRetriever(BaseRetriever):
    """Retrieves through RAG Shield: tampered, unknown and quarantined chunks never come back, and the rest carry their trust decision."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    shield: Any
    k: int = 5

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        return [Document(id=hit['chunk_id'], page_content=hit['text'], metadata=trust_metadata(hit)) for hit in self.shield.retrieve(query, self.k)]
