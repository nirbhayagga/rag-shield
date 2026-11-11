"""LlamaIndex retriever: pip install "rag-shield[llamaindex]"

    from ragshield.integrations.llamaindex import ShieldRetriever
    retriever = ShieldRetriever(shield, k=5)   # e.g. RetrieverQueryEngine.from_args(retriever)
    nodes = retriever.retrieve('How do contractors get VPN access?')
"""
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from . import trust_metadata


class ShieldRetriever(BaseRetriever):
    """Retrieves through RAG Shield: tampered, unknown and quarantined chunks never come back, and the rest carry their trust decision."""

    def __init__(self, shield, k=5, **kwargs):
        self.shield, self.k = shield, k
        super().__init__(**kwargs)

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        nodes = []
        for hit in self.shield.retrieve(query_bundle.query_str, self.k):
            metadata = trust_metadata(hit)
            # Keep the trust fields out of the embedding and the prompt text; the application decides how to show them.
            node = TextNode(id_=hit['chunk_id'], text=hit['text'], metadata=metadata,
                            excluded_embed_metadata_keys=list(metadata), excluded_llm_metadata_keys=list(metadata))
            nodes.append(NodeWithScore(node=node, score=hit['final_score']))
        return nodes
