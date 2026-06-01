import uuid
from typing import List, Dict, Any

import httpx
import chromadb
import openai


class _GatewayEmbeddingFunction:
    def __init__(self, api_key: str, model: str, base_url: str):
        self._client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=httpx.Client(verify=False),
        )
        self._model = model

    def name(self) -> str:
        return "gateway_openai"

    def __call__(self, input: List[str]) -> List[List[float]]:
        response = self._client.embeddings.create(input=input, model=self._model)
        return [item.embedding for item in response.data]


class VectorStore:
    """ChromaDB-backed vector store with per-session collection isolation."""

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        openai_api_key: str = "",
        embedding_model: str = "text-embedding-3-small",
        openai_base_url: str = "https://keygateway.arshnivlabs.com/v1",
    ):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self._embedding_fn = _GatewayEmbeddingFunction(
            api_key=openai_api_key,
            model=embedding_model,
            base_url=openai_base_url,
        )

    # ------------------------------------------------------------------
    def _collection_name(self, session_id: str) -> str:
        return f"session_{session_id.replace('-', '_')}"

    def _get_or_create(self, session_id: str) -> chromadb.Collection:
        return self.client.get_or_create_collection(
            name=self._collection_name(session_id),
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    def add_documents(
        self,
        session_id: str,
        chunks: List[str],
        metadata: Dict[str, Any] | None = None,
    ) -> int:
        if not chunks:
            return 0
        collection = self._get_or_create(session_id)
        meta = metadata or {}
        ids = [str(uuid.uuid4()) for _ in chunks]
        metas = [dict(meta) for _ in chunks]

        batch_size = 50
        for i in range(0, len(chunks), batch_size):
            collection.add(
                documents=chunks[i: i + batch_size],
                ids=ids[i: i + batch_size],
                metadatas=metas[i: i + batch_size],
            )
        return len(chunks)

    def query(
        self,
        session_id: str,
        query_text: str,
        n_results: int = 6,
    ) -> List[str]:
        collection = self._get_or_create(session_id)
        count = collection.count()
        if count == 0:
            return []
        try:
            results = collection.query(
                query_texts=[query_text],
                n_results=min(n_results, count),
            )
            return results["documents"][0] if results["documents"] else []
        except Exception:
            return []

    def get_all_documents(self, session_id: str, limit: int = 40) -> List[str]:
        collection = self._get_or_create(session_id)
        count = collection.count()
        if count == 0:
            return []
        results = collection.get(limit=min(count, limit))
        return results["documents"] or []

    def delete_session(self, session_id: str) -> None:
        try:
            self.client.delete_collection(self._collection_name(session_id))
        except Exception:
            pass
