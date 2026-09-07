"""
Hybrid Retriever Module for CardWise AI
Combines BM25 Keyword Search and FAISS Dense Vector Search using HuggingFace Embeddings
with Reciprocal Rank Fusion (RRF).
"""

from typing import List, Dict, Any, Optional
from langchain_core.documents import Document

try:
    from langchain_community.retrievers import BM25Retriever
except ImportError:
    BM25Retriever = None

try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    FAISS = None

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        HuggingFaceEmbeddings = None

from data.card_benefits_data import CARD_BENEFITS, get_all_benefits


class CardWiseHybridRetriever:
    """
    Hybrid retriever combining BM25 sparse search and FAISS vector search
    with HuggingFace 'sentence-transformers/all-MiniLM-L6-v2' embeddings.
    Uses Reciprocal Rank Fusion (RRF) to merge and rank results.
    """

    def __init__(
        self,
        benefits_data: Optional[List[Dict[str, Any]]] = None,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        bm25_weight: float = 0.5,
        faiss_weight: float = 0.5,
        rrf_k: int = 60
    ):
        self.benefits_data = benefits_data or get_all_benefits()
        self.embedding_model_name = embedding_model_name
        self.bm25_weight = bm25_weight
        self.faiss_weight = faiss_weight
        self.rrf_k = rrf_k

        self.documents = self._prepare_documents()
        self.bm25_retriever = None
        self.faiss_vectorstore = None
        self.embeddings = None

        self._initialize_retrievers()

    def _prepare_documents(self) -> List[Document]:
        """Converts benefit dictionaries into LangChain Document objects with enriched search context."""
        docs = []
        for benefit in self.benefits_data:
            keywords_str = ", ".join(benefit.get("keywords", []))
            text_content = (
                f"Card: {benefit['card']}\n"
                f"Topic: {benefit['topic']}\n"
                f"Category: {benefit['category']}\n"
                f"Details: {benefit['content']}\n"
                f"Keywords: {keywords_str}"
            )
            metadata = {
                "id": benefit["id"],
                "card": benefit["card"],
                "card_code": benefit["card_code"],
                "topic": benefit["topic"],
                "category": benefit["category"],
                "annual_credit_value": benefit.get("annual_credit_value", 0)
            }
            docs.append(Document(page_content=text_content, metadata=metadata))
        return docs

    def _initialize_retrievers(self):
        """Initializes BM25 and FAISS retrievers with error fallbacks."""
        # 1. BM25 Sparse Retriever
        if BM25Retriever:
            try:
                self.bm25_retriever = BM25Retriever.from_documents(self.documents)
            except Exception as e:
                print(f"Notice: BM25Retriever initialization notice: {e}")

        # 2. FAISS Dense Vector Store
        if FAISS and HuggingFaceEmbeddings:
            try:
                self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
                self.faiss_vectorstore = FAISS.from_documents(self.documents, self.embeddings)
            except Exception as e:
                print(f"Notice: FAISS/Embeddings initialization notice: {e}")

    def _reciprocal_rank_fusion(
        self,
        doc_lists: List[List[Document]],
        weights: List[float]
    ) -> List[Document]:
        """Performs Reciprocal Rank Fusion (RRF) across multiple ranked result lists."""
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        for doc_list, weight in zip(doc_lists, weights):
            for rank, doc in enumerate(doc_list, start=1):
                doc_id = doc.metadata.get("id", doc.page_content[:50])
                doc_map[doc_id] = doc
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = 0.0
                rrf_scores[doc_id] += weight * (1.0 / (self.rrf_k + rank))

        # Sort documents by accumulated RRF score descending
        sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        return [doc_map[doc_id] for doc_id in sorted_doc_ids]

    def _fallback_keyword_search(self, query: str, top_k: int = 4) -> List[Document]:
        """Simple keyword matching fallback if indexers are unavailable."""
        query_words = set(query.lower().split())
        scored_docs = []
        for doc in self.documents:
            content_lower = doc.page_content.lower()
            score = sum(1 for word in query_words if word in content_lower)
            scored_docs.append((score, doc))
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:top_k]]

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        card_filter: Optional[str] = None
    ) -> List[Document]:
        """
        Retrieves top relevant benefit documents using BM25 + FAISS + RRF.
        Optionally filters by card_code ('amex_plat' or 'csr').
        """
        retrieved_lists = []
        weights = []

        # 1. BM25 Retrieval
        if self.bm25_retriever:
            try:
                self.bm25_retriever.k = top_k * 2
                bm25_docs = self.bm25_retriever.invoke(query)
                retrieved_lists.append(bm25_docs)
                weights.append(self.bm25_weight)
            except Exception:
                pass

        # 2. FAISS Retrieval
        if self.faiss_vectorstore:
            try:
                faiss_docs = self.faiss_vectorstore.similarity_search(query, k=top_k * 2)
                retrieved_lists.append(faiss_docs)
                weights.append(self.faiss_weight)
            except Exception:
                pass

        # If both retrievers produced results, fuse with RRF
        if retrieved_lists:
            fused_docs = self._reciprocal_rank_fusion(retrieved_lists, weights)
        else:
            fused_docs = self._fallback_keyword_search(query, top_k=top_k * 2)

        # Apply card filter if specified
        if card_filter:
            fused_docs = [
                doc for doc in fused_docs
                if doc.metadata.get("card_code") == card_filter
            ]

        return fused_docs[:top_k]

    def format_context_for_prompt(self, docs: List[Document]) -> str:
        """Formats retrieved documents into a clean context string for the LLM prompt."""
        formatted_chunks = []
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            chunk_str = (
                f"--- Document [{i}] ---\n"
                f"Card: {meta.get('card')}\n"
                f"Topic: {meta.get('topic')}\n"
                f"Category: {meta.get('category')}\n"
                f"Annual Credit Value: ${meta.get('annual_credit_value', 0)}\n"
                f"Policy Details:\n{doc.page_content}\n"
            )
            formatted_chunks.append(chunk_str)

        return "\n".join(formatted_chunks)
