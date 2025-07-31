import os
from typing import List, Dict, Any
from langchain.schema import Document

from mcp.mcp_message import MCPMessage
from mcp.message_bus import mcp_bus

from vector_store.faiss_store import FAISSVectorStore

AGENT_NAME = "RetrievalAgent"

def retrieval_agent(message: MCPMessage) -> None:
    """
    Handles document retrieval after ingestion is complete.
    Looks for top matching documents based on user query.
    Sends result to LLMResponseAgent.
    """
    if message.msg_type != "documents_parsed":
        raise ValueError(f"{AGENT_NAME} only handles 'documents_parsed' messages.")

    payload = message.payload
    query = payload.get("query", "")
    if not query:
        print("❌ No query provided. Cannot perform retrieval.")
        return
    
    chunks = payload.get("chunks", [])
    metadata = payload.get("metadata", [])

    # Convert to Document objects
    documents = []
    for i, chunk in enumerate(chunks):
        doc_metadata = metadata[i] if i < len(metadata) else {}
        documents.append(Document(page_content=chunk, metadata=doc_metadata))

    # Vector Store
    vector_store = FAISSVectorStore()
    vector_store.build_index(documents)

    retriever = vector_store.as_retriever(k=3)
    
    print(f"🔍 {AGENT_NAME}: Performing retrieval for query: '{query}'")

    # Get relevant documents
    retrieved_docs: List[Document] = retriever.invoke(query)
    print(f"✅ Retrieved {len(retrieved_docs)} documents")

    # Prepare clean output
    retrieved_payload: List[Dict[str, Any]] = []
    for doc in retrieved_docs:
        retrieved_payload.append({
            "text": doc.page_content,
            "metadata": doc.metadata
        })

    chat_history = mcp_bus.get_chat_history(message.trace_id)

    # Dispatch message to LLMResponseAgent
    mcp_bus.dispatch(
        sender=AGENT_NAME,
        receiver="LLMResponseAgent",
        msg_type="chunks_retrieved",
        payload={
            "query": query,
            "retrieved_docs": retrieved_payload,
            "total_chunks": len(retrieved_payload),
            "chat_history": chat_history
        },
        trace_id=message.trace_id
    )
