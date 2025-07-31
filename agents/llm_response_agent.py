import os
from typing import List, Dict
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

from mcp.mcp_message import MCPMessage
from mcp.message_bus import mcp_bus

# Load environment variables
load_dotenv()

AGENT_NAME = "LLMResponseAgent"

# Prompt template
QA_TEMPLATE = """You are a helpful assistant. Use the following retrieved context to answer the user's question.

Context:
{context}

Question:
{question}

Helpful Answer:"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=QA_TEMPLATE
)

# Initialize ChatOpenAI
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")


def llm_response_agent(message: MCPMessage) -> None:
    """
    Handles 'chunks_retrieved' messages.
    Uses LLM to generate response from context, then dispatches result to User.
    """
    if message.msg_type != "chunks_retrieved":
        raise ValueError(f"{AGENT_NAME} only handles 'chunks_retrieved' messages.")

    payload = message.payload
    query = payload.get("query")
    docs_data = payload.get("retrieved_docs")

    if not query or not docs_data:
        print(f"❌ {AGENT_NAME}: Missing query or retrieved_docs in payload.")
        return

    print(f"🤖 {AGENT_NAME}: Generating answer for: '{query}'")

    # Convert to Document objects
    try:
        retrieved_docs: List[Document] = []
        for doc_data in docs_data:
            retrieved_docs.append(Document(
                page_content=doc_data["text"],
                metadata=doc_data.get("metadata", {})
            ))
    except Exception as e:
        print(f"❌ {AGENT_NAME}: Failed to convert docs: {e}")
        return
    
    chat_history = payload.get("chat_history", [])
    history_text = "\n".join([f"Q: {turn['query']}\nA: {turn['answer']}" for turn in chat_history])

    # Build context
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

    final_prompt = prompt.format(
        context=context_text,
        question=f"{history_text}\nFollow-up: {query}" if history_text else query
    )

    try:
        response = llm.invoke(final_prompt)
        answer = response.content.strip()

        print(f"✅ {AGENT_NAME}: Response generated successfully.")

        source_chunks = [
            {
                "text": doc.page_content[:300],
                "metadata": doc.metadata
            } for doc in retrieved_docs
        ]

        # Dispatch message back to user
        mcp_bus.dispatch(
            sender=AGENT_NAME,
            receiver="User",
            msg_type="final_response",
            payload={
                "response": answer,
                "query": query,
                "context_used": len(retrieved_docs),
                "source_chunks": source_chunks
            },
            trace_id=message.trace_id
        )

    except Exception as e:
        print(f"❌ {AGENT_NAME}: Error during LLM call: {e}")
        mcp_bus.dispatch(
            sender=AGENT_NAME,
            receiver="User",
            msg_type="response_error",
            payload={
                "error": str(e),
                "query": query,
                "context_used": len(retrieved_docs)
            },
            trace_id=message.trace_id
        )
