import os
from langchain.schema import Document
from langchain.document_loaders import (
    PyMuPDFLoader, CSVLoader, TextLoader,
    UnstructuredPowerPointLoader, UnstructuredWordDocumentLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

from mcp.mcp_message import MCPMessage
from mcp.message_bus import mcp_bus

AGENT_NAME = "IngestionAgent"

def load_file(file_path: str):
    ext = os.path.splitext(file_path)[-1].lower()
    loader = None

    if ext == ".pdf":
        loader = PyMuPDFLoader(file_path)
        return loader.load()
    elif ext == ".csv":
        loader = CSVLoader(file_path)
        docs = loader.load()

        # Merge all rows into one text block
        full_text = "\n\n".join([doc.page_content for doc in docs])
        return [Document(page_content=full_text, metadata={"source": file_path})]
    elif ext in [".txt", ".md"]:
        loader = TextLoader(file_path)
        return loader.load()
    elif ext == ".pptx":
        loader = UnstructuredPowerPointLoader(file_path)
        return loader.load()
    elif ext == ".docx":
        loader = UnstructuredWordDocumentLoader(file_path)
        return loader.load()
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

def ingestion_agent(message: MCPMessage) -> None:
    payload = message.payload
    file_paths = payload.get("file_paths", [])
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    all_docs = []
    for path in file_paths:
        try:
            docs = load_file(path)
            all_docs.extend(docs)
            print(f"✅ Loaded {len(docs)} docs from {os.path.basename(path)}")
        except Exception as e:
            print(f"❌ Failed to load {path}: {str(e)}")

    if not all_docs:
        print("⚠️ No documents loaded. Halting.")
        return

    # Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = splitter.split_documents(all_docs)

    if not chunks:
        print("⚠️ No chunks after splitting. Halting.")
        return

    # Build response payload
    result_payload = {
        "query": payload.get("query", ""),
        "chunks": [chunk.page_content for chunk in chunks],
        "metadata": [chunk.metadata for chunk in chunks],
    }

    # Send message to Retrieval Agent
    mcp_bus.dispatch(
        sender=AGENT_NAME,
        receiver="RetrievalAgent",
        msg_type="documents_parsed",
        payload=result_payload,
        trace_id=message.trace_id
    )

    print(f"📨 Sent {len(chunks)} chunks to RetrievalAgent [trace_id={message.trace_id}]")
