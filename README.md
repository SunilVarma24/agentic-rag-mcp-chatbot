# Agentic RAG Chatbot for Multi-Format Document QA using MCP

## Project Overview  
An **agentic Retrieval-Augmented Generation (RAG) chatbot** that answers user questions over diverse document formats (PDF, PPTX, CSV, DOCX, TXT/Markdown). Agents communicate via a **Model Context Protocol (MCP)** message bus. Built with **LangChain**, **FAISS**, and **Streamlit**, it parses, retrieves, and generates answers with source context.

## Introduction  
Multi-format document QA poses challenges in parsing, indexing and retrieving across heterogeneous files. This project uses an **agentic architecture**—ingestion, retrieval, and response agents that exchange **MCPMessage** objects over an in-memory bus. Each agent fulfills a discrete responsibility, enabling clear routing, traceability, and extensibility.

## How It Works  

1. **User Input**  
   - Upload one or more documents (PDF, PPTX, CSV, DOCX, TXT/MD) and submit a question via Streamlit UI.  

2. **IngestionAgent**  
   - Parses files using specialized loaders  
   - Splits text into chunks  
   - Dispatches a `documents_parsed` MCPMessage to `RetrievalAgent`.

3. **RetrievalAgent**  
   - Embeds chunks via **HuggingFaceEmbeddings** (`bge-small-en-v1.5`) & builds a **FAISS** index  
   - Retrieves top-k relevant chunks for the query  
   - Dispatches a `chunks_retrieved` MCPMessage to `LLMResponseAgent`.

4. **LLMResponseAgent**  
   - Formats a prompt with retrieved context + optional chat history  
   - Invokes **gpt-4o-mini** to generate an answer  
   - Dispatches a `final_response` MCPMessage back to `User`.

5. **Streamlit UI**  
   - Displays answer and “Source Chunks Used”  
   - Maintains multi-turn chat history  

Throughout, all messages carry a **trace_id** for end-to-end logging and history retrieval.

## MCP Communication
Agents pass structured messages using a custom Model Context Protocol:

```json
{
  "sender": "RetrievalAgent",
  "receiver": "LLMResponseAgent",
  "type": "chunks_retrieved",
  "trace_id": "rag-457",
  "payload": {
    "query": "What are the KPIs?",
    "retrieved_docs": [...],
    "chat_history": [...]
  }
}
```

## Features
- 🔄 Multi-Format Ingestion: PDF, PPTX, CSV, DOCX, TXT/Markdown
- 🤖 Agentic Pipeline: Ingestion → Retrieval → Response
- 📡 Model Context Protocol (MCP): Structured message passing with trace_id
- 📚 FAISS + Embeddings: bge-small-en-v1.5 for chunk indexing
- 💬 gpt-4o-mini: High-quality LLM responses
- 🌐 Traceable History: Retrieve past Q&A via MCPBus logs
- 📊 Streamlit UI: Multi-turn chat, source context, and history

## Project Structure
```
agentic-rag-mcp-chatbot/
│
├── agents/
│   ├── ingestion_agent.py       # Parses & chunks documents
│   ├── retrieval_agent.py       # Embeds & retrieves chunks
│   └── llm_response_agent.py    # Formats prompt & calls LLM
│
├── mcp/
│   ├── mcp_message.py           # MCPMessage class
│   └── message_bus.py           # MCPBus implementation
│
├── vector_store/
│   └── faiss_store.py           # FAISS wrapper with HF embeddings
│
├── streamlit_ui/
│   └── app.py                   # Streamlit UI & coordinator loop
│
├── requirements.txt             # Python dependencies
├── .env                         # API keys
└── README.md                    # This file
```

## Installation  

1. **Clone the repo**  
   ```bash
   git clone https://github.com/SunilVarma24/agentic-rag-mcp-chatbot.git
   cd agentic-rag-mcp-chatbot
    ```

2. **Create & activate a virtual environment**
    ```bash
    python3 -m venv venv
    source venv/bin/activate    # macOS/Linux
    venv\Scripts\activate       # Windows
    ```

3. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4. **Configure environment**
    - Create a .env file:
        ```bash
        OPENAI_API_KEY=your_openai_key
        ```

## Usage
1. **Run the Streamlit app**

```bash
streamlit run streamlit_ui/app.py
```

2. **In the browser**

- Go to `http://localhost:8501` if not opened
- Switch to Chat Interface
- Upload documents and enter your question
- Click Submit
- View answer + expandable “Source Chunks”
- Check your Chat History tab for past Q&A

## Conclusion
This Agentic RAG framework demonstrates a modular, traceable approach to document-based QA over heterogeneous formats. By leveraging MCP for inter-agent communication, it achieves clear routing, logging, and extensibility—ideal for production-grade RAG systems.

