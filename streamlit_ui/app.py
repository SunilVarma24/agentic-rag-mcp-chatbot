import os
import sys
import tempfile

import streamlit as st

# Dynamically add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import core components
from agents.ingestion_agent import ingestion_agent
from agents.retrieval_agent import retrieval_agent
from agents.llm_response_agent import llm_response_agent
from mcp.mcp_message import MCPMessage
from mcp.message_bus import mcp_bus

# Streamlit UI config
st.set_page_config(page_title="📄 Agentic RAG: Document QA", layout="centered")

st.title("📄 Multi-Format Document QA Chatbot")
st.caption("Powered by Agentic Workflow + LangChain + FAISS")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

tab1, tab2 = st.tabs(["💬 Chat Interface", "📊 Chat History"])

with tab1:
    st.subheader("💬 Chat with your documents")

    # Inputs
    question = st.text_input("💬 Ask a question about the uploaded document:", placeholder="e.g. What KPIs were tracked in Q1?")
    uploaded_files = st.file_uploader("📎 Upload one or more documents", type=["pdf", "docx", "csv", "txt", "pptx"], accept_multiple_files=True)

    submit = st.button("🚀 Submit")

    if submit:
        if not question or not uploaded_files:
            st.warning("⚠️ Please provide both a question and a document.")
        else:
            # Save uploaded files to temporary directory
            temp_file_paths = []
            temp_dir = tempfile.TemporaryDirectory()
            for file in uploaded_files:
                file_path = os.path.join(temp_dir.name, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getvalue())
                temp_file_paths.append(file_path)

            # Initialize MCPMessage
            initial_msg = MCPMessage(
                sender="User",
                receiver="IngestionAgent",
                msg_type="user_input",
                payload={
                    "query": question,
                    "file_paths": temp_file_paths
                }
            )

            # Reset message bus for clean run
            mcp_bus.reset()

            with st.spinner("🤖 Running..."):
                current_msg = initial_msg
                while current_msg.receiver != "User":                    
                    if current_msg.receiver == "IngestionAgent":
                        ingestion_agent(current_msg)
                    elif current_msg.receiver == "RetrievalAgent":
                        retrieval_agent(current_msg)
                    elif current_msg.receiver == "LLMResponseAgent":
                        llm_response_agent(current_msg)
                    elif current_msg.receiver == "User":
                        # this is our final message
                        break
                    else:
                        st.error(f"❌ Unknown receiver: {current_msg.receiver}")
                        break

                    # Get the next message from the bus
                    next_msg = None
                    # Check for messages to different agents in order of likely flow
                    for agent in ["RetrievalAgent", "LLMResponseAgent", "User"]:
                        next_msg = mcp_bus.collect(agent)
                        if next_msg:
                            break
                        
                    if next_msg is None:
                        st.error("❌ No message returned; pipeline seems stuck.")
                        break
                            
                    current_msg = next_msg
                        
                    # If it's the final response, break
                    if current_msg.msg_type in ["final_response", "response_error"]:
                        break

            # Display output
            if current_msg and current_msg.msg_type == "final_response":
                response_text = current_msg.payload.get("response", "No answer returned.")
                        
                st.subheader("🧠 Agent Answer")
                st.success(response_text)
                        
                # Update chat history
                st.session_state.chat_history.append({
                    "query": question,
                    "answer": response_text
                })
                        
                # Optional: show source chunks
                if "source_chunks" in current_msg.payload:
                    with st.expander("📄 Source Chunks Used"):
                        for i, chunk in enumerate(current_msg.payload["source_chunks"], 1):
                            chunk_text = chunk.get("text", str(chunk))
                            st.markdown(f"**Chunk {i}:**\n\n{chunk_text}\n")
                                    
            elif current_msg and current_msg.msg_type == "response_error":
                st.error(f"❌ Error during response generation:\n\n{current_msg.payload.get('error', 'Unknown error')}")
            else:
                st.error("❌ No response received from agents.")

with tab2:
    # Display chat history
    if st.session_state.chat_history:
        st.subheader("💬 Chat History")
        for i, entry in enumerate(st.session_state.chat_history):
            # Handle both 'query' and 'user' keys for backward compatibility
            question_text = entry.get('query', entry.get('user', 'Unknown question'))
            st.markdown(f"**Q{i+1}:** {question_text}")
            if 'answer' in entry:
                st.markdown(f"**A{i+1}:** {entry['answer']}")
            st.markdown("---")

    # Add a reset button
    if st.button("🔄 Reset Chat History"):
        st.session_state.chat_history = []
        mcp_bus.reset()
        st.rerun()
