import streamlit as st
import os
import json
import time
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import PineconeHybridSearchRetriever

# --- Page Configuration ---
st.set_page_config(
    page_title="Hybrid Search RAG", 
    page_icon="🔍", 
    layout="wide"
)

# Custom CSS for a clean UI
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; font-weight: bold; }
    .search-result { 
        padding: 1.5rem; 
        border-radius: 10px; 
        background-color: #f8f9fa; 
        border-left: 5px solid #4CAF50; 
        margin-bottom: 1.2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .status-box { padding: 1rem; border-radius: 5px; margin-bottom: 1rem; }
    </style>
    """, unsafe_allow_html=True)

# --- Session State Initialization ---
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ System Setup")
    
    # API Key Handling
    pinecone_api_key = st.secrets.get("PINECONE_API_KEY", "")
    
    if pinecone_api_key:
        st.success("✅ API Key Detected")
    else:
        st.error("❌ Missing PINECONE_API_KEY in Secrets")
        st.info("Add it to .streamlit/secrets.toml or Streamlit Cloud Secrets.")

    index_name = st.text_input("Pinecone Index Name", value="hybrid-rag-demo")
    
    if st.button("🚀 Initialize Hybrid Engine", type="primary"):
        if not pinecone_api_key:
            st.error("Cannot initialize without API Key.")
        else:
            with st.spinner("Waking up the vectors..."):
                try:
                    # 1. Initialize Pinecone
                    pc = Pinecone(api_key=pinecone_api_key)
                    
                    # 2. Create Index if needed (Dot Product is mandatory for Hybrid)
                    existing_indexes = [idx.name for idx in pc.list_indexes()]
                    if index_name not in existing_indexes:
                        pc.create_index(
                            name=index_name,
                            dimension=384, # Dimensions for all-MiniLM-L6-v2
                            metric="dotproduct",
                            spec=ServerlessSpec(cloud="aws", region="us-east-1")
                        )
                        time.sleep(5) # Wait for cloud propagation
                    
                    index = pc.Index(index_name)
                    
                    # 3. Dense Embeddings (HuggingFace)
                    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                    
                    # 4. Sparse Encoder (BM25)
                    # We start with a default state to avoid immediate errors
                    bm25_encoder = BM25Encoder().default()
                    
                    # 5. Hybrid Retriever Initialization
                    retriever = PineconeHybridSearchRetriever(
                        embeddings=embeddings,
                        sparse_encoder=bm25_encoder,
                        index=index
                    )
                    
                    st.session_state.retriever = retriever
                    st.session_state.bm25_encoder = bm25_encoder
                    st.session_state.initialized = True
                    st.success("Engine Online!")
                except Exception as e:
                    st.error(f"Initialization Failed: {e}")

# --- Main Application Logic ---
st.title("🔍 Hybrid Search RAG Engine")
st.markdown("---")

if not st.session_state.initialized:
    st.warning("⚠️ Action Required: Please initialize the engine from the sidebar to start indexing data.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("""
        **What is Hybrid Search?**
        It combines **Semantic Search** (understanding meaning) with **Keyword Search** (exact word matching) to give the most accurate results possible.
        """)
    with col2:
        # Adding a placeholder or 'pass' to fix the IndentationError
        st.markdown("### 🏗️ Architecture")
        st.write("Hybrid RAG uses Dense Vectors for context and Sparse Vectors for keywords.")

else:
    tab1, tab2 = st.tabs(["📄 Document Management", "🔎 Search Interface"])

    # --- TAB 1: DATA INGESTION ---
    with tab1:
        st.header("Add Data to Index")
        
        # Option A: Manual Entry
        with st.expander("✍️ Add Single Document", expanded=True):
            user_text = st.text_area("Content:", placeholder="Paste text you want to index...")
            if st.button("Index Document"):
                if user_text:
                    with st.spinner("Updating BM25 vocabulary and uploading..."):
                        # Add to local tracking
                        st.session_state.documents.append(user_text)
                        # Re-fit BM25 on the entire updated corpus
                        st.session_state.bm25_encoder.fit(st.session_state.documents)
                        # Sync retriever with the new encoder state
                        st.session_state.retriever.sparse_encoder = st.session_state.bm25_encoder
                        # Upload to Pinecone
                        st.session_state.retriever.add_texts([user_text])
                        st.success("Document added!")
                        time.sleep(1)
                        st.rerun()

        # Option B: Bulk Samples
        st.markdown("---")
        st.subheader("Bulk Operations")
        if st.button("📚 Load Sample Knowledge Base"):
            samples = [
                "The James Webb Space Telescope is the most powerful telescope ever built.",
                "Photosynthesis is the process used by plants to convert light into energy.",
                "Python is a high-level, interpreted programming language known for readability.",
                "Quantum computing uses qubits to perform complex calculations faster than classical PCs.",
                "The Great Barrier Reef is the world's largest coral reef system."
            ]
            with st.spinner("Bulk indexing samples..."):
                # Avoid duplicates in local list
                for s in samples:
                    if s not in st.session_state.documents:
                        st.session_state.documents.append(s)
                
                # Fit and Sync
                st.session_state.bm25_encoder.fit(st.session_state.documents)
                st.session_state.retriever.sparse_encoder = st.session_state.bm25_encoder
                # Upload
                st.session_state.retriever.add_texts(samples)
                st.success(f"Successfully indexed {len(samples)} sample docs!")
                time.sleep(1)
                st.rerun()

        # Display Current Stats
        st.markdown("---")
        st.metric("Documents in local context", len(st.session_state.documents))
        if st.button("🗑️ Clear Local Document History"):
            st.session_state.documents = []
            st.rerun()

    # --- TAB 2: SEARCH ---
    with tab2:
        st.header("Search the Knowledge Base")
        
        query = st.text_input("Enter your query:", placeholder="e.g., 'How do plants make energy?'")
        
        col_a, col_b = st.columns([1, 1])
        with col_a:
            alpha = st.slider(
                "Search Balance (Alpha)", 
                0.0, 1.0, 0.5, 
                help="0.0 = Keyword only, 1.0 = Semantic only, 0.5 = Balanced"
            )
        with col_b:
            top_k = st.number_input("Results to return:", min_value=1, max_value=10, value=3)

        if st.button("🔍 Execute Hybrid Search", type="primary"):
            if query:
                if len(st.session_state.documents) == 0:
                    st.error("The index is empty! Please add documents in the first tab.")
                else:
                    with st.spinner("Calculating Reciprocal Rank Fusion..."):
                        # Update retriever settings
                        st.session_state.retriever.alpha = alpha
                        
                        # Search
                        results = st.session_state.retriever.invoke(query)
                        
                        st.subheader(f"Top {len(results)} Matches:")
                        for i, doc in enumerate(results[:top_k]):
                            st.markdown(f"""
                            <div class="search-result">
                                <strong>Rank #{i+1}</strong><br>
                                <p style="margin-top:0.5rem;">{doc.page_content}</p>
                            </div>
                            """, unsafe_allow_html=True)
            else:
                st.warning("Please enter a query first.")

# Footer
st.markdown("---")
st.caption("Hybrid RAG App | Built with LangChain, Pinecone, and HuggingFace")
