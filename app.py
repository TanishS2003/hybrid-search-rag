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

st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; font-weight: bold; }
    .search-result { 
        padding: 1.5rem; 
        border-radius: 10px; 
        background-color: #ffffff; 
        border-left: 5px solid #4CAF50; 
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: #1f1f1f !important;
    }
    .search-result strong { color: #2e7d32 !important; }
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
    
    pinecone_api_key = st.secrets.get("PINECONE_API_KEY", "")
    
    if pinecone_api_key:
        st.success("✅ API Key Detected")
    else:
        st.error("❌ Missing PINECONE_API_KEY in Secrets")

    index_name = st.text_input("Pinecone Index Name", value="hybrid-rag-demo")
    
    if st.button("🚀 Initialize Hybrid Engine", type="primary"):
        if not pinecone_api_key:
            st.error("Cannot initialize without API Key.")
        else:
            with st.spinner("Initializing Pinecone and Encoders..."):
                try:
                    pc = Pinecone(api_key=pinecone_api_key)
                    
                    # Ensure Dot Product for Hybrid Search
                    existing_indexes = [idx.name for idx in pc.list_indexes()]
                    if index_name not in existing_indexes:
                        pc.create_index(
                            name=index_name,
                            dimension=384, 
                            metric="dotproduct",
                            spec=ServerlessSpec(cloud="aws", region="us-east-1")
                        )
                        time.sleep(5)
                    
                    index = pc.Index(index_name)
                    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                    
                    # Initialize BM25 with a default to prevent empty vector errors
                    bm25_encoder = BM25Encoder().default()
                    
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

# --- Main Application UI ---
st.title("🔍 Hybrid Search RAG Engine")
st.markdown("---")

if not st.session_state.initialized:
    st.warning("⚠️ Please initialize the engine from the sidebar to start.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("""
        **What is Hybrid Search?**
        It combines **Semantic Search** (meaning) with **Keyword Search** (exact matching).
        """)
    with col2:
        # FIXED: Added content here to resolve IndentationError at line 111
        st.markdown("### 🏗️ Architecture")
        st.write("Using Reciprocal Rank Fusion to merge Dense and Sparse results.")
        

else:
    tab1, tab2 = st.tabs(["📄 Document Management", "🔎 Search Interface"])

    # --- TAB 1: DATA INGESTION ---
    with tab1:
        st.header("Add Data to Index")
        
        with st.expander("✍️ Add Single Document (Resume, Text, etc.)", expanded=True):
            user_text = st.text_area("Content:", height=300)
            if st.button("Index Document"):
                if user_text:
                    with st.spinner("Updating BM25 and Indexing..."):
                        # Refit BM25 on the updated collection to avoid 'at least one value' error
                        st.session_state.documents.append(user_text)
                        st.session_state.bm25_encoder.fit(st.session_state.documents)
                        st.session_state.retriever.sparse_encoder = st.session_state.bm25_encoder
                        
                        st.session_state.retriever.add_texts([user_text])
                        st.success("Document added!")
                        time.sleep(1)
                        st.rerun()

        if st.button("📚 Load Sample Knowledge Base"):
            samples = ["Python programming for data science.", "AI in autonomous vehicles."]
            for s in samples:
                if s not in st.session_state.documents:
                    st.session_state.documents.append(s)
            st.session_state.bm25_encoder.fit(st.session_state.documents)
            st.session_state.retriever.sparse_encoder = st.session_state.bm25_encoder
            st.session_state.retriever.add_texts(samples)
            st.rerun()

    # --- TAB 2: SEARCH ---
    with tab2:
        st.header("Search Knowledge Base")
        query = st.text_input("Enter query:")
        
        # Alpha controls the balance between Keyword (0.0) and Semantic (1.0)
        alpha = st.slider("Alpha (Keyword vs Semantic)", 0.0, 1.0, 0.5)
        
        if st.button("🔍 Execute Search", type="primary"):
            if query and st.session_state.documents:
                st.session_state.retriever.alpha = alpha
                results = st.session_state.retriever.invoke(query)
                
                st.subheader(f"Top Matches:")
                for i, doc in enumerate(results):
                    st.markdown(f"""
                    <div class="search-result">
                        <strong>Rank #{i+1}</strong><br>
                        <p>{doc.page_content}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("Please index text first and enter a query.")

st.markdown("---")
st.caption("Hybrid RAG App | Built with LangChain and Pinecone")
