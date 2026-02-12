import streamlit as st
import os
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain.schema import Document
import time

# Page configuration
st.set_page_config(
    page_title="Hybrid Search RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .search-result {
        padding: 1rem;
        border-radius: 5px;
        background-color: #f0f2f6;
        border-left: 4px solid #4CAF50;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'pinecone_client' not in st.session_state:
    st.session_state.pinecone_client = None
if 'index' not in st.session_state:
    st.session_state.index = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

# Title
st.title("🔍 Hybrid Search RAG Application")
st.markdown("**Combining Semantic Search + Keyword Search with Pinecone**")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Get API key from secrets
    try:
        pinecone_api_key = st.secrets.get("PINECONE_API_KEY", None)
        if pinecone_api_key:
            st.success("✅ Pinecone API key loaded from secrets")
        else:
            st.error("❌ Pinecone API key not found in secrets")
            st.info("Please add PINECONE_API_KEY to your Streamlit secrets")
    except Exception as e:
        st.error("❌ Error loading secrets")
        pinecone_api_key = None
    
    # Index configuration
    st.subheader("Index Settings")
    index_name = st.text_input(
        "Index Name",
        value="hybrid-search-demo",
        help="Name for your Pinecone index"
    )
    
    dimension = st.number_input(
        "Embedding Dimension",
        value=384,
        help="Dimension of embeddings (384 for all-MiniLM-L6-v2)"
    )
    
    metric = st.selectbox(
        "Distance Metric",
        ["dotproduct", "cosine", "euclidean"],
        index=0,
        help="dotproduct is required for hybrid search with sparse vectors"
    )
    
    cloud = st.selectbox("Cloud Provider", ["aws"], index=0)
    region = st.selectbox("Region", ["us-east-1"], index=0)
    
    st.markdown("---")
    
    # Initialize button
    if st.button("🚀 Initialize System", type="primary"):
        if not pinecone_api_key:
            st.error("❌ Pinecone API key not found in secrets. Please add it in Streamlit Cloud settings.")
            st.info("Go to: App Dashboard → Settings → Secrets → Add PINECONE_API_KEY")
        else:
            with st.spinner("Initializing Pinecone and embeddings..."):
                try:
                    # Initialize Pinecone
                    pc = Pinecone(api_key=pinecone_api_key)
                    st.session_state.pinecone_client = pc
                    
                    # Create or connect to index
                    existing_indexes = [index.name for index in pc.list_indexes()]
                    
                    if index_name not in existing_indexes:
                        st.info(f"Creating new index: {index_name}")
                        pc.create_index(
                            name=index_name,
                            dimension=dimension,
                            metric=metric,
                            spec=ServerlessSpec(
                                cloud=cloud,
                                region=region
                            )
                        )
                        # Wait for index to be ready
                        time.sleep(5)
                    
                    # Connect to index
                    index = pc.Index(index_name)
                    st.session_state.index = index
                    
                    # Initialize embeddings
                    embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2"
                    )
                    
                    # Initialize BM25 encoder with proper default corpus
                    bm25_encoder = BM25Encoder()
                    
                    # Fit BM25 on a meaningful default corpus to avoid empty vectors
                    default_corpus = [
                        "This is a sample document for initialization",
                        "Machine learning and artificial intelligence",
                        "Natural language processing and text analysis",
                        "Python programming and data science",
                        "Information retrieval and search systems"
                    ]
                    bm25_encoder.fit(default_corpus)
                    
                    # Store BM25 encoder in session state
                    st.session_state.bm25_encoder = bm25_encoder
                    
                    # Initialize retriever
                    retriever = PineconeHybridSearchRetriever(
                        embeddings=embeddings,
                        sparse_encoder=bm25_encoder,
                        index=index
                    )
                    
                    st.session_state.retriever = retriever
                    st.session_state.initialized = True
                    
                    st.success("✅ System initialized successfully!")
                    
                except Exception as e:
                    st.error(f"Error initializing system: {str(e)}")
    
    st.markdown("---")
    
    # Info section
    st.header("ℹ️ About Hybrid Search")
    st.info("""
    **Hybrid Search** combines:
    
    1️⃣ **Semantic Search** (Dense Vectors)
    - Uses embeddings to find similar meanings
    - Powered by HuggingFace transformers
    
    2️⃣ **Keyword Search** (Sparse Vectors)
    - Uses BM25 for exact keyword matches
    - Better for specific terms
    
    3️⃣ **Reciprocal Rank Fusion (RRF)**
    - Combines both results
    - Optimizes final ranking
    """)
    
    st.header("📊 System Status")
    if st.session_state.initialized:
        st.success("🟢 System Ready")
        st.metric("Documents Indexed", len(st.session_state.documents))
    else:
        st.warning("🟡 Not Initialized")

# Main content area
if not st.session_state.initialized:
    st.warning("⚠️ Please initialize the system using the sidebar")
    
    # Quick start guide
    st.markdown("### 🚀 Quick Start Guide")
    st.markdown("""
    1. Make sure your Pinecone API key is configured in Streamlit secrets
    2. Click "Initialize System" in the sidebar
    3. Add documents using the tabs below
    4. Start searching!
    """)
    
    # Configuration help
    with st.expander("⚙️ How to Configure Secrets"):
        st.markdown("""
        **In Streamlit Cloud:**
        1. Go to your app dashboard
        2. Click "Settings" → "Secrets"
        3. Add:
        ```toml
        PINECONE_API_KEY = "your-api-key-here"
        ```
        4. Click "Save"
        5. App will restart automatically
        
        **For Local Development:**
        1. Create `.streamlit/secrets.toml`
        2. Add the same content as above
        3. This file is gitignored for security
        """)
    
    # Example usage
    with st.expander("📖 Example Documents"):
        st.code("""
# Sample documents you can add:
- "In 2023 I visited Paris and saw the Eiffel Tower"
- "Machine learning is a subset of artificial intelligence"
- "Python is a popular programming language for data science"
- "The quick brown fox jumps over the lazy dog"
- "Climate change is affecting global weather patterns"
        """)
    
    with st.expander("🔍 Example Queries"):
        st.code("""
# Try these queries after adding documents:
- "Paris travel" (will match keyword and semantic meaning)
- "AI and ML" (will find machine learning content)
- "programming" (will find Python-related content)
- "2023 trip" (will find travel content with date)
        """)

else:
    # Document management
    st.header("📄 Document Management")
    
    tab1, tab2, tab3 = st.tabs(["➕ Add Documents", "🔍 Search", "📋 View Documents"])
    
    with tab1:
        st.subheader("Add Documents to Index")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            doc_input_method = st.radio(
                "Input Method",
                ["Single Document", "Multiple Documents", "Upload Text File"],
                horizontal=True
            )
        
        if doc_input_method == "Single Document":
            doc_text = st.text_area(
                "Document Content",
                placeholder="Enter your document text here...",
                height=150
            )
            
            doc_metadata = st.text_input(
                "Metadata (optional)",
                placeholder='{"source": "manual", "category": "example"}'
            )
            
            if st.button("Add Document", type="primary"):
                if doc_text:
                    try:
                        # Track document first
                        st.session_state.documents.append({
                            'text': doc_text,
                            'metadata': doc_metadata if doc_metadata else '{}'
                        })
                        
                        # Refit BM25 encoder with ALL documents (including new one)
                        all_texts = [doc['text'] for doc in st.session_state.documents]
                        st.session_state.bm25_encoder.fit(all_texts)
                        
                        # Update retriever's sparse encoder to use the refitted one
                        st.session_state.retriever.sparse_encoder = st.session_state.bm25_encoder
                        
                        # Now add to retriever with properly fitted BM25
                        st.session_state.retriever.add_texts(
                            texts=[doc_text],
                            metadatas=[eval(doc_metadata) if doc_metadata else {}]
                        )
                        
                        st.success(f"✅ Document added! Total documents: {len(st.session_state.documents)}")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error adding document: {str(e)}")
                        # Remove from documents if add failed
                        if st.session_state.documents and st.session_state.documents[-1]['text'] == doc_text:
                            st.session_state.documents.pop()
                else:
                    st.warning("Please enter document text")
        
        elif doc_input_method == "Multiple Documents":
            st.info("Enter one document per line")
            bulk_docs = st.text_area(
                "Documents (one per line)",
                placeholder="In 2023 I visited Paris\nMachine learning is amazing\nPython is great for AI",
                height=200
            )
            
            if st.button("Add All Documents", type="primary"):
                if bulk_docs:
                    try:
                        docs = [line.strip() for line in bulk_docs.split('\n') if line.strip()]
                        
                        # Track documents first
                        for doc in docs:
                            st.session_state.documents.append({
                                'text': doc,
                                'metadata': '{}'
                            })
                        
                        # Refit BM25 encoder with ALL documents
                        all_texts = [doc['text'] for doc in st.session_state.documents]
                        st.session_state.bm25_encoder.fit(all_texts)
                        
                        # Update retriever's sparse encoder
                        st.session_state.retriever.sparse_encoder = st.session_state.bm25_encoder
                        
                        # Now add to retriever
                        st.session_state.retriever.add_texts(texts=docs)
                        
                        st.success(f"✅ Added {len(docs)} documents! Total: {len(st.session_state.documents)}")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error adding documents: {str(e)}")
                else:
                    st.warning("Please enter documents")
        
        else:  # Upload Text File
            uploaded_file = st.file_uploader("Upload Text File", type=['txt'])
            
            if uploaded_file:
                content = uploaded_file.read().decode('utf-8')
                st.text_area("File Preview", content, height=200, disabled=True)
                
                if st.button("Add from File", type="primary"):
                    try:
                        docs = [line.strip() for line in content.split('\n') if line.strip()]
                        
                        # Add documents
                        st.session_state.retriever.add_texts(texts=docs)
                        
                        # Track documents
                        for doc in docs:
                            st.session_state.documents.append({
                                'text': doc,
                                'metadata': f'{{"source": "{uploaded_file.name}"}}'
                            })
                        
                        # Refit BM25 encoder with all documents
                        all_texts = [doc['text'] for doc in st.session_state.documents]
                        st.session_state.retriever.sparse_encoder.fit(all_texts)
                        
                        st.success(f"✅ Added {len(docs)} documents from file!")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        # Quick add sample documents
        st.markdown("---")
        st.subheader("📚 Quick Sample Documents")
        if st.button("Add Sample Documents"):
            sample_docs = [
                "In 2023 I visited Paris and saw the Eiffel Tower",
                "Machine learning is a subset of artificial intelligence that focuses on data-driven algorithms",
                "Python is a popular programming language widely used in data science and AI",
                "The quick brown fox jumps over the lazy dog",
                "Climate change is affecting global weather patterns and ecosystems",
                "Natural language processing enables computers to understand human language",
                "Deep learning uses neural networks with multiple layers",
                "Paris is the capital city of France, known for its art, culture, and cuisine"
            ]
            
            try:
                # Track documents first
                for doc in sample_docs:
                    st.session_state.documents.append({
                        'text': doc,
                        'metadata': '{"source": "sample"}'
                    })
                
                # Refit BM25 encoder with ALL documents
                all_texts = [doc['text'] for doc in st.session_state.documents]
                st.session_state.bm25_encoder.fit(all_texts)
                
                # Update retriever's sparse encoder
                st.session_state.retriever.sparse_encoder = st.session_state.bm25_encoder
                
                # Now add to retriever
                st.session_state.retriever.add_texts(texts=sample_docs)
                
                st.success(f"✅ Added {len(sample_docs)} sample documents!")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    with tab2:
        st.subheader("🔍 Hybrid Search Query")
        
        if len(st.session_state.documents) == 0:
            st.info("Add some documents first to test search functionality")
        else:
            query = st.text_input(
                "Search Query",
                placeholder="Enter your search query...",
                help="Try: 'Paris travel', 'AI and ML', 'programming', etc."
            )
            
            k = st.slider("Number of Results", min_value=1, max_value=10, value=3)
            
            col1, col2 = st.columns(2)
            with col1:
                alpha = st.slider(
                    "Hybrid Weight (α)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                    help="0 = keyword only, 1 = semantic only, 0.5 = balanced"
                )
            
            if st.button("🔍 Search", type="primary"):
                if query:
                    try:
                        with st.spinner("Searching..."):
                            # Perform hybrid search
                            results = st.session_state.retriever.invoke(
                                query,
                                search_kwargs={'k': k}
                            )
                            
                            st.success(f"Found {len(results)} results")
                            
                            # Display results
                            st.markdown("### 📊 Search Results")
                            for i, doc in enumerate(results, 1):
                                st.markdown(f"""
                                <div class="search-result">
                                    <strong>Result {i}</strong><br>
                                    <p>{doc.page_content}</p>
                                    <small>Metadata: {doc.metadata}</small>
                                </div>
                                """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Search error: {str(e)}")
                else:
                    st.warning("Please enter a search query")
            
            # Search examples
            st.markdown("---")
            st.markdown("#### 💡 Try These Example Queries:")
            example_queries = [
                "Paris travel 2023",
                "machine learning AI",
                "Python programming",
                "climate change weather",
                "natural language"
            ]
            
            cols = st.columns(len(example_queries))
            for idx, example in enumerate(example_queries):
                with cols[idx]:
                    if st.button(f"🔍 {example}", key=f"ex_{idx}"):
                        st.session_state.example_query = example
                        st.rerun()
    
    with tab3:
        st.subheader("📋 Indexed Documents")
        
        if len(st.session_state.documents) == 0:
            st.info("No documents indexed yet")
        else:
            st.metric("Total Documents", len(st.session_state.documents))
            
            # Display documents
            for i, doc in enumerate(st.session_state.documents, 1):
                with st.expander(f"Document {i}: {doc['text'][:50]}..."):
                    st.write("**Content:**")
                    st.write(doc['text'])
                    st.write("**Metadata:**")
                    st.code(doc['metadata'])
            
            # Clear all documents
            st.markdown("---")
            if st.button("🗑️ Clear All Documents", type="secondary"):
                if st.session_state.index:
                    try:
                        # Delete all vectors from index
                        st.session_state.index.delete(delete_all=True)
                        st.session_state.documents = []
                        st.success("All documents cleared!")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error clearing documents: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>🔍 Hybrid Search RAG | Powered by Pinecone + LangChain + HuggingFace</p>
        <p><small>Combining Semantic Search (Dense Vectors) + Keyword Search (Sparse Vectors)</small></p>
    </div>
    """, unsafe_allow_html=True)
