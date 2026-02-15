import streamlit as st
import time
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter

st.set_page_config(page_title="Hybrid Search RAG", page_icon="🔍", layout="wide")

st.markdown("""<style>
.main { padding: 2rem; }
.search-result { padding: 1.5rem; border-radius: 10px; background-color: #fff; 
border-left: 5px solid #4CAF50; margin-bottom: 1.2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
.search-result p { color: #1f1f1f !important; font-size: 15px; line-height: 1.6; }
</style>""", unsafe_allow_html=True)

if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

with st.sidebar:
    st.header("⚙️ System Setup")
    
    try:
        pinecone_api_key = st.secrets.get("PINECONE_API_KEY", "")
        if pinecone_api_key:
            st.success("✅ API Key Detected")
        else:
            st.error("❌ Missing API Key")
    except:
        pinecone_api_key = ""
        st.error("❌ Missing API Key")
    
    index_name = st.text_input("Index Name", value="hybrid-rag-demo")
    
    if st.button("🚀 Initialize", type="primary"):
        if pinecone_api_key:
            with st.spinner("Initializing..."):
                try:
                    pc = Pinecone(api_key=pinecone_api_key)
                    if index_name not in [i.name for i in pc.list_indexes()]:
                        pc.create_index(name=index_name, dimension=384, metric="dotproduct",
                                      spec=ServerlessSpec(cloud="aws", region="us-east-1"))
                        time.sleep(5)
                    
                    index = pc.Index(index_name)
                    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                    
                    # Proper BM25 initialization
                    bm25 = BM25Encoder()
                    bm25.fit(["Initialize system", "Setup complete"])
                    
                    retriever = PineconeHybridSearchRetriever(
                        embeddings=embeddings, 
                        sparse_encoder=bm25, 
                        index=index
                    )
                    
                    st.session_state.update({
                        'retriever': retriever, 
                        'bm25_encoder': bm25, 
                        'index': index,
                        'initialized': True
                    })
                    st.success("✅ Ready!")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    if st.session_state.initialized:
        st.markdown("---")
        st.metric("📊 Documents", len(st.session_state.documents))
        
        # CRITICAL: Clear index button
        if st.button("🗑️ Clear All Data", type="secondary"):
            try:
                # Delete all vectors from Pinecone
                st.session_state.index.delete(delete_all=True)
                # Clear local documents
                st.session_state.documents = []
                # Refit BM25 on empty
                st.session_state.bm25_encoder.fit(["Initialize system", "Setup complete"])
                st.session_state.retriever.sparse_encoder = st.session_state.bm25_encoder
                st.success("✅ All data cleared!")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

st.title("🔍 Hybrid Search RAG Engine")
st.markdown("*Semantic + Keyword Search on YOUR documents*")
st.markdown("---")

if not st.session_state.initialized:
    st.warning("⚠️ Initialize engine from sidebar")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("""
        **What is Hybrid Search?**
        
        Combines:
        - **Semantic**: Understands meaning
        - **Keyword**: Exact term matching
        - **RRF**: Merges both for accuracy
        """)
    with col2:
        st.info("""
        **Quick Start:**
        
        1. Initialize engine (sidebar)
        2. Upload YOUR document
        3. Search your content
        4. Adjust balance (Alpha)
        """)
else:
    tab1, tab2 = st.tabs(["📄 Your Documents", "🔎 Search"])
    
    with tab1:
        st.header("📄 Add Your Documents")
        
        st.info("⚠️ **Important**: Clear old data before adding new documents to avoid mixing content!")
        
        with st.expander("✍️ Upload Your Document", expanded=True):
            text = st.text_area("Paste your content (resume, notes, etc.):", height=300, 
                              placeholder="Paste your text here...")
            
            c1, c2 = st.columns(2)
            chunk_size = c1.number_input("Chunk size (chars)", 300, 2000, 500)
            overlap = c2.number_input("Overlap", 0, 200, 50)
            
            if st.button("📥 Index My Document", type="primary"):
                if text:
                    with st.spinner("Processing..."):
                        try:
                            # Split into chunks if needed
                            if len(text) > chunk_size:
                                splitter = RecursiveCharacterTextSplitter(
                                    chunk_size=chunk_size, 
                                    chunk_overlap=overlap
                                )
                                chunks = splitter.split_text(text)
                            else:
                                chunks = [text]
                            
                            # Add to documents
                            st.session_state.documents.extend(chunks)
                            
                            # Refit BM25
                            st.session_state.bm25_encoder.fit(st.session_state.documents)
                            st.session_state.retriever.sparse_encoder = st.session_state.bm25_encoder
                            
                            # Add to Pinecone
                            st.session_state.retriever.add_texts(chunks)
                            
                            st.success(f"✅ Indexed {len(chunks)} chunk(s)!")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                else:
                    st.warning("⚠️ Please paste some text")
        
        st.markdown("---")
        
        # Sample documents section
        with st.expander("📚 Or Try Sample Documents"):
            st.info("These are demo documents about Paris, ML, Python, etc.")
            
            if st.button("Load Sample Data"):
                samples = [
                    "In 2023 I visited Paris France and saw the Eiffel Tower",
                    "Machine learning is AI subset that learns from data",
                    "Python is most popular for data science and AI",
                    "Climate change impacts global weather patterns",
                    "Natural language processing enables language understanding",
                    "Deep learning uses multi-layer neural networks"
                ]
                
                try:
                    st.session_state.documents.extend(samples)
                    st.session_state.bm25_encoder.fit(st.session_state.documents)
                    st.session_state.retriever.sparse_encoder = st.session_state.bm25_encoder
                    st.session_state.retriever.add_texts(samples)
                    st.success("✅ Loaded samples!")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ {str(e)}")
        
        if st.session_state.documents:
            st.markdown("---")
            st.subheader(f"📋 Current Index ({len(st.session_state.documents)} chunks)")
            
            with st.expander("View indexed content"):
                for i, doc in enumerate(st.session_state.documents[:10], 1):
                    st.text_area(f"Chunk {i}", doc[:200] + "...", height=80, 
                               disabled=True, key=f"doc_{i}", label_visibility="collapsed")
                if len(st.session_state.documents) > 10:
                    st.caption(f"Showing 10 of {len(st.session_state.documents)} chunks")
    
    with tab2:
        st.header("🔎 Search Your Content")
        
        if not st.session_state.documents:
            st.info("ℹ️ No documents indexed yet. Upload your document first!")
        else:
            st.info(f"📊 Searching across {len(st.session_state.documents)} document chunks")
            
            query = st.text_input("Enter search query:", 
                                placeholder="e.g., education, experience, skills...")
            
            c1, c2 = st.columns([3,1])
            alpha = c1.slider("Balance", 0.0, 1.0, 0.5, 0.05, 
                            help="0=keyword only, 0.5=balanced, 1=semantic only")
            c1.caption(f"{'🔤 Keyword' if alpha<0.3 else '🧠 Semantic' if alpha>0.7 else '⚖️ Balanced'}")
            k = c2.number_input("Results", 1, 10, 5)
            
            if st.button("🔍 Search", type="primary"):
                if query:
                    with st.spinner("Searching..."):
                        try:
                            results = st.session_state.retriever.invoke(
                                query, 
                                search_kwargs={'k': k}
                            )
                            
                            if results:
                                st.markdown(f"### 🎯 Results for: *\"{query}\"*")
                                st.caption(f"*Found {len(results)} matches*")
                                st.markdown("---")
                                
                                for i, doc in enumerate(results, 1):
                                    st.markdown(f'<div class="search-result">'
                                              f'<strong>Match #{i}</strong><br><br>'
                                              f'<p>{doc.page_content}</p>'
                                              f'</div>', unsafe_allow_html=True)
                            else:
                                st.warning("No results found. Try different keywords.")
                        except Exception as e:
                            st.error(f"❌ Search error: {str(e)}")
                else:
                    st.warning("⚠️ Enter a search query")
            
            st.markdown("---")
            st.caption("💡 Tip: Try searching for specific terms from your document")

st.markdown("---")
st.caption("🔍 Hybrid RAG | Pinecone + LangChain + HuggingFace")
