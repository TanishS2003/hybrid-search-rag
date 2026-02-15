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
    pinecone_api_key = st.secrets.get("PINECONE_API_KEY", "")
    st.success("✅ API Key Detected") if pinecone_api_key else st.error("❌ Missing API Key")
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
                    
                    # CRITICAL: Proper BM25 initialization
                    bm25 = BM25Encoder()
                    bm25.fit(["Technology and innovation", "AI and machine learning",
                             "Natural language processing", "Python data science",
                             "Information retrieval", "Deep learning networks",
                             "Cloud computing", "Database systems"])
                    
                    retriever = PineconeHybridSearchRetriever(embeddings=embeddings, 
                                                            sparse_encoder=bm25, index=index)
                    st.session_state.update({'retriever': retriever, 'bm25_encoder': bm25, 
                                           'initialized': True})
                    st.success("✅ Ready!"); st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    
    if st.session_state.initialized:
        st.metric("📊 Documents", len(st.session_state.documents))

st.title("🔍 Hybrid Search RAG Engine")
st.markdown("*Semantic + Keyword Search*"); st.markdown("---")

if not st.session_state.initialized:
    st.warning("⚠️ Initialize engine from sidebar")
else:
    tab1, tab2 = st.tabs(["📄 Documents", "🔎 Search"])
    
    with tab1:
        if st.button("📚 Load Samples", type="primary"):
            samples = [
                "In 2023 I visited Paris France and saw the Eiffel Tower cultural experience",
                "Machine learning is AI subset that learns patterns from data automatically",
                "Python is most popular language for data science and AI development",
                "Climate change causes severe impacts on global weather and ecosystems",
                "Natural language processing enables computers to understand human language",
                "Deep learning uses multi-layer neural networks for complex patterns",
                "Louvre Museum Paris houses 35000 artworks including Mona Lisa",
                "AI revolutionizing healthcare finance transportation manufacturing",
                "Eiffel Tower 330 meters tall Paris most visited monument globally",
                "Data science combines programming statistics domain knowledge insights",
                "Paris capital of France known for art fashion cuisine landmarks",
                "Neural networks inspired by biological systems power deep learning"
            ]
            try:
                st.session_state.documents.extend([s for s in samples if s not in st.session_state.documents])
                st.session_state.bm25_encoder.fit(st.session_state.documents)
                st.session_state.retriever.sparse_encoder = st.session_state.bm25_encoder
                st.session_state.retriever.add_texts(samples)
                st.success(f"✅ Loaded {len(samples)} docs!"); time.sleep(1); st.rerun()
            except Exception as e:
                st.error(f"❌ {e}")
        
        with st.expander("✍️ Add Custom"):
            text = st.text_area("Content:", height=200)
            c1, c2 = st.columns(2)
            chunk_size = c1.number_input("Chunk size", 300, 2000, 500)
            overlap = c2.number_input("Overlap", 0, 200, 50)
            
            if st.button("📥 Index"):
                if text:
                    try:
                        chunks = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                chunk_overlap=overlap).split_text(text) if len(text) > chunk_size else [text]
                        st.session_state.documents.extend(chunks)
                        st.session_state.bm25_encoder.fit(st.session_state.documents)
                        st.session_state.retriever.sparse_encoder = st.session_state.bm25_encoder
                        st.session_state.retriever.add_texts(chunks)
                        st.success(f"✅ {len(chunks)} chunks!"); time.sleep(1); st.rerun()
                    except Exception as e:
                        st.error(f"❌ {e}")
    
    with tab2:
        if not st.session_state.documents:
            st.info("Load documents first")
        else:
            query = st.text_input("Query:", placeholder="Paris, machine learning, Python...")
            c1, c2 = st.columns([3,1])
            alpha = c1.slider("Balance", 0.0, 1.0, 0.5, 0.05, 
                            help="0=keyword, 0.5=balanced, 1=semantic")
            c1.caption(f"{'🔤 Keyword' if alpha<0.3 else '🧠 Semantic' if alpha>0.7 else '⚖️ Balanced'}")
            k = c2.number_input("Results", 1, 10, 5)
            
            if st.button("🔍 Search", type="primary"):
                if query:
                    try:
                        results = st.session_state.retriever.invoke(query, search_kwargs={'k': k})
                        if results:
                            st.markdown(f"### 🎯 Results: *\"{query}\"*"); st.markdown("---")
                            for i, doc in enumerate(results, 1):
                                st.markdown(f'<div class="search-result"><strong>#{i}</strong><br><br>'
                                          f'<p>{doc.page_content}</p></div>', unsafe_allow_html=True)
                        else:
                            st.warning("No results")
                    except Exception as e:
                        st.error(f"❌ {e}")
            
            st.markdown("---"); st.markdown("**💡 Examples:**")
            c1,c2,c3,c4 = st.columns(4)
            if c1.button("🗼 Paris"): st.rerun()
            if c2.button("🤖 ML"): st.rerun()
            if c3.button("🐍 Python"): st.rerun()
            if c4.button("🌍 Climate"): st.rerun()

st.caption("🔍 Hybrid RAG | Pinecone + LangChain + HuggingFace")
