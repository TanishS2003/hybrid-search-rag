"""
Test script for Hybrid Search RAG components
Run this to verify your setup before deploying
"""

import sys
import os

def test_imports():
    """Test if all required packages are installed"""
    print("🔍 Testing imports...")
    
    try:
        import streamlit
        print("✅ streamlit")
    except ImportError:
        print("❌ streamlit - run: pip install streamlit")
        return False
    
    try:
        from pinecone import Pinecone
        print("✅ pinecone-client")
    except ImportError:
        print("❌ pinecone-client - run: pip install pinecone-client")
        return False
    
    try:
        from pinecone_text.sparse import BM25Encoder
        print("✅ pinecone-text")
    except ImportError:
        print("❌ pinecone-text - run: pip install pinecone-text")
        return False
    
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        print("✅ langchain-huggingface")
    except ImportError:
        print("❌ langchain-huggingface - run: pip install langchain-huggingface")
        return False
    
    try:
        from langchain_community.retrievers import PineconeHybridSearchRetriever
        print("✅ langchain-community")
    except ImportError:
        print("❌ langchain-community - run: pip install langchain-community")
        return False
    
    try:
        import sentence_transformers
        print("✅ sentence-transformers")
    except ImportError:
        print("❌ sentence-transformers - run: pip install sentence-transformers")
        return False
    
    print("\n✅ All imports successful!\n")
    return True


def test_embeddings():
    """Test if embeddings model can be loaded"""
    print("🔍 Testing embeddings model...")
    
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Test encoding
        text = "This is a test sentence"
        vector = embeddings.embed_query(text)
        
        assert len(vector) == 384, f"Expected 384 dimensions, got {len(vector)}"
        print(f"✅ Embeddings model loaded (dimension: {len(vector)})")
        print(f"✅ Test encoding successful\n")
        return True
        
    except Exception as e:
        print(f"❌ Embeddings test failed: {e}\n")
        return False


def test_bm25():
    """Test BM25 encoder"""
    print("🔍 Testing BM25 encoder...")
    
    try:
        from pinecone_text.sparse import BM25Encoder
        
        # Create encoder
        bm25 = BM25Encoder()
        
        # Fit with sample documents
        docs = [
            "This is a test document",
            "Another test document here",
            "One more document for testing"
        ]
        bm25.fit(docs)
        
        # Encode query
        query = "test document"
        sparse_vec = bm25.encode_queries([query])
        
        print(f"✅ BM25 encoder initialized")
        print(f"✅ Fitted on {len(docs)} documents")
        print(f"✅ Query encoding successful\n")
        return True
        
    except Exception as e:
        print(f"❌ BM25 test failed: {e}\n")
        return False


def test_pinecone_connection(api_key=None):
    """Test Pinecone connection (requires API key)"""
    print("🔍 Testing Pinecone connection...")
    
    if not api_key:
        print("⚠️  Skipping Pinecone test (no API key provided)")
        print("   To test: python test_setup.py YOUR_PINECONE_API_KEY\n")
        return True
    
    try:
        from pinecone import Pinecone
        
        pc = Pinecone(api_key=api_key)
        
        # List indexes
        indexes = pc.list_indexes()
        print(f"✅ Connected to Pinecone")
        print(f"✅ Found {len(indexes)} existing indexes\n")
        return True
        
    except Exception as e:
        print(f"❌ Pinecone connection failed: {e}\n")
        return False


def run_all_tests(api_key=None):
    """Run all tests"""
    print("=" * 60)
    print("🧪 HYBRID SEARCH RAG - SETUP TESTS")
    print("=" * 60)
    print()
    
    results = []
    
    # Test imports
    results.append(("Imports", test_imports()))
    
    # Test embeddings
    results.append(("Embeddings", test_embeddings()))
    
    # Test BM25
    results.append(("BM25", test_bm25()))
    
    # Test Pinecone (if API key provided)
    results.append(("Pinecone", test_pinecone_connection(api_key)))
    
    # Summary
    print("=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:20} {status}")
    
    print()
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("🎉 All tests passed! You're ready to go!")
        print()
        print("Next steps:")
        print("1. Run: streamlit run app.py")
        print("2. Enter your Pinecone API key in the sidebar")
        print("3. Click 'Initialize System'")
        print("4. Start searching!")
    else:
        print("⚠️  Some tests failed. Please fix the issues above.")
        print()
        print("Common fixes:")
        print("- Run: pip install -r requirements.txt")
        print("- Check your Pinecone API key")
        print("- Ensure Python 3.9+ is installed")
    
    print()
    return all_passed


if __name__ == "__main__":
    # Get API key from command line if provided
    api_key = sys.argv[1] if len(sys.argv) > 1 else None
    
    success = run_all_tests(api_key)
    sys.exit(0 if success else 1)
