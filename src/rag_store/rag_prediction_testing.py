import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores.upstash import UpstashVectorStore
from rag import RAG


# Functions for terminal testing
def test_retrieval_only(rag_system, query, k=3):
    load_dotenv()
    
    """Test only the retrieval part without LLM generation"""
    print(f"🔍 Testing retrieval for: '{query}'")
    print("-" * 50)
    
    context, source_documents = rag_system.get_context(query, k=k)
    
    if not source_documents:
        print("❌ No documents found for this query.")
        return
    
    print(f"✅ Found {len(source_documents)} relevant documents:")
    print()
    
    for i, (doc, score) in enumerate(source_documents, 1):
        print(f"📄 Document {i} (Score: {score:.4f})")
        print(f"Title: {doc.metadata.get('title', 'No title')}")
        print(f"Authors: {doc.metadata.get('authors', 'Unknown')}")
        print(f"Content: {doc.page_content[:200]}...")
        print("-" * 30)


def test_full_rag(rag_system:RAG, query):
    load_dotenv()
    
    """Test the full RAG pipeline (retrieval + generation)"""
    print(f"🤖 Testing full RAG for: '{query}'")
    print("=" * 60)
    
    prediction = rag_system.predict(query)
    
    print("📋 ANSWER:")
    print(prediction["answer"])
    print()
    
    print("📚 SOURCE DOCUMENTS:")
    source_docs = prediction["source_documents"]
    if source_docs:
        for i, (doc, score) in enumerate(source_docs, 1):
            print(f"{i}. {doc.metadata.get('title', 'No title')} (Score: {score:.3f})")
    else:
        print("No source documents found.")
    print()


def interactive_mode(rag_system):
    """Interactive mode for testing RAG system"""
    print("🚀 Interactive RAG Testing Mode")
    print("Commands:")
    print("  - Type your question to get an answer")
    print("  - Type 'search:<query>' to test retrieval only")
    print("  - Type 'quit' to exit")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\n💭 Your input: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if user_input.startswith('search:'):
                query = user_input[7:].strip()
                test_retrieval_only(rag_system, query)
            elif user_input:
                test_full_rag(rag_system, user_input)
            else:
                print("Please enter a question or command.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Main function to test RAG system in terminal"""
    print("🧪 RAG System Terminal Tester")
    print("=" * 60)
    
    # Check environment variables
    required_vars = ["OPENAI_API_KEY", "UPSTASH_VECTOR_REST_URL", "UPSTASH_VECTOR_REST_TOKEN"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("\nPlease add them to your .env file and run the indexing script first:")
        print("python index_papers.py index --query 'your topic' --max_papers 20")
        return
    
    try:
        # Initialize embeddings
        print("🔧 Initializing OpenAI embeddings...")
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.environ.get("OPENAI_API_KEY")
        )
        
        # Initialize RAG system (no chat_box for terminal)
        print("🔧 Initializing RAG system...")
        rag = RAG(chat_box=None, embeddings=embeddings)
        
        # Test connection
        print("🔧 Testing vector store connection...")
        if not rag.test_connection():
            print("❌ Vector store connection failed. Make sure you've indexed some papers first:")
            print("python index_papers.py index --query 'attention mechanism' --max_papers 20")
            return
        
        print("✅ RAG system initialized successfully!")
        print()
        
        # Quick test with sample queries
        sample_queries = [
            "What are attention mechanisms?",
            "How do transformers work?",
            "What is self-attention?"
        ]
        
        print("🧪 Quick test with sample queries:")
        for query in sample_queries:
            print(f"\n🔍 Testing: '{query}'")
            context, results = rag.get_context(query, k=2)
            if results:
                print(f"✅ Found {len(results)} relevant documents")
            else:
                print("❌ No relevant documents found")
        
        print("\n" + "=" * 60)
        
        # Start interactive mode
        interactive_mode(rag)
        
    except Exception as e:
        print(f"❌ Error initializing RAG system: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure your .env file has the correct credentials")
        print("2. Run the indexing script first to populate the vector database")
        print("3. Check that your OpenAI API key is valid")


if __name__ == "__main__":
    main()