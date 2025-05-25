from langchain_openai import OpenAIEmbeddings 
from langchain_community.vectorstores.upstash import UpstashVectorStore
import click
import os
from dotenv import load_dotenv
import sys

# Handle imports for both src/ and root directory usage
try:
    from extraction import extract_papers
except ImportError:
    # If running from src/ directory, try parent directory
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
       from extraction import extract_papers
    except ImportError:
        print(f"Exception in import")
        
def create_embeddings(batch_size, splits, embedding_model):
    # Initialize Upstash vector store with OpenAI embeddings
    try:
        # Use OpenAI embeddings
        embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=os.environ.get("OPENAI_API_KEY")
        )
        
        upstash_vector_store = UpstashVectorStore(embedding=embeddings)
        
        click.echo(f"Indexing {len(splits)} chunks to Upstash...")
        ids = upstash_vector_store.add_documents(splits, batch_size=batch_size)
        click.echo(f"✅ Successfully indexed {len(ids)} vectors to Upstash")
        
    except Exception as e:
        click.echo(f"❌ Error during indexing: {e}")
        raise
    
