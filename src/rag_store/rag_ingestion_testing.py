import os
import click
from dotenv import load_dotenv

print(f"Current working directory: {os.getcwd()}")
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings 
from langchain_community.vectorstores.upstash import UpstashVectorStore
import sys

# Handle imports for both src/ and root directory usage
try:
    from extraction import extract_papers
    from indexing import extract_paper_abstracts, split_and_chunk_documents
    from embeddings import create_embeddings
except ImportError:
    # If running from src/ directory, try parent directory
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from extraction import extract_papers
        from indexing import extract_paper_abstracts, split_and_chunk_documents
        from embeddings import create_embeddings
    except ImportError:
        print(f"Exception in import")
        


@click.command()
@click.option("--query", type=str, required=True, help="Search query for papers")
@click.option("--batch_size", type=int, default=32, help="Batch size for indexing")
@click.option("--max_papers", type=int, default=5, help="Maximum number of papers to extract")
@click.option("--max_chunks", type=int, default=None, help="Maximum number of text chunks to index")
@click.option("--embedding_model", type=str, default="text-embedding-3-small", help="OpenAI embedding model")
def test_rag_ingestion(query, max_papers, batch_size, max_chunks, embedding_model):
    """Test Complete RAG"""
    load_dotenv()
    
    # Check required environment variables
    required_vars = ["UPSTASH_VECTOR_REST_URL", "UPSTASH_VECTOR_REST_TOKEN", "OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        click.echo(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        click.echo("\nPlease add them to your .env file:")
        for var in missing_vars:
            if var == "OPENAI_API_KEY":
                click.echo(f"  {var}=sk-your-openai-api-key-here")
            elif var == "UPSTASH_VECTOR_REST_URL":
                click.echo(f"  {var}=https://your-upstash-url.upstash.io")
            elif var == "UPSTASH_VECTOR_REST_TOKEN":
                click.echo(f"  {var}=your-upstash-token-here")
        return
    
    
    click.echo(f"Testing Complete RAG for query: '{query}'")
    paper_abstracts = extract_paper_abstracts(query=query,max_papers=max_papers,embedding_model=embedding_model)
    if not paper_abstracts:
        return
    
    click.echo(f"Testing Splitting and Chunking: '{query}'")
    splits = split_and_chunk_documents(max_chunks=max_chunks,papers_with_abstracts=paper_abstracts)
    if not splits:
        return
    
    create_embeddings(batch_size=batch_size,splits=splits,embedding_model=embedding_model)
    


@click.command()
def test_upstash():
    """Test Upstash vector store connection"""
    load_dotenv()
    
    required_vars = ["UPSTASH_VECTOR_REST_URL", "UPSTASH_VECTOR_REST_TOKEN", "OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        click.echo(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        return
    
    try:
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.environ.get("OPENAI_API_KEY")
        )
        
        # Test UpstashVectorStore initialization
        upstash_vector_store = UpstashVectorStore(embedding=embeddings)
        
        click.echo("✅ UpstashVectorStore initialized successfully!")
        
        # Test adding a simple document
        from langchain.docstore.document import Document
        test_doc = Document(
            page_content="This is a test document for Upstash vector store",
            metadata={"test": True}
        )
        
        ids = upstash_vector_store.add_documents([test_doc])
        click.echo(f"✅ Test document added with ID: {ids[0]}")
        
        # Test similarity search
        results = upstash_vector_store.similarity_search("test document", k=1)
        if results:
            click.echo(f"✅ Similarity search working! Found: '{results[0].page_content[:50]}...'")
        
    except Exception as e:
        click.echo(f"❌ Upstash test failed: {e}")
        if "API key" in str(e).lower():
            click.echo("💡 Check if your OpenAI API key is correct")
        elif "upstash" in str(e).lower():
            click.echo("💡 Check if your Upstash credentials are correct")
            
            

@click.command()
@click.option("--query", type=str, required=True)
@click.option("--max_papers", type=int, default=5)
def test_extraction(query, max_papers):
    """Test paper extraction without indexing"""
    load_dotenv()
    
    click.echo(f"Testing extraction for query: '{query}'")
    papers = extract_papers(query, max_results=max_papers)
    
    click.echo(f"Found {len(papers)} papers:")
    for i, paper in enumerate(papers, 1):
        title = paper.get("title", "No title")[:80]
        abstract_preview = paper.get("abstract", "No abstract")[:100]
        click.echo(f"{i}. {title}")
        click.echo(f"   Abstract: {abstract_preview}...")
        click.echo(f"   Authors: {paper.get('authors', 'Unknown')}")
        click.echo("")




@click.group()
def main():
    """Papers with Code indexing tool with OpenAI embeddings and Upstash Vector"""
    pass

main.add_command(test_rag_ingestion, name="test-full-ingestion")
main.add_command(test_extraction, name="test-extraction")
main.add_command(test_upstash, name="test-upstash")


if __name__ == "__main__":
    main()