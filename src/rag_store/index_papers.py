import os
import click
from dotenv import load_dotenv
import sys

# Handle imports for both src/ and root directory usage
try:
    from indexing import extract_paper_abstracts, split_and_chunk_documents
    from embeddings import create_embeddings
except ImportError:
    # If running from src/ directory, try parent directory
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
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
def index_papers(query, max_papers, batch_size, max_chunks, embedding_model):
    
    """Index Papers, Create Embeddings and store in Upstash Vector DB"""
    
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
    
    
    

@click.group()
def main():
    """Papers with Code indexing tool with OpenAI embeddings and Upstash Vector"""
    pass

main.add_command(index_papers, name="index-papers")


if __name__ == "__main__":
    main()