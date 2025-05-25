import os
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import sys
import click

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


def extract_paper_abstracts(query, max_papers, embedding_model):
    
    click.echo(f"Extracting papers matching query: '{query}'")
    click.echo(f"Maximum papers to fetch: {max_papers}")
    click.echo(f"Using OpenAI embedding model: {embedding_model}")
    
    # Extract papers with the specified limit
    papers = extract_papers(query, max_results=max_papers)
    click.echo(f"Extraction complete ✅: ({len(papers)} papers)")
    
    if not papers:
        click.echo("❌ No papers found. Try a different query.")
        return
    
    # Filter papers that have abstracts (required for indexing)
    papers_with_abstracts = [
        paper for paper in papers 
        if paper.get("abstract") and paper.get("abstract").strip()
    ]
    
    click.echo(f"Papers with abstracts: {len(papers_with_abstracts)}")
    
    if not papers_with_abstracts:
        click.echo("❌ No papers with abstracts found. Cannot proceed with indexing.")
        return
    
    return papers_with_abstracts

def split_and_chunk_documents(max_chunks, papers_with_abstracts):
    
    # Create documents from papers
    documents = []
    for paper in papers_with_abstracts:
        # Handle missing fields gracefully
        doc = Document(
            page_content=paper.get("abstract", ""),
            metadata={
                "id": paper.get("id", ""),
                "arxiv_id": paper.get("arxiv_id", ""),
                "url_pdf": paper.get("url_pdf", ""),
                "title": paper.get("title", ""),
                "authors": paper.get("authors", []),
                "published": paper.get("published", ""),
                "url": paper.get("url", ""),
                "paper_url": paper.get("paper_url", ""),
            },
        )
        documents.append(doc)

    click.echo(f"Created {len(documents)} documents")

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", ".", " "],
    )
    
    click.echo("Splitting documents into chunks...")
    splits = text_splitter.split_documents(documents)
    click.echo(f"Created {len(splits)} text chunks")
    
    # Apply chunk limit if specified
    if max_chunks and max_chunks < len(splits):
        splits = splits[:max_chunks]
        click.echo(f"Limited to {len(splits)} chunks")

    if not splits:
        click.echo("❌ No text chunks created. Cannot proceed with indexing.")
        return
    
    return splits
