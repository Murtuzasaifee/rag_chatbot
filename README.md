# RAG Chatbot for Academic Papers

A Retrieval-Augmented Generation (RAG) chatbot that extracts, indexes, and enables conversational search of academic papers. This project uses OpenAI embeddings and Upstash Vector database to create a powerful research assistant for academic literature.

## 🌟 Features

- **Paper Extraction**: Search and extract papers from Papers With Code API
- **Smart Chunking**: Split papers into semantic chunks for better retrieval
- **Vector Embeddings**: Generate embeddings using OpenAI's models
- **Vector Database**: Store and query embeddings using Upstash Vector
- **Conversational Interface**: Interact with research papers through a natural language interface
- **Containerized Deployment**: Easy deployment with Docker

## 📋 Prerequisites

- Python 3.11+
- OpenAI API key
- Upstash Vector database account
- Docker (optional, for containerized deployment)

## 🚀 Installation

### Option 1: Local Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd rag_chatbot
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -e .
   ```

### Option 2: Docker Installation

1. Build the Docker image:
   ```bash
   docker build -t rag-chatbot .
   ```

2. Run the container:
   ```bash
   docker run -d --name streamlit-rag-chatbot -p 8501:8501  -e OPENAI_API_KEY="$(grep OPENAI_API_KEY .env | cut -d '=' -f2)" -e UPSTASH_VECTOR_REST_URL="$(grep UPSTASH_VECTOR_REST_URL .env | cut -d '=' -f2)"   -e UPSTASH_VECTOR_REST_TOKEN="$(grep UPSTASH_VECTOR_REST_TOKEN .env | cut -d '=' -f2)"  rag-chatbot
   ```
   or
   
   ```bash
   docker run -d --name streamlit-rag-chatbot -p 8501:8501 --env-file .env streamlit-rag-chatbot
   ```
   or

   ```bash
   docker run -d --name streamlit-rag-chatbot -p 8501:8501 rag-chatbot
   ```

## ⚙️ Configuration

Create a `.env` file in the project root with the following variables:

```
OPENAI_API_KEY=your-openai-api-key
UPSTASH_VECTOR_REST_URL=your-upstash-vector-url
UPSTASH_VECTOR_REST_TOKEN=your-upstash-vector-token
```

## 📊 Usage

### Indexing Papers

To search for papers and index them in the vector database:

```bash
python -m src.rag_store.index_papers index-papers --query "attention mechanism" --max_papers 10
```

Options:
- `--query`: Search query for papers (required)
- `--batch_size`: Batch size for indexing (default: 32)
- `--max_papers`: Maximum number of papers to extract (default: 5)
- `--max_chunks`: Maximum number of text chunks to index (default: None)
- `--embedding_model`: OpenAI embedding model (default: "text-embedding-3-small")

### Running the Chatbot Interface

To start the Streamlit web interface:

```bash
streamlit run main.py
```

The interface will be available at http://localhost:8501

## 🧩 Project Structure

```
rag_chatbot/
├── src/
│   ├── prompts/
│   │   ├── __init__.py
│   │   └── rag_prompt.py       # Prompt templates for RAG
│   ├── rag_store/
│   │   ├── __init__.py
│   │   ├── extraction.py       # Paper extraction functionality
│   │   ├── index_papers.py     # Paper indexing CLI
│   │   ├── indexing.py         # Document processing and chunking
│   │   └── embeddings.py       # Embedding generation and storage
│   └── __init__.py
├── .env                        # Environment variables (not in repo)
├── .gitignore
├── Dockerfile                  # Docker configuration
├── README.md                   # This file
├── main.py                     # Streamlit application
├── pyproject.toml              # Project metadata and dependencies
└── requirements.txt            # Dependencies for Docker
```

## 🔄 How It Works

1. **Paper Extraction**: The system searches for academic papers based on your query using the Papers With Code API.
2. **Document Processing**: Papers are processed and their abstracts are extracted.
3. **Chunking**: Long abstracts are split into smaller, semantically meaningful chunks.
4. **Embedding Generation**: Each chunk is converted into a vector embedding using OpenAI's embedding models.
5. **Vector Storage**: Embeddings are stored in Upstash Vector database for efficient retrieval.
6. **Query Processing**: When you ask a question, it's converted to an embedding and used to find relevant paper chunks.
7. **Response Generation**: The most relevant chunks are used as context for generating a comprehensive answer.

## 🛠️ Advanced Configuration

### Customizing the RAG Prompt

You can modify the RAG prompt template in `src/prompts/rag_prompt.py` to change how the system responds to queries.

### Embedding Models

The system supports different OpenAI embedding models. You can specify the model using the `--embedding_model` parameter when indexing papers.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- [Papers With Code](https://paperswithcode.com/) for providing the API to access academic papers
- [OpenAI](https://openai.com/) for the embedding and language models
- [Upstash Vector](https://upstash.com/) for the vector database
- [Streamlit](https://streamlit.io/) for the web interface framework