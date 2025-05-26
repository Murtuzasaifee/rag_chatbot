# APP LEVEL COMMANDS

```
python src/rag_store/index_papers.py index-papers --query "attention mechanism" --max_papers 20

streamlit run main.py --theme.primaryColor "#135aaf"

python src/rag_store/rag_ingestion_testing.py test-full-ingestion --query "attention mechanism" --max_papers 20

python src/rag_store/rag_ingestion_testing.py test-upstash

python src/rag_store/rag_ingestion_testing.py test-extraction --query "attention mechanism" --max_papers 20

python src/rag_store/rag_prediction_testing.py

```