# Hybrid RAG

This project implements a Retrieval-Augmented Generation (RAG) system using a fine-tuned GPT-2 model. It combines the power of information retrieval with advanced language generation to provide context-aware responses.

## Architecture

![diagram-export-03-09-2024-18_09_22](https://github.com/user-attachments/assets/fb59328f-eae5-4f0b-8510-45eaf7f5c01d)


## Features

- Fine-tuned GPT-2 model for domain-specific text generation
- FAISS-based vector indexing for efficient document retrieval
- FastAPI integration for easy API access



## Usage

1. Start the FastAPI server:
   ```
   uvicorn main:app --reload
   ```

2. Access the API at `http://localhost:8000`

3. Use the `/rag_generate` endpoint for text generation:
   ```
   curl -X POST "http://localhost:8000/rag_generate" -H "Content-Type: application/json" -d '{"query": "Your query here", "max_length": 200}'
   ```

## Project Structure

- `main.py`: FastAPI application setup
- `rag_system.py`: Core RAG system implementation
- `requirements.txt`: List of Python dependencies

