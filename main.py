from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag import RAGSystem

app = FastAPI()

# Initialize the RAG system
model_path = "./results/fine_tuned_gpt2"  
documents_path = "./documents"  
rag = RAGSystem(model_path, documents_path)

class RAGRequest(BaseModel):
    query: str
    max_length: int = 200

@app.post("/rag_generate")
async def rag_generate(request: RAGRequest):
    try:
        result = rag.rag_generate(request.query, request.max_length)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to the RAG-enhanced GPT-2 text generation API"}