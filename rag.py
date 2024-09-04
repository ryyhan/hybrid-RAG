import os
import faiss
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class RAGSystem:
    def __init__(self, model_path: str, documents_path: str):
        # Load the fine-tuned model and tokenizer
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model.eval()

        # Load the sentence transformer for encoding
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

        # Load and index documents
        self.documents = self.load_documents(documents_path)
        self.index = self.create_index()

    def load_documents(self, documents_path: str) -> List[str]:
        documents = []
        for filename in os.listdir(documents_path):
            if filename.endswith('.txt'):
                with open(os.path.join(documents_path, filename), 'r') as f:
                    documents.extend(f.read().split('\n'))
        return documents

    def create_index(self) -> faiss.IndexFlatL2:
        embeddings = self.encoder.encode(self.documents)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings.astype('float32'))
        return index

    def retrieve(self, query: str, k: int = 5) -> List[str]:
        query_embedding = self.encoder.encode([query])
        _, indices = self.index.search(query_embedding.astype('float32'), k)
        return [self.documents[i] for i in indices[0]]

    def generate(self, prompt: str, max_length: int = 100) -> str:
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        output = self.model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def rag_generate(self, query: str, max_length: int = 200) -> Dict[str, str]:
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(query)

        # Construct the prompt
        context = "\n".join(retrieved_docs)
        prompt = f"Context:\n{context}\n\nQuery: {query}\n\nAnswer:"

        # Generate the answer
        answer = self.generate(prompt, max_length)

        return {
            "query": query,
            "context": context,
            "answer": answer
        }

# Usage example
if __name__ == "__main__":
    model_path = "./results/fine_tuned_gpt2"  
    documents_path = "./documents"  

    rag = RAGSystem(model_path, documents_path)
    
    query = "What are the challenges of exploring Mars?"
    result = rag.rag_generate(query)
    
    print(f"Query: {result['query']}")
    print(f"Context: {result['context']}")
    print(f"Generated Answer: {result['answer']}")