# Placeholder for KB retrieval logic
import re
from sentence_transformers import SentenceTransformer, util
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

class Retriever:
    def __init__(self, kb_text, model_name='all-MiniLM-L6-v2', rag_model_name='google/flan-t5-small'):
        self.kb_text = kb_text
        self.sentences = re.split(r'(?<=[.!?])\s+', kb_text)
        self.model = SentenceTransformer(model_name)
        self.sentence_embeddings = self.model.encode(self.sentences, convert_to_tensor=True)
        # Load a small generative model for RAG
        self.rag_tokenizer = AutoTokenizer.from_pretrained(rag_model_name)
        self.rag_model = AutoModelForSeq2SeqLM.from_pretrained(rag_model_name)
        self.rag_pipe = pipeline('text2text-generation', model=self.rag_model, tokenizer=self.rag_tokenizer)

    def search(self, query, top_k=3, use_rag=True):
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        # Compute cosine similarities
        cos_scores = util.pytorch_cos_sim(query_embedding, self.sentence_embeddings)[0]
        cos_scores = cos_scores.cpu().numpy()  # Move tensor to CPU for NumPy
        top_results = np.argpartition(-cos_scores, range(top_k))[:top_k]
        matches = [self.sentences[idx] for idx in top_results]
        # Filter out empty matches
        matches = [m for m in matches if m.strip()]
        if not matches:
            return "Sorry, I couldn't find relevant information in the knowledge base."
        if use_rag:
            # Compose context for the generator
            context = "\n".join(matches)
            prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
            result = self.rag_pipe(prompt, max_new_tokens=64, do_sample=False)[0]['generated_text']
            return result.strip()
        else:
            return "\n".join(matches)
