import os
import time
import random
import requests
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class OpenAIEmbeddingFunction:
    """
    Most cost-effective for large indexing.
    Model: text-embedding-3-small
    Cost: $0.02 per 1M tokens (3000 chunks will cost ~$0.05)
    Dimensions: 1536
    """
    def __init__(self, model="text-embedding-3-small"):
        self.model = model
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=OPENAI_API_KEY)
        except ImportError:
            self.client = None

    def name(self):
        return "OpenAIEmbeddingFunction"

    def __call__(self, input):
        return self.embed_documents(input)

    def embed_query(self, text):
        if not self.client:
            return self._fallback_embed([text])[0]
        response = self.client.embeddings.create(input=[text], model=self.model)
        return response.data[0].embedding

    def embed_documents(self, texts):
        if not self.client:
            return self._fallback_embed(texts)
        # OpenAI handles batching automatically for large lists
        response = self.client.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in response.data]

    def _fallback_embed(self, texts):
        """Pure requests fallback if openai lib is missing."""
        url = "https://api.openai.com/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {"input": texts, "model": self.model}
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return [item["embedding"] for item in response.json()["data"]]

class GeminiEmbeddingFunction:
    def __init__(self, model_name="models/gemini-embedding-001"):
        self.model_name = model_name
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        self.genai = genai

    def name(self):
        return "GeminiEmbeddingFunction"

    def __call__(self, input):
        return self.embed_documents(input)

    def embed_query(self, text):
        return self._embed(text, task_type="retrieval_query")

    def embed_documents(self, texts):
        return self._embed(texts, task_type="retrieval_document")

    def _embed(self, content, task_type):
        from google.api_core import exceptions
        max_retries = 10
        for i in range(max_retries):
            try:
                result = self.genai.embed_content(
                    model=self.model_name,
                    content=content,
                    task_type=task_type
                )
                return result['embedding']
            except exceptions.ResourceExhausted:
                wait = (2 ** i) + random.uniform(1, 3)
                time.sleep(wait)
            except Exception as e:
                if "429" in str(e):
                    time.sleep((2 ** i) + 1)
                else:
                    if i == max_retries - 1: raise e
                    time.sleep(2)
        return []
