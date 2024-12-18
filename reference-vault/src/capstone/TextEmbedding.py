import openai
import numpy as np

class TextEmbedding:
    def __init__(self, model="text-embedding-003"):
        # Configure the OpenAI API key
        openai.api_key = 'your-openai-api-key'
        self.model = model

    def get_embedding(self, text: str):
        """
        Gets the embedding of a given text using the OpenAI model.
        """
        response = openai.Embedding.create(input=text, model=self.model)
        return response['data'][0]['embedding']

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculates the cosine similarity between two vectors.
        """
        dot_product = np.dot(a, b)
        magnitude_a = np.linalg.norm(a)
        magnitude_b = np.linalg.norm(b)
        return dot_product / (magnitude_a * magnitude_b)

    def compare_samples(self, samples: list):
        """
        Gets the embeddings of the samples and calculates the cosine similarity.
        """
        embeddings = [self.get_embedding(sample) for sample in samples]
        
        # Calculate cosine similarity for all pairs of embeddings
        similarities = [
            self.cosine_similarity(embeddings[i], embeddings[j]) 
            for i in range(len(embeddings)) 
            for j in range(i+1, len(embeddings))
        ]
        
        return similarities, embeddings
