import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

MODEL = SentenceTransformer('all-MiniLM-L6-v2')
EMBEDDING_DIM = MODEL.get_sentence_embedding_dimension()


class VectorStore:
    def __init__(self, index_store, tweet_store, explanation_store, sentiment_store, rebuild_if_exists=False):
        if not (os.path.exists(index_store) and os.path.exists(tweet_store) and os.path.exists(explanation_store)) or rebuild_if_exists:
            tweet_dir = os.path.dirname(tweet_store)
            explanation_dir = os.path.dirname(explanation_store)
            sentiment_dir = os.path.dirname(sentiment_store)
            os.makedirs(tweet_dir, exist_ok=True)
            os.makedirs(explanation_dir, exist_ok=True)
            os.makedirs(sentiment_dir, exist_ok=True)

            print("Building new Vector Store.")
            # create empty stores
            self.texts = []
            self.explanations = []
            self.sentiments = []

            np.save(tweet_store, np.array([], dtype=object))
            np.save(explanation_store, np.array([], dtype=object))
            np.save(sentiment_store, np.array([], dtype=object))

            self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
            faiss.write_index(self.index, index_store)
        else:
            print("Loading existing Vector Store.")
            self.index = faiss.read_index(index_store)
            self.texts = np.load(tweet_store, allow_pickle=True).tolist()
            self.explanations = np.load(explanation_store, allow_pickle=True).tolist()
            self.sentiments = np.load(sentiment_store, allow_pickle=True).tolist()

        self.index_store = index_store
        self.tweet_store = tweet_store
        self.explanation_store = explanation_store
        self.sentiment_store = sentiment_store


    def build_from(self, texts, explanations, sentiments):
        raw_embeddings = np.array(MODEL.encode(texts), dtype=np.float32)
        norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
        embeddings = raw_embeddings / norms   # Normalize each embedding
        self.index = faiss.IndexFlatIP(embeddings.shape[1])  # Create inner-product index
        self.index.add(embeddings)
        
        faiss.write_index(self.index, self.index_store)
        np.save(self.text_store, np.array(texts, dtype=object))
        np.save(self.explanation_store, np.array(explanations, dtype=object))
        np.save(self.sentiment_store, np.array(sentiments, dtype=object))

        self.texts = texts
        self.explanations = explanations
        self.sentiments = sentiments

    
    def find_nearest(self, text, k=1):
        raw_embedding = np.array(MODEL.encode([text]), dtype=np.float32)
        norm = np.linalg.norm(raw_embedding, axis=1, keepdims=True)
        embedding = raw_embedding / norm
        distances, indices = self.index.search(embedding, k=k)

        if len(indices) == 0 or len(indices[0]) == 0 or indices[0][0] == -1:
            return (None, None, None, None, None, None)
        
        nearest_texts = [self.texts[i] for i in indices[0]]
        nearest_explanations = [self.explanations[i] for i in indices[0]]
        nearest_sentiments = [self.sentiments[i] for i in indices[0]]
        nearest_embeddings = [self.index.reconstruct(int(i)) for i in indices[0]]

        
        return (distances[0], nearest_texts, nearest_explanations, nearest_sentiments, embedding, nearest_embeddings)

    
    def store_new_text(self, text, explanation, sentiment, similarity_threshold=0.75):
        raw_embedding = np.array(MODEL.encode([text]), dtype=np.float32)
        norm = np.linalg.norm(raw_embedding, axis=1, keepdims=True)
        new_embedding = raw_embedding / norm
        distances, indices = self.index.search(new_embedding, k=1)  # Top-1 closest match

        if len(self.texts) != 0 and len(distances) != 0 and distances[0][0] > similarity_threshold:
            return (True, distances[0][0], self.texts[indices[0][0]], self.explanations[indices[0][0]], self.sentiments[indices[0][0]])
        else:
            self.texts.append(text)
            self.index.add(new_embedding)
            self.explanations.append(explanation)
            self.sentiments.append(sentiment)

        faiss.write_index(self.index, self.index_store)
        np.save(self.tweet_store, np.array(self.texts, dtype=object))
        np.save(self.explanation_store, np.array(self.explanations, dtype=object))
        np.save(self.sentiment_store, np.array(self.sentiments, dtype=object))

        return (False, None, None, None, None)
    
    def __len__(self):
        return len(self.texts)
