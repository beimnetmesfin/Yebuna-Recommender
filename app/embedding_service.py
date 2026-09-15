import numpy as np 
from sentence_transformers import SentenceTransformer

class EmbeddingService:
    def __init__(self,embedding_model_name:str,device:str| None=None):

        self.model=SentenceTransformer(
            embedding_model_name,
            device=device
        )
    #Create Embeddings

    def encode(self,texts:list[str],batch_size:int=32)->np.ndarray:
        embedding=self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding.astype("float32")

    
