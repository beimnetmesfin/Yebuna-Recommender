import pytest
import numpy as np

from app.config import embedding_model_name
from app.embedding_service import EmbeddingService

#creates one reusable DataLoader for the tests instead of writing
#the same three lines repeatedly.
@pytest.fixture(scope="module")
def embedding_service():
    return EmbeddingService(
        embedding_model_name
    )

def test_embedding_dimesion(embedding_service):
    assert embedding_service.dimension==384

def test_encode(embedding_service):
    texts=[
        "Ethiopia coffee premuim Arabica",
        "Tradditional handmade Ethiopia Basket"
    ]
    embeddings=embedding_service.encode(texts)
    assert isinstance(embeddings,np.ndarray)
    assert embeddings.dtype==np.float32
    assert embeddings.shape==(2,384)

def test_encode_empty_texts(embedding_service):
    with pytest.raises(ValueError):
        embedding_service.encode([])
