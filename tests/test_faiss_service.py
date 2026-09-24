import pandas as pd
import numpy as np
import pytest
from app.faiss_service import FaissService

from app.config import test_index

@pytest.fixture
def Faiss_Service():
    return (
        FaissService(dimension=3)
    )

@pytest.fixture
def embeddings():   
    return (np.array(
        [
            [1.0,0.0,0.0],
            [0.0,1.0,0.0],
            [0.0,0.0,1.0],
        ],
        dtype=np.float32
    ))

def test_build_index(
        Faiss_Service,
        embeddings
):
    Faiss_Service.build_index(embeddings)
    assert Faiss_Service.index.ntotal==3

def test_search_returns_results(
        Faiss_Service,embeddings
):
    Faiss_Service.build_index(embeddings)
    query=np.array(
        [[1.0,0.0,0.0]],
        dtype=np.float32
    )
    scores,indices=Faiss_Service.search(
        query,k=2
    )
    assert scores.shape==(1,2)
    assert indices.shape==(1,2)

def test_search_returns_similar_vectors(
        Faiss_Service,embeddings
):
    Faiss_Service.build_index(embeddings)

    query=np.array([[1.0,0.0,0.0]],dtype=np.float32)
    scores,indices=Faiss_Service.search(query,k=3)

    assert indices[0,0]==0
    assert np.isclose(scores[0,0],1.0)

def test_save_and_load_index(Faiss_Service,embeddings):
    Faiss_Service.build_index(embeddings)
    Faiss_Service.save_index(test_index)
    assert test_index.exists()

    new_service=FaissService(dimension=3)
    new_service.load_index(test_index)
    assert new_service.index.ntotal==3
    
