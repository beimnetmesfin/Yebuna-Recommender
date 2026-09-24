import pandas as pd
import numpy as np
import pytest

from app.user_profile_vector import UserProfileBuilder

@pytest.fixture
def builder():
    return UserProfileBuilder(
        embedding_dimension=3
    )

@pytest.fixture

def item_embeddings():
    return np.array(
        [
            [1.0,0.0,0.0],
            [0.0,1.0,0.0],
            [0.0,0.0,1.0]
        ],
        dtype=np.float32
    )

@pytest.fixture
def user_events():
    return (
        pd.DataFrame(
            {
                "user_id":[1,1],
                "item_id":[0,1],
                "event_type":['click',"view"],
                "recency_rank":[1,2]           
            }
        )
        )
def test_buider_profile_returns_vector(
        builder,
        item_embeddings,
        user_events
):
    profile=builder.build_profile(
        user_events,
        item_embeddings,
        user_id=1
    )
    assert isinstance(profile,np.ndarray)

def test_profile_has_correct_dimension(
        builder,
        item_embeddings,
        user_events
):
    profile=builder.build_profile(
        user_events,
        item_embeddings,
        user_id=1
    )
    assert profile.shape==(3,)

def test_profile_is_float(
        builder,
        item_embeddings,
        user_events
):
    profile=builder.build_profile(
        user_events,
        item_embeddings,
        user_id=1
    )
    assert profile.dtype==np.float32

def test_profile_is_norimalized(
        builder,
        item_embeddings,
        user_events
):
    profile=builder.build_profile(
        user_events,
        item_embeddings,
        user_id=1
    )
    norm=np.linalg.norm(profile)
    assert np.isclose(norm,1.0)
