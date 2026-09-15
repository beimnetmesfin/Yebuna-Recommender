from app import config

def test_project_paths():
    assert config.data_dir.exists()
    assert config.models_dir.exists()

def test_file():
    assert config.products_file.exists()
    assert config.user_events_file.exists()
      

