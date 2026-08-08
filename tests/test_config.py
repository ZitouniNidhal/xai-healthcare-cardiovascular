import pytest
from src.utils.config import Config

def test_config_load_existing_files(tmp_path):
    # Create temp config files
    params_file = tmp_path / "params.yaml"
    paths_file = tmp_path / "paths.yaml"
    
    # Write sample yaml data
    params_file.write_text("learning_rate: 0.001\nbatch_size: 32\n")
    paths_file.write_text("data_dir: data/raw\nmodel_dir: models/\n")
    
    # Initialize config
    config = Config(params_path=str(params_file), paths_path=str(paths_file))
    
    # Assert values are loaded correctly
    assert config.get("learning_rate") == 0.001
    assert config.get("batch_size") == 32
    assert config.get("non_existent", "default_val") == "default_val"
    
    assert config.get_path("data_dir") == "data/raw"
    assert config.get_path("model_dir") == "models/"
    assert config.get_path("non_existent_path", "default_path") == "default_path"

def test_config_fallback_non_existent_files():
    # Initialize with non-existent paths
    config = Config(params_path="non_existent_params.yaml", paths_path="non_existent_paths.yaml")
    
    # Verify we get empty dict fallbacks
    assert config.params == {}
    assert config.paths == {}
    assert config.get("any_key", "default") == "default"
    assert config.get_path("any_path", "default") == "default"
