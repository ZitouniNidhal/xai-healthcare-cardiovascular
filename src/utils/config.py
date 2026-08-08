import os
import yaml

class Config:
    """Config helper class to load and access YAML configurations."""
    def __init__(self, params_path="configs/params.yaml", paths_path="configs/paths.yaml"):
        self.params = self._load_yaml(params_path)
        self.paths = self._load_yaml(paths_path)

    def _load_yaml(self, path):
        if not os.path.exists(path):
            # Fallback to default structure if configs folder hasn't been fully populated yet
            return {}
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def get(self, key, default=None):
        """Get parameter configuration value."""
        return self.params.get(key, default)

    def get_path(self, key, default=None):
        """Get path configuration value."""
        return self.paths.get(key, default)
