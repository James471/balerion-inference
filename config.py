import os

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

_CONFIG_FILENAME = "config.toml"


def _find_config_path():
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(here, _CONFIG_FILENAME)
    if not os.path.exists(candidate):
        raise FileNotFoundError(
            f"No config.toml found at {candidate}. "
            "Copy config.toml.example to config.toml and fill in your machine's paths."
        )
    return candidate


def load_config():
    with open(_find_config_path(), "rb") as f:
        cfg = tomllib.load(f)
    for key in ("models_dir",):
        if not cfg.get(key):
            raise ValueError(f"config.toml is missing required key: {key}")
    return cfg


def get_models_dir(cfg=None):
    cfg = cfg or load_config()
    return cfg["models_dir"]
