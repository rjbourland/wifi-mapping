"""Configuration loader for YAML config files."""

import yaml
from pathlib import Path
from typing import Any


CONFIG_DIR = Path(__file__).parent.parent.parent / "configs"


def load_config(name: str) -> dict[str, Any]:
    """Load a YAML configuration file by name.

    Args:
        name: Config filename without path (e.g. 'anchors', 'collection', 'algorithm').
              Also accepts full path.

    Returns:
        Parsed YAML as a dictionary.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
    """
    path = Path(name)
    if not path.suffix:
        path = path.with_suffix(".yaml")

    if not path.is_absolute():
        path = CONFIG_DIR / path

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        return yaml.safe_load(f)


def get_anchors() -> list[dict]:
    """Load anchor positions from config and return as a list of dicts.

    Returns:
        List of anchor dicts with 'id', 'position', 'height', 'hardware', 'ip', 'channel', 'bandwidth'.
    """
    config = load_config("anchors")
    anchors = []
    for anchor_id, info in config.get("anchors", {}).items():
        anchors.append({"id": anchor_id, **info})
    return anchors


def get_room_dimensions() -> dict[str, float]:
    """Load room dimensions from config.

    Returns:
        Dict with 'length_x', 'width_y', 'height_z'.
    """
    config = load_config("anchors")
    return config.get("room", {"length_x": 4.0, "width_y": 4.0, "height_z": 2.5})