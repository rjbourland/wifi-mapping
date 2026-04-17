"""Adapters to promote 2D position types to 3D numpy arrays for mapping modules.

The mapping layer expects (N, 3) numpy arrays, but the localization pipeline
produces 2D dataclasses (Position, SmoothedPosition). These adapters bridge
that gap without modifying the upstream dataclasses.
"""

import numpy as np
from datetime import datetime

from ..localization.trilateration import Position
from ..localization.kalman_filter import SmoothedPosition
from ..utils.data_formats import LocalizedPosition


def to_xyz(
    pos,
    z: float = 0.0,
) -> np.ndarray:
    """Promote any position type to a (3,) numpy array [x, y, z].

    Args:
        pos: A Position, SmoothedPosition, LocalizedPosition, or ndarray.
        z: Z-coordinate assigned to 2D types (default 0.0 = floor level).

    Returns:
        np.ndarray of shape (3,).

    Raises:
        TypeError: If pos is an unrecognized type.
    """
    if isinstance(pos, np.ndarray):
        arr = pos.astype(float)
        if arr.shape == (3,):
            return arr
        if arr.shape == (2,):
            return np.array([arr[0], arr[1], z])
        raise ValueError(f"Expected ndarray shape (2,) or (3,), got {arr.shape}")

    if isinstance(pos, LocalizedPosition):
        return to_xyz(pos.position, z=z)

    if isinstance(pos, Position):
        return np.array([pos.x, pos.y, z])

    if isinstance(pos, SmoothedPosition):
        return np.array([pos.x, pos.y, z])

    raise TypeError(f"Cannot promote {type(pos).__name__} to 3D coordinates")


def positions_to_array(
    positions: list,
    z: float = 0.0,
) -> np.ndarray:
    """Convert a list of any position types to an (N, 3) array.

    Args:
        positions: List of Position, SmoothedPosition, LocalizedPosition, or ndarray.
        z: Default z for 2D types.

    Returns:
        (N, 3) float array.
    """
    if not positions:
        return np.empty((0, 3))
    return np.array([to_xyz(p, z=z) for p in positions])