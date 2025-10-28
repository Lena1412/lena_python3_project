"""
A small helper module to read/write NumPy arrays to/from a checkpoint file (.npz).

Functions:
- save_checkpoint(path: str, **arrays) -> None
- load_checkpoint(path: str) -> dict[str, numpy.ndarray]

Checkpoint format: compressed NumPy .npz archive. Each named array is stored with its key.
"""
from __future__ import annotations
from typing import Dict
import numpy as np
import os

def save_checkpoint(path: str, /, **arrays: np.ndarray) -> None:
    """
    Save named NumPy arrays to a checkpoint file.

    Parameters
    ----------
    path
        Path to output checkpoint file. Recommended extension: .npz
    **arrays
        Named NumPy arrays to save, e.g., image=arr1, features=arr2

    Raises
    ------
    ValueError
        If no arrays provided.
    OSError
        If the file cannot be written.
    """
    if not arrays:
        raise ValueError("No arrays provided to save_checkpoint.")

    # Ensure directory exists
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    # Convert inputs to numpy arrays if necessary
    normalized = {name: np.asarray(arr) for name, arr in arrays.items()}

    # Use savez_compressed to produce a simple checkpoint bundle
    try:
        np.savez_compressed(path, **normalized)
    except Exception as exc:
        raise OSError(f"Failed to write checkpoint to {path!r}: {exc}") from exc


def load_checkpoint(path: str) -> Dict[str, np.ndarray]:
    """
    Load a checkpoint file saved by save_checkpoint.

    Parameters
    ----------
    path
        Path to checkpoint file (.npz).

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping array-name -> numpy.ndarray

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    OSError
        If the file cannot be read.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    try:
        with np.load(path, allow_pickle=False) as data:
            return {name: data[name].copy() for name in data.files}
    except Exception as exc:
        raise OSError(f"Failed to read checkpoint {path!r}: {exc}") from exc