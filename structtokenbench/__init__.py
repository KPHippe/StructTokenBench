"""StructTokenBench: A structure tokenization framework for protein analysis."""

__version__ = "0.1.0"
__author__ = "AminoSeed Team"
__email__ = ""

from .vqvae_model import VQVAEModel
from .protein_chain import WrappedProteinChain

__all__ = [
    "VQVAEModel",
    "WrappedProteinChain",
]