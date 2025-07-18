from pathlib import Path

from src.vqvae_model import VQVAEModel
from src.protein_chain import WrappedProteinChain

# If you want to expose the ESM3 tokenizer, import it here as well
# from src.tokenizer import WrappedESM3Tokenizer

# Optionally, you can define convenience functions for model loading, input preparation, etc.

__all__ = [
    "VQVAEModel",
    "WrappedProteinChain",
    # "WrappedESM3Tokenizer",  # Uncomment if needed
] 