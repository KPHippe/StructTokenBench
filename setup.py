#!/usr/bin/env python3
"""Setup script for aminoseed package."""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# Read requirements if available
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

# Requirements based on the README
requirements = [
    "torch>=1.9.0",
    "numpy",
    "lmdb",
    "packaging",
    "hydra-core",
    "lightning",
    "transformers",
    "deepspeed",
    "tensorboard",
    "ipdb",
    "esm",
    "cloudpathlib",
    "pipreqs",
    "lxml",
    "proteinshake",
    "tmtools",
    "tape_proteins",
    "torch-scatter",
    "accelerate",
    "torch_geometric",
    "line_profiler",
    "mini3di",
    "dm-tree",
    "colorcet",
    "ogb==1.2.1",
    "sympy",
    "ase",
    "torch-cluster",
    "jax==0.4.25",
    "tensorflow",
    "biopython",
    "seaborn",
    "omegaconf",
]

setup(
    name="structtokenbench",
    version="0.1.0",
    author="AminoSeed Team",
    description="StructTokenBench: A structure tokenization framework for protein analysis",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/aminoseed",  # Update with actual repo
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest", "black", "flake8", "mypy"],
        "docs": ["sphinx", "sphinx-rtd-theme"],
    },
    package_data={
        "structtokenbench": ["*.yaml", "*.yml", "*.json"],
    },
    entry_points={
        "console_scripts": [
            "structtokenbench-inference=structtokenbench.inference:main",
        ],
    },
    include_package_dirs=True,
    zip_safe=False,
)