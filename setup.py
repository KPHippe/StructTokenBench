from setuptools import setup, find_packages

setup(
    name="structtokenbench",
    version="0.1.0",
    description="A minimal package for the AminoSeed structure tokenizer from StructTokenBench.",
    author="Your Name",
    author_email="your.email@example.com",
    packages=["structtokenbench"],
    install_requires=[
        "torch",
        "numpy",
        "hydra-core",
        "omegaconf",
        # Add any other dependencies required for inference
        "biotite",
        "biopython",
        "cloudpathlib",
        # "esm"  # If ESM3 is required and available on PyPI
    ],
    python_requires=">=3.8",
    include_package_data=True,
    zip_safe=False,
    url="https://github.com/yourusername/structtokenbench",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
) 