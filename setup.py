from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="torchmill",
    version="0.1.0",
    description="Standalone, model-agnostic PyTorch training framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="torchmill contributors",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy",
        "pandas",
        "pydantic>=2.0",
        "pyyaml",
        "tensorboard",
        "tqdm",
        "scikit-learn",
    ],
    extras_require={
        "relbench": ["relbench"],
    },
    entry_points={
        "console_scripts": [
            "torchmill=torchmill.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
