from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="clica",
    version="0.1.0",
    author="Edan Meyer",
    description="A CLI tool for interactively training a coding agent",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ejmejm/clica",
    packages=find_packages(),
    classifiers=[
        "Intended Audience :: Developers",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
    install_requires=[
        "hydra-core",
        "omegaconf",
        "torch",
        "transformers",
        "gymnasium",
        "litellm",
        "pandas",
        "datasets",
    ],
    entry_points={
        "console_scripts": [
            "clica=scripts.run_cli:run_cli",
        ],
    },
    license="MIT License with Commons Clause",
)
