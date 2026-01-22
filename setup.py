"""
PFT_GEM: Posterior Fossa Tumor - Geometric Expansion Model

Setup script for installing PFT_GEM package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="pft_gem",
    version="0.1.0",
    author="Stuart McAfee",
    author_email="",
    description="Geometric Expansion Model for Posterior Fossa Tumor Displacement",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/StuMcAfee/PFT_GEM",
    project_urls={
        "Bug Tracker": "https://github.com/StuMcAfee/PFT_GEM/issues",
        "Documentation": "https://github.com/StuMcAfee/PFT_GEM#readme",
    },
    packages=find_packages(exclude=["tests*", "examples*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "full": [
            "nibabel>=4.0.0",
            "matplotlib>=3.5.0",
            "jupyter",
        ],
        "viz": [
            "matplotlib>=3.5.0",
        ],
        "nifti": [
            "nibabel>=4.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
    include_package_data=True,
    package_data={
        "pft_gem": ["py.typed"],
    },
    entry_points={
        "console_scripts": [
            "pft-gem=pft_gem.cli:main",
        ],
    },
)
