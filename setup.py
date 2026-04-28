#!/usr/bin/env python3
"""
Setup script for AI Health System
Medical image analysis and processing tools
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "AI Health System - Medical Image Analysis"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="ai-health-system",
    version="1.0.0",
    description="AI-powered medical image analysis system for chest X-rays",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="AI Health System Team",
    author_email="team@aihealthsystem.com",
    url="https://github.com/yourusername/ai-health-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ai-health=src.main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="medical imaging, chest x-ray, pneumonia detection, deep learning, healthcare",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/ai-health-system/issues",
        "Source": "https://github.com/yourusername/ai-health-system",
        "Documentation": "https://ai-health-system.readthedocs.io/",
    },
)
