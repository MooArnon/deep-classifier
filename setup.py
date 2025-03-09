from setuptools import setup, find_packages
import os

# Read the contents of your requirements.txt file.
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="deep_classifier",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A feature engineering package for deep learning-based asset classification.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/deep_classifier",  # update this URL
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # or your chosen license
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)
