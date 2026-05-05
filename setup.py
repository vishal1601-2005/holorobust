from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="holorobust",
    version="0.1.0",
    author="Vishal",
    description="Holographic & Geometric Physics-Informed Robust ML Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vishal1601-2005/holorobust",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.2.0",
        "matplotlib>=3.7.0",
        "h5py>=3.8.0",
        "onnx>=1.14.0",
        "pandas>=2.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    keywords=[
        "machine learning", "physics-informed", "holographic",
        "adversarial robustness", "anomaly detection",
        "high energy physics", "cybersecurity", "PyTorch"
    ],
)
