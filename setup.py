from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="network-metrics-package",
    version="0.1.0",
    author="Hua Cheng",
    author_email="trernghwhuare@aliyun.com",
    description="A package for analyzing neural network structures using graph theory metrics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/trernghwhuare/network-analysis-workflow",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "analyze-network-metrics=network_metrics_package.main:main",
        ],
    },
)