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
    maintainer="Hua Cheng",
    maintainer_email="trernghwhuare@aliyun.com",
    description="A comprehensive Python framework for analyzing complex networks using graph theory metrics with built-in visualization and reproducible research capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/trernghwhuare/metrics-analysis-project",
    project_urls={
        "Bug Reports": "https://github.com/trernghwhuare/metrics-analysis-project/issues",
        "Source": "https://github.com/trernghwhuare/metrics-analysis-project",
        "Documentation": "https://github.com/trernghwhuare/metrics-analysis-project#readme",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Natural Language :: English",
    ],
    keywords="network-analysis, graph-theory, complex-networks, data-science, reproducible-research, visualization",
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
        ],
        "docs": [
            "jupyter-book>=0.12.0",
            "myst-parser>=0.15.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)