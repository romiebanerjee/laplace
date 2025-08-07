from setuptools import setup, find_packages

setup(
    name="laplace",  # Required
    version="0.1.0",           # Required
    author="Romie Banerjee",        # Optional
    author_email="romie.banerjee@gmail.com",  # Optional
    description="Laplace-BNN inference",  # Optional
    long_description=open("README.md").read(),  # Optional
    long_description_content_type="text/markdown",  # Optional if using markdown
    url="https://github.com/romiebanerjee/laplace",  # Optional
    packages=find_packages(),  # Required (finds all packages in your project)
    install_requires=[         # Optional list of dependencies
        'numpy>=1.18.0',
        'torch>=2.1',
    ],
    classifiers=[              # Optional metadata
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',   # Optional Python version requirements
)