import setuptools

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="simplerag",
    version="0.0.1",
    author="Samarth Sarin",
    author_email="sarin.samarth07@gmail.com",
    description="This package will help you talk to your data using Retrieval Augmented Generation (RAG)",
    long_description=long_description,
    long_description_content_type = "text/markdown",
    url = "package URL",
    project_urls = {
        "Bug Tracker": "package issues URL",
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    install_requires=["<transformers>",
    "<accelerate>",
    "<SentencePiece>",
    "<langchain>;python_version<'<0.0.165>'",
    "<chromadb>;python_version<'<0.3.22>'",
    "<typing-inspect>;python_version<'<0.8.0>'",
    "<typing_extensions>;python_version<'<4.5.0>'"],
    python_requires=">=3.6"
)