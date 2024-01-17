import os
from langchain.embeddings import HuggingFaceInstructEmbeddings
from chromadb.config import Settings
import glob
from langchain.vectorstores import Chroma
from multiprocessing import Pool
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    # ".docx": (Docx2txtLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}

class IntructorEmbeddings:
    def __init__(self, model_name):
        self.model_name = model_name

    def load_embeddings_model(self, device='cpu'):
        instructor_embeddings = HuggingFaceInstructEmbeddings(model_name=self.model_name,
                                                              model_kwargs={"device": device})

        return instructor_embeddings

def does_vectorstore_exist(persist_directory):
    """
    Checks if vectorstore exists
    """
    if os.path.exists(os.path.join(persist_directory, 'index')):
        if os.path.exists(os.path.join(persist_directory, 'chroma-collections.parquet')) and os.path.exists(
                os.path.join(persist_directory, 'chroma-embeddings.parquet')):
            list_index_files = glob.glob(os.path.join(persist_directory, 'index/*.bin'))
            list_index_files += glob.glob(os.path.join(persist_directory, 'index/*.pkl'))
            # At least 3 documents are needed in a working vectorstore
            if len(list_index_files) > 3:
                return True
    return False

class Data:
    def __init__(self, source_directory, chunk_size=500, chunk_overlap=50):
        self.source_directory = source_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_single_document(self,file_path):
        ext = "." + file_path.rsplit(".", 1)[-1]
        if ext in LOADER_MAPPING:
            loader_class, loader_args = LOADER_MAPPING[ext]
            loader = loader_class(file_path, **loader_args)
            return loader.load()

        raise ValueError(f"Unsupported file extension '{ext}'")

    def load_documents(self, source_directory, ignored_files=[]):
        """
        Loads all documents from the source documents directory, ignoring specified files
        """
        all_files = []
        for ext in LOADER_MAPPING:
            all_files.extend(
                glob.glob(os.path.join(source_directory, f"**/*{ext}"), recursive=True)
            )
        filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]

        with Pool(processes=os.cpu_count()) as pool:
            results = []
            with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
                for i, docs in enumerate(pool.imap_unordered(self.load_single_document, filtered_files)):
                    results.extend(docs)
                    pbar.update()

        return results

    def process_documents(self, ignored_files=[]):
        """
        Load documents and split in chunks
        """
        print(f"Loading documents from {self.source_directory}")
        documents = self.load_documents(self.source_directory, ignored_files)
        if not documents:
            print("No new documents to load")
            exit(0)
        print(f"Loaded {len(documents)} new documents from {self.source_directory}")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        texts = text_splitter.split_documents(documents)
        print(f"Split into {len(texts)} chunks of text (max. {self.chunk_size} tokens each)")
        return texts


class DataBase:
    def __init__(self, database_type, database_directory):
        self.database_type = database_type
        self.database_directory = database_directory

    def get_chroma_settings(self,chorma_db_impl = 'duckdb+parquet'):
        CHROMA_SETTINGS = Settings(
            chroma_db_impl=chorma_db_impl,
            persist_directory=self.database_directory,
            anonymized_telemetry=False
        )

        return CHROMA_SETTINGS

    def append_in_db(self, texts, instructor_embeddings, chroma_settings):
        # Update and store locally vectorstore
        print(f"Appending to existing vectorstore at {self.database_directory}")
        db = Chroma(persist_directory=self.database_directory, embedding_function=instructor_embeddings,
                    client_settings=chroma_settings)
        #collection = exisitng_db.get()
        #texts = process_documents([metadata['source'] for metadata in collection['metadatas']])
        print(f"Creating embeddings. May take some minutes...")
        db.add_documents(texts)
        db.persist()
        db = None

    def create_new_db(self, texts, instructor_embeddings, chroma_settings):
        #texts = process_documents()
        print(f"Creating embeddings. May take some minutes...")
        db = Chroma.from_documents(texts, instructor_embeddings, persist_directory=self.database_directory,
                                   client_settings=chroma_settings)

        db.persist()
        db = None



