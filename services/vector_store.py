"""
services/vector_store.py

Creates and loads FAISS vector database.
"""

from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS

from langchain_core.documents import Document


class VectorStoreService:

    def __init__(self):

        self.embedding = HuggingFaceEmbeddings(

            model_name="sentence-transformers/all-MiniLM-L6-v2"

        )

        self.db_path = Path("data/vector_db")

    # ----------------------------------------

    def create_from_documents(self, documents):

        """
        documents should be

        List[Document]
        """

        vectorstore = FAISS.from_documents(

            documents,

            self.embedding

        )

        vectorstore.save_local(

            str(self.db_path)

        )

        return vectorstore

    # ----------------------------------------

    def load_vectorstore(self):

        if not self.db_path.exists():

            raise FileNotFoundError(

                "Vector database not found."

            )

        return FAISS.load_local(

            str(self.db_path),

            self.embedding,

            allow_dangerous_deserialization=True

        )

    # ----------------------------------------

    def vector_exists(self):

        return self.db_path.exists()

    # ----------------------------------------

    def get_retriever(self):

        vectorstore = self.load_vectorstore()

        return vectorstore.as_retriever(

            search_type="similarity",

            search_kwargs={

                "k": 4

            }

        )

    # ----------------------------------------

    def add_documents(self, new_documents):

        vectorstore = self.load_vectorstore()

        vectorstore.add_documents(new_documents)

        vectorstore.save_local(

            str(self.db_path)

        )

        return vectorstore