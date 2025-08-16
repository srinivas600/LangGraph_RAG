
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from config.settings import Config
import os
from typing import List

# Custom embedding class for models requiring specific prefixes (like e5 and bge)
class PrefixedEmbeddings(HuggingFaceEmbeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Adds the 'passage: ' prefix to each document before embedding.
        This is a common requirement for models like e5 and bge for retrieval tasks.
        """
        # The BGE model instructions are to use this for passage embeddings.
        prefixed_texts = [f"passage: {text}" for text in texts]
        return super().embed_documents(prefixed_texts)

    def embed_query(self, text: str) -> List[float]:
        """
        Adds the 'query: ' prefix to the query before embedding.
        The BGE model instructions specify this for query embeddings.
        """
        prefixed_text = f"query: {text}"
        return super().embed_query(prefixed_text)

class ChromaVectorStore:
    def __init__(self):
        # Initialize embedding model using the custom class and the model from config
        self.embedding_model = PrefixedEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'} # Or 'cuda' if you have a GPU
        )

        # Initialize Chroma vector store
        self.vector_store = Chroma(
            collection_name=Config.COLLECTION_NAME,
            embedding_function=self.embedding_model,
            persist_directory=Config.CHROMA_PERSIST_DIRECTORY
        )

    def add_documents(self, documents, metadatas=None, ids=None):
        """
        Add documents to the vector store.
        """
        if not isinstance(documents, list) or not all(isinstance(doc, Document) for doc in documents):
            raise ValueError("documents must be a list of LangChain Document objects.")

        self.vector_store.add_documents(documents=documents, ids=ids)
        self.vector_store.persist()

    def similarity_search(self, query, k=5):
        """Search for similar documents and return them in the desired format."""
        results_with_scores = self.vector_store.similarity_search_with_score(query, k=k)

        formatted_documents = []
        for doc, score in results_with_scores:
            formatted_documents.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'score': score
            })
        return formatted_documents

    def get_collection_count(self):
        """Get the number of documents in the collection"""
        return self.vector_store._collection.count()

    def add_documents_with_embeddings(self, page_contents: list, metadatas: list, embeddings: list, ids: list = None):
        """
        Inject documents with pre-computed embeddings directly into the Chroma collection.
        """
        if not (isinstance(page_contents, list) and isinstance(metadatas, list) and isinstance(embeddings, list)):
            raise ValueError("page_contents, metadatas, and embeddings must be lists.")
        
        if len(page_contents) != len(metadatas) or len(page_contents) != len(embeddings):
            raise ValueError("The lengths of page_contents, metadatas, and embeddings must be the same.")

        self.vector_store._collection.add(
            embeddings=embeddings,
            documents=page_contents,
            metadatas=metadatas,
            ids=ids
        )
        self.vector_store.persist()
