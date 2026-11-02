import chromadb
import uuid
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from typing import List, Tuple
from dotenv import load_dotenv
import os

load_dotenv()

class VectorStore:
    def __init__(self, collection_name: str = "documents", persist_directory: str = "./data/vector_store"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name = collection_name)
        print(f"Vector store initialized: {collection_name}")
        print(f"Documents in collection: {self.collection.count()}")
    
    def add_documents(self, documents: List[Document]):
        try:
            ids = []
            texts = []
            metadatas = []
            embeddings_list = []
            
            for i, doc in enumerate(documents):
                ids.append(f"doc_{uuid.uuid4().hex[:8]}_{i}")
                texts.append(doc.page_content)
                metadatas.append(doc.metadata)
            embeddings_list = self.embeddings.embed_documents(texts)

            self.collection.add(
                ids = ids,
                documents = texts,
                metadatas = metadatas,
                embeddings = embeddings_list
            )
            print(f"Added {len(documents)} documents. Total: {self.collection.count()}")
    
        except Exception as e:
            print(f"Error adding documents: {e}")
            return 0
        
    def get_all_documents(self) -> List[Document]:
        results = self.collection.get(include = ["documents", "metadatas"])
        docs = []
        for text, metadata in zip(results["documents"], results["metadatas"]):
             docs.append(Document(page_content=text, metadata=metadata))
        return docs
    
    def search(self, question: str, k: int = 4) -> List[Document]:
        question_embed = self.embeddings.embed_query(question)   #Use embeddings from self
        results = self.collection.query(  #Use selfs connection 
            query_embeddings = [question_embed],
            n_results = k
            )         
        docs = []
        for text, metadata in zip(results["documents"][0], results["metadatas"][0]):    #Zip means combine
            docs.append(Document(page_content=text, metadata = metadata))       #Create document objects get the page content and metadata and return it
        return docs
    
    def search_with_scores(self, question: str, k: int = 4) -> List[Tuple[Document,float]]:
        try:
            question_embed = self.embeddings.embed_query(question)
            results = self.collection.query(
                query_embeddings = [question_embed],
                n_results = k,
                include = ["documents", "metadatas", "distances"]
            )
            docs = [] # append tuples with Document object and distance
            for doc, metadata, distance in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
                docs.append((Document(page_content = doc, metadata = metadata), distance))
            return docs
        except Exception as e:
            print(f"Error loading documents {e}")
            return []
    
    def search_with_filter(self, question: str, where: dict, k: int = 4) -> list[Document]:
        #where is a ChromaDB metadata filter, e.g. {"company": "apple"}
        try:
            question_embed = self.embeddings.embed_query(question)
            results = self.collection.query(
                query_embeddings = [question_embed],
                n_results = k,
                where = where
            )
            docs = []
            for text, metadata in zip(results["documents"][0], results["metadatas"][0]):
                docs.append(Document(page_content=text, metadata=metadata))
            return docs
        except Exception as e:
            print(f" Error searching documents {e}")
            return []
    def count(self) -> int:
        try:
            return self.collection.count()
        except Exception as e:
            print(f"Error counting: {e}")
            return 0
    def clear(self):
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        print("Collection cleared")
        
if __name__ == "__main__":
    vs = VectorStore()
    print(f"Chunks in store: {vs.count()}")
    
    results = vs.search("Attention Mechanism", k = 2)
    for doc in results:
        print(doc.page_content[:200])
        print(doc.metadata)
        