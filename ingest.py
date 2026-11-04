from llama_parse import LlamaParse
from langchain_text_splitters import MarkdownTextSplitter
from langchain_core.documents import Document
from vectorstore import VectorStore
from dotenv import load_dotenv
import os

load_dotenv()

def ingest_pdf(pdf_path, company: str, year: str):
    parser = LlamaParse(result_type="markdown", api_key=os.getenv("LLAMAPARSE_API_KEY"))
    documents = parser.load_data(pdf_path)

    splitter = MarkdownTextSplitter(
        chunk_size=2000,
        chunk_overlap=500,
        keep_separator=True,
        add_start_index=True
    )
    filt_docs = []
    for i, doc in enumerate(documents):
        page_num = doc.metadata.get("page_label", doc.metadata.get("page", i + 1))
        filt_docs.append(Document(page_content=doc.text, metadata={"page": page_num}))

    chunks = splitter.split_documents(filt_docs)
    filtered_chunks = [c for c in chunks if len(c.page_content) > 100]

    for chunk in filtered_chunks:
        chunk.metadata["company"] = company
        chunk.metadata["year"] = year
        chunk.metadata["doc_type"] = "10-K"
        chunk.metadata["source"] = pdf_path

    vs = VectorStore()
    vs.add_documents(filtered_chunks)
    print(f"Saved {len(filtered_chunks)} chunks to ChromaDB")

if __name__ == "__main__":
    ingest_pdf("apple.pdf", "apple", "2024")
