from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from vectorstore import VectorStore
from dotenv import load_dotenv

load_dotenv()

def ingest_pdf(pdf_path, company: str, year: str):
    reader = PdfReader(pdf_path)
    raw_docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            raw_docs.append(Document(page_content=text, metadata={"page": i + 1}))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=500,
    )
    chunks = splitter.split_documents(raw_docs)
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
