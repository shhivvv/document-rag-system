import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import services from dependencies container
from app.dependencies import document_loader, document_chunker, vector_store_service

def ingest_all():
    doc_dir = Path(__file__).resolve().parent / "documents"
    if not doc_dir.exists():
        print(f"Error: Directory '{doc_dir}' does not exist. Run 'data/bootstrap_documents.py' first.")
        return
        
    print(f"Scanning directory: {doc_dir} for documents...")
    files = [f for f in doc_dir.iterdir() if f.is_file() and f.name != ".gitkeep"]
    
    if not files:
        print("No documents found to ingest.")
        return
        
    total_added = 0
    total_skipped = 0
    
    for f_path in files:
        print(f"\nProcessing '{f_path.name}'...")
        try:
            # 1. Load document pages
            pages = document_loader.load_document(f_path)
            # 2. Split pages into semantic chunks
            chunks = document_chunker.chunk_document(pages)
            if not chunks:
                print(f"Warning: No text chunks extracted from '{f_path.name}'. Skipping.")
                continue
            
            # 3. Add chunks to vector store
            result = vector_store_service.add_chunks(chunks)
            total_added += result["added"]
            total_skipped += result["skipped"]
            print(f"Successfully processed '{f_path.name}': Ingested {result['added']} chunks, skipped {result['skipped']} duplicates.")
            
        except Exception as e:
            print(f"Failed to ingest '{f_path.name}': {str(e)}")
            
    print("\n" + "="*50)
    print("Ingestion Run Completed:")
    print(f"- Total chunks added: {total_added}")
    print(f"- Total duplicates skipped: {total_skipped}")
    print(f"- Unique documents indexed: {vector_store_service.get_document_count()}")
    print(f"- Total chunks in database: {vector_store_service.get_chunk_count()}")
    print("="*50)

if __name__ == "__main__":
    ingest_all()
