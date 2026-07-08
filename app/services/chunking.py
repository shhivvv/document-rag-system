import re
import hashlib
import logging
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.models import DocumentChunk

logger = logging.getLogger("rag_system.chunking")

class DocumentChunker:
    """
    Intelligent document chunking service.
    Splits text by semantic sections to keep headings intact and groups content by paragraphs.
    Uses RecursiveCharacterTextSplitter as a sub-engine for large sections.
    """
    
    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 120):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Standard recursive character text splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def _generate_hash_id(self, content: str) -> str:
        """Generates a stable unique hash identifier for a given string content."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _detect_sections(self, text: str) -> List[Dict[str, str]]:
        """
        Parses text and splits it into structural sections based on heading markers.
        Supported heading markers:
        - Markdown/Text headings: lines starting with #, ##, ### etc.
        - Capitalized headings: short lines (3-80 chars) in ALL CAPS.
        Returns a list of dicts: [{'title': 'Section Name', 'content': 'Section Text'}]
        """
        # Split text into lines
        lines = text.split("\n")
        sections = []
        
        current_section_title = "Introduction"
        current_section_lines = []
        
        # Regex for md headings (e.g. "# Heading Name")
        md_heading_pattern = re.compile(r"^#+\s+(.+)$")
        # Regex for short ALL CAPS lines that could represent a heading
        caps_heading_pattern = re.compile(r"^[A-Z0-9\s\-\,\.\:\'\(\)\&]{3,80}$")

        for line in lines:
            stripped = line.strip()
            if not stripped:
                current_section_lines.append(line)
                continue

            # Check if this line is an MD heading
            md_match = md_heading_pattern.match(stripped)
            is_heading = False
            heading_title = ""
            
            if md_match:
                is_heading = True
                heading_title = md_match.group(1).strip()
            elif caps_heading_pattern.match(stripped) and len(stripped) < 60:
                # Exclude lines that end with standard sentence punctuation to avoid catching normal capitalized sentences
                if not stripped.endswith((".", "?", "!")):
                    is_heading = True
                    heading_title = stripped

            if is_heading:
                # Save previous section if it has content
                if current_section_lines:
                    sections.append({
                        "title": current_section_title,
                        "content": "\n".join(current_section_lines).strip()
                    })
                
                # Start new section
                current_section_title = heading_title
                current_section_lines = [line]  # Keep heading in the text
            else:
                current_section_lines.append(line)

        # Append last section
        if current_section_lines:
            sections.append({
                "title": current_section_title,
                "content": "\n".join(current_section_lines).strip()
            })

        # Filter empty sections
        sections = [s for s in sections if s["content"]]
        if not sections:
            sections = [{"title": "Introduction", "content": text}]
            
        return sections

    def chunk_document(self, loaded_pages: List[Dict[str, Any]]) -> List[DocumentChunk]:
        """
        Processes loaded pages of a document and splits them into DocumentChunks.
        Ensures metadata (filename, page, section_title, chunk_id, document_id, chunk_index) is attached.
        """
        if not loaded_pages:
            return []

        filename = loaded_pages[0]["metadata"]["filename"]
        document_id = self._generate_hash_id(filename)
        
        chunks: List[DocumentChunk] = []
        chunk_idx = 0

        for page_data in loaded_pages:
            page_text = page_data["text"]
            page_num = page_data["metadata"]["page"]
            
            # Detect sections on this page
            sections = self._detect_sections(page_text)
            
            for section in sections:
                sec_title = section["title"]
                sec_content = section["content"]
                
                # Split section content using recursive character splitter
                sub_chunks = self.splitter.split_text(sec_content)
                
                for sub_chunk_text in sub_chunks:
                    sub_chunk_text = sub_chunk_text.strip()
                    if not sub_chunk_text:
                        continue
                    
                    # If the sub-chunk doesn't contain the section title, prepend it to keep context
                    # (only if the splitter divided the text and left the heading behind)
                    chunk_text = sub_chunk_text
                    if sec_title != "Introduction" and sec_title not in chunk_text:
                        # But make sure chunk size limits aren't violated massively.
                        chunk_text = f"[{sec_title}]\n{sub_chunk_text}"

                    # Unique chunk ID based on doc ID, page, index, and content hash
                    content_hash = self._generate_hash_id(f"{document_id}_{page_num}_{chunk_idx}_{chunk_text}")
                    chunk_id = f"doc_{document_id[:8]}_p{page_num}_c{chunk_idx}_{content_hash[:8]}"
                    
                    metadata = {
                        "filename": filename,
                        "page_number": page_num,
                        "section_title": sec_title,
                        "chunk_id": chunk_id,
                        "document_id": document_id,
                        "chunk_index": chunk_idx
                    }
                    
                    doc_chunk = DocumentChunk(
                        id=chunk_id,
                        document_id=document_id,
                        text=chunk_text,
                        metadata=metadata
                    )
                    
                    chunks.append(doc_chunk)
                    chunk_idx += 1

        logger.info(f"Chunked document {filename} into {len(chunks)} chunks.")
        return chunks
