import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from backend.src.config import Config
from backend.src.utils import get_logger

logger = get_logger("rag_pipeline")

class RAGPipeline:
    """Manages document chunking, indexing, storage, and semantic search retrieval."""
    
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
        self.db_path = Config.DB_PATH
        self.docs_dir = Config.DOCS_DIR
        self.vector_store = None
        
        # Initialize index
        self.get_or_create_index()

    def get_or_create_index(self):
        """Loads index from disk if it exists, otherwise builds a new one from /docs."""
        index_file = self.db_path / "index.faiss"
        if index_file.exists():
            try:
                logger.info(f"Loading existing FAISS index from: {self.db_path}")
                self.vector_store = FAISS.load_local(
                    str(self.db_path), 
                    self.embeddings,
                    distance_strategy=DistanceStrategy.COSINE,
                    allow_dangerous_deserialization=True
                )
                logger.info("FAISS index loaded successfully.")
                return
            except Exception as e:
                logger.error(f"Failed loading FAISS index: {str(e)}. Rebuilding index.")
        
        self.rebuild_index()

    def rebuild_index(self):
        """Scans the /docs directory, processes documents, builds and saves a new FAISS index."""
        logger.info(f"Scanning documents directory: {self.docs_dir}")
        if not self.docs_dir.exists():
            self.docs_dir.mkdir(parents=True, exist_ok=True)
            logger.warning(f"Docs directory was missing. Created it at: {self.docs_dir}")
            
        documents = []
        for file_path in self.docs_dir.iterdir():
            if file_path.is_file():
                ext = file_path.suffix.lower()
                logger.info(f"Loading file: {file_path.name}")
                try:
                    if ext in [".txt", ".md"]:
                        loader = TextLoader(str(file_path), encoding="utf-8")
                        documents.extend(loader.load())
                    elif ext == ".pdf":
                        loader = PyPDFLoader(str(file_path))
                        documents.extend(loader.load())
                    else:
                        logger.warning(f"Skipping unsupported file format: {file_path.name}")
                except Exception as e:
                    logger.error(f"Error loading {file_path.name}: {str(e)}")

        if not documents:
            logger.warning("No documents found in docs directory. Creating an empty fallback index.")
            # Create a small dummy document so the vector store is initialized
            from langchain_core.documents import Document
            documents = [Document(page_content="SupportSense AI Knowledge Base placeholder.", metadata={"source": "system_init"})]

        # Text Splitting
        logger.info(f"Splitting {len(documents)} document pages...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks from source documents.")
        
        # Enforce metadata structure: source, page, section
        for chunk in chunks:
            source_path = chunk.metadata.get("source", "unknown")
            # clean source path to display only filename
            chunk.metadata["source"] = Path(source_path).name
            if "page" not in chunk.metadata:
                chunk.metadata["page"] = 1
            # Infer section name from content if possible
            first_line = chunk.page_content.split("\n")[0]
            if first_line.startswith("#"):
                chunk.metadata["section"] = first_line.replace("#", "").strip()
            else:
                chunk.metadata["section"] = "General Content"

        # Create FAISS database
        logger.info("Generating embeddings and building FAISS index...")
        try:
            self.vector_store = FAISS.from_documents(
                chunks, 
                self.embeddings,
                distance_strategy=DistanceStrategy.COSINE
            )
            self.vector_store.save_local(str(self.db_path))
            logger.info(f"FAISS index successfully saved to: {self.db_path}")
        except Exception as e:
            logger.critical(f"Failed to create FAISS index: {str(e)}")
            raise e

    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves the top k document chunks matching the query.
        
        Returns:
            List[Dict]: Containing "page_content", "metadata", and "score" (0.0 to 1.0).
        """
        if not self.vector_store:
            logger.warning("Vector store is not initialized. Returning empty results.")
            return []

        logger.info(f"Querying vector database for: '{query[:50]}...'")
        try:
            # similarity_search_with_score returns Tuple[Document, float]
            # Since DistanceStrategy is COSINE, the score is distance = 1 - cosine_similarity.
            # Thus similarity score = 1 - distance.
            results = self.vector_store.similarity_search_with_score(query, k=k)
            
            retrieved_chunks = []
            for doc, distance in results:
                # Convert distance to normalized similarity score [0.0, 1.0]
                similarity = 1.0 - distance
                similarity = max(0.0, min(1.0, similarity))
                
                # Check for extreme out-of-distribution cases
                if doc.metadata.get("source") == "system_init":
                    continue
                    
                retrieved_chunks.append({
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": round(similarity, 3)
                })
                
            logger.info(f"Retrieved {len(retrieved_chunks)} relevant chunks.")
            return retrieved_chunks
        except Exception as e:
            logger.error(f"Error during retrieval: {str(e)}")
            return []
