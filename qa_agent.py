import warnings
warnings.filterwarnings("ignore", message="libmagic is unavailable")

import os
import warnings
warnings.filterwarnings("ignore", message="libmagic is unavailable")

import os
import logging
import time
import traceback
from typing import Optional
from requests.exceptions import JSONDecodeError
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader  # retained if needed elsewhere
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
# from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings, HuggingFaceEmbeddings  # (commented out per request to avoid HuggingFace)
# from langchain_community.llms import HuggingFaceHub  # (commented out)
from langchain_openai import ChatOpenAI  # For Perplexity (OpenAI-compatible)
from dotenv import load_dotenv
logging.basicConfig(level=logging.INFO)


load_dotenv()

# Constants
DATA_DIR = "data"
VECTOR_STORE_PATH = "vector_store"
EMBEDDING_DIM = 512  # dimension for local hashing embeddings
# HF related constants commented out
# EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # (unused now)
# HF_TOKEN_ENV = "HUGGINGFACEHUB_API_TOKEN"
# LLM_MODEL = "google/flan-t5-large"
PPLX_MODEL = "sonar-small-online"  # default Perplexity model (updated to valid name)
PPLX_API_ENV = "PPLX_API_KEY"
PPLX_MODEL_ENV = "PPLX_MODEL"  # optional override via .env

class LocalHashEmbeddings:
    """Simple local embedding using scikit-learn HashingVectorizer (no HuggingFace dependency).

    Produces fixed-size dense vectors suitable for FAISS.
    """
    def __init__(self, n_features: int = EMBEDDING_DIM):
        from sklearn.feature_extraction.text import HashingVectorizer
        self.n_features = n_features
        self.vectorizer = HashingVectorizer(
            n_features=n_features,
            norm='l2',
            alternate_sign=False,
            analyzer='word'
        )

    def _to_list(self, matrix):
        # matrix is sparse; convert to dense list of lists (could optimize for large scale)
        return matrix.toarray().tolist()

    def embed_documents(self, texts):
        return self._to_list(self.vectorizer.transform(texts))

    def embed_query(self, text: str):
        return self._to_list(self.vectorizer.transform([text]))[0]

    # Make instance callable so FAISS can treat it as embedding_function if it stored it that way
    def __call__(self, text: str):  # pragma: no cover
        return self.embed_query(text)

def _build_embeddings():
    print("Using LocalHashEmbeddings (HashingVectorizer, no HuggingFace).")
    return LocalHashEmbeddings(n_features=EMBEDDING_DIM)


def create_vector_store():
    """
    Loads documents, splits them, creates embeddings in batches, and saves
    them to a FAISS vector store to avoid network connection issues.
    """
    print("Starting document loading process...")
    print(f"Looking for files in {os.path.abspath(DATA_DIR)}...")
    
    # Check if data directory exists and has files
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: Data directory {DATA_DIR} does not exist!")
        return
    
    # Count files to process
    file_count = 0
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith('.txt'):
                file_count += 1
    
    print(f"Found {file_count} text files to process.")
    
    if file_count == 0:
        print("No text files found. Please make sure data has been downloaded.")
        return
        
    # Load documents with custom tolerant loader
    print("Loading documents with tolerant multi-encoding reader...")
    documents = []
    failed_files = []
    encodings_to_try = ["utf-8", "cp1252", "latin-1"]
    processed = 0
    for root, _, files in os.walk(DATA_DIR):
        for fname in files:
            if not fname.lower().endswith('.txt'):
                continue
            fpath = os.path.join(root, fname)
            text = None
            for enc in encodings_to_try:
                try:
                    with open(fpath, 'r', encoding=enc, errors='strict') as fh:
                        text = fh.read()
                    break
                except Exception:
                    continue
            if text is None:
                # Last resort: binary read and ignore undecodable bytes
                try:
                    with open(fpath, 'rb') as fb:
                        raw = fb.read()
                    text = raw.decode('utf-8', errors='ignore')
                    if not text.strip():
                        raise ValueError("Empty after decode ignore")
                except Exception:
                    failed_files.append(fpath)
                    continue
            documents.append(Document(page_content=text, metadata={"source": fpath}))
            processed += 1
            if processed % 200 == 0:
                print(f"  Loaded {processed} documents so far...")
    print(f"Successfully loaded {len(documents)} documents. Skipped {len(failed_files)} problematic file(s).")
    if failed_files:
        print("First 5 failed files:")
        for ff in failed_files[:5]:
            print("  -", ff)
    if not documents:
        print("No documents could be loaded. Aborting.")
        return

    # Split documents into chunks (slightly larger chunks to reduce total embeddings)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    if not texts:
        print("No text chunks produced. Aborting.")
        return

    # Initialize embeddings (with fallback)
    embeddings = _build_embeddings()  # returns LocalHashEmbeddings

    # --- Batch processing to avoid connection errors ---
    batch_size = 128  # larger batch size for efficiency (local only)
    remote_failed = False  # retained for compatibility
    
    # Check if a partial vector store exists to resume
    if os.path.exists(VECTOR_STORE_PATH):
        print("Existing vector store found. Resuming...")
        vector_store = FAISS.load_local(
            VECTOR_STORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("Loaded existing store. Now adding any new documents.")
    else:
        print("No existing vector store found. Creating a new one.")
        vector_store = None

    total_batches = (len(texts) + batch_size - 1) // batch_size
    for i in range(0, len(texts), batch_size):
        current_batch_num = (i // batch_size) + 1
        batch_texts = texts[i:i + batch_size]
        
        print(f"--- Processing batch {current_batch_num}/{total_batches} ---")
        
        try:
            if vector_store is None:
                # First batch creates the store
                print("Creating store with the first batch...")
                vector_store = FAISS.from_documents(batch_texts, embeddings)
                print("Vector store created successfully.")
            else:
                # Subsequent batches add to the existing store
                vector_store.add_documents(batch_texts)
                print(f"Batch {current_batch_num} added successfully.")

            # Save progress after each successful batch
            vector_store.save_local(VECTOR_STORE_PATH)
            print(f"Progress saved after batch {current_batch_num}.")

        except JSONDecodeError:
            print("JSONDecodeError encountered unexpectedly (remote embeddings disabled). Aborting batch.")
            return
        except Exception as e:
            print(f"\n---!!! AN ERROR OCCURRED ON BATCH {current_batch_num} !!!---")
            print("Full error traceback:")
            print("-------------------------------------------------")
            traceback.print_exc()
            print("-------------------------------------------------")
            print("Saving any progress made before this error.")
            if vector_store:
                vector_store.save_local(VECTOR_STORE_PATH)
                print("Progress saved. You can re-run the script to resume.")
            else:
                print("No progress to save as the first batch failed.")
            return

        # Pause between batches to be less aggressive on the network
        time.sleep(0.3)
    
    print("\nVector store creation complete and saved successfully.")
    return vector_store

def _build_llm() -> Optional[object]:
    """Build Perplexity LLM only with model fallback sequence.

    Tries an override model from env (PPLX_MODEL) first, then fallback list.
    Returns None if all attempts fail.
    """
    pplx_key = os.getenv(PPLX_API_ENV, "").strip().strip('"')
    if not pplx_key:
        print("Perplexity API key not set; LLM disabled.")
        return None

    override_model = os.getenv(PPLX_MODEL_ENV, "").strip().strip('"')
    tried = set()
    candidates = []
    if override_model:
        candidates.append(override_model)
    # Common currently valid Perplexity model identifiers (update as needed)
    candidates.extend([
        PPLX_MODEL,
        "sonar-medium-online",
        "sonar-large-online",
        "pplx-70b-online",
        "pplx-7b-online",
    ])
    # Deduplicate while preserving order
    unique_candidates = []
    for m in candidates:
        if m and m not in unique_candidates:
            unique_candidates.append(m)

    for model_name in unique_candidates:
        if model_name in tried:
            continue
        tried.add(model_name)
        try:
            print(f"Attempting Perplexity model: {model_name}")
            llm = ChatOpenAI(
                base_url="https://api.perplexity.ai",
                api_key=pplx_key,
                model=model_name,
                temperature=0.1,
                max_tokens=512,
            )
            # Minimal probe
            llm.invoke([("human", "ping")])  # ChatOpenAI accepts list of messages
            print(f"Perplexity model '{model_name}' initialized successfully.")
            return llm
        except Exception as e:
            msg = str(e).lower()
            if "invalid model" in msg or "not found" in msg:
                print(f"Model '{model_name}' invalid or unavailable; trying next...")
                continue
            print(f"Model '{model_name}' failed ({e}); trying next.")
            continue

    print("All Perplexity model attempts failed; falling back to retrieval-only mode.")
    return None


def load_qa_chain(retriever_k: int = 4):
    """Loads vector store and returns a RetrievalQA chain (with fallback behavior)."""
    if not os.path.exists(VECTOR_STORE_PATH):
        raise FileNotFoundError("Vector store not found. Build it first.")
    embeddings = _build_embeddings()
    vector_store = FAISS.load_local(
        VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True
    )
    llm = _build_llm()

    if llm is None:
        # Lightweight fallback: return a shim object mimicking RetrievalQA interface
        class FallbackQA:
            def __init__(self, vs):
                self.vs = vs
            def __call__(self, inputs):
                q = inputs.get("query") or inputs.get("question") or ""
                docs = self.vs.similarity_search(q, k=retriever_k)
                combined = "\n---\n".join(d.page_content[:500] for d in docs)
                answer = (
                    "(Fallback) Unable to use LLM. Showing top retrieved snippets:\n" + combined
                )
                return {"result": answer, "source_documents": docs}
        return FallbackQA(vector_store)

    retriever = vector_store.as_retriever(search_kwargs={"k": retriever_k})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    return qa_chain


def smoke_diagnose(sample_query: str = "Summarize Apple 10-K risk factors"):
    """Run a diagnostic to help debug smoke test failures."""
    print("--- Smoke Diagnose ---")
    print(f"Vector store path exists: {os.path.isdir(VECTOR_STORE_PATH)}")
    if not os.path.isdir(VECTOR_STORE_PATH):
        print("Vector store missing.")
        return
    try:
        embeddings = _build_embeddings(prefer_remote=False)  # force local for diagnostics
        vs = FAISS.load_local(
            VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True
        )
        print("Loaded FAISS index (diagnostic load with local embeddings).")
        docs = vs.similarity_search("Apple revenue", k=3)
        print(f"Top doc sources: {[d.metadata.get('source') for d in docs]}")
        qa = load_qa_chain(retriever_k=3)
        res = qa({"query": sample_query})
        print("Sample query:", sample_query)
        print("Answer snippet:", res["result"][:400])
    except Exception as e:
        print("Diagnostic failure:", e)
        traceback.print_exc()

if __name__ == "__main__":
    # This can be run once to create the vector store initially
    if not os.path.exists(VECTOR_STORE_PATH):
        create_vector_store()

    # Example usage
    qa = load_qa_chain()
    query = "What are the primary revenue drivers for Apple?"
    result = qa({"query": query})

    print("Question:", query)
    print("Answer:", result["result"])
    print("Source Documents:", [doc.metadata['source'] for doc in result['source_documents']])

