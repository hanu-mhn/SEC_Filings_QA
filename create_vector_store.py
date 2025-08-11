import logging
import warnings
import os
import shutil
from qa_agent import create_vector_store

# Suppress warnings
warnings.filterwarnings("ignore", message="libmagic is unavailable")
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    print("Starting vector store creation...")
    
    # Delete vector_store directory if it exists to start fresh
    if os.path.exists("vector_store"):
        print("Removing existing vector_store directory...")
        shutil.rmtree("vector_store")
        print("Existing vector_store directory removed.")
    
    print("Creating the vector store... This may take a while.")
    create_vector_store()
    
    # Verify if vector_store was created
    if os.path.exists("vector_store"):
        print("SUCCESS: Vector store created and saved to vector_store directory.")
    else:
        print("ERROR: Vector store directory was not created.")
