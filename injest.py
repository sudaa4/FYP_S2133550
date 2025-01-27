import os
import glob
import json
import uuid
import requests
from PyPDF2 import PdfReader
import chromadb
from chromadb.config import Settings
from nltk.tokenize import sent_tokenize
from transformers import pipeline
import numpy as np

##Run this file first

###############################################################################
# 1. A function to call Ollama for embeddings (nomic-embed-text)
###############################################################################
def get_ollama_embedding(text, server="http://127.0.0.1:11434", model="nomic-embed-text"):
    """
    Calls Ollama's /api/embeddings endpoint to generate an embedding for the given text.
    Returns a Python list of floats representing the embedding vector.
    """
    endpoint = f"{server}/api/embeddings"
    payload = {"model": model, "prompt": text}

    try:
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()
        data = response.json()
        embedding = data.get("embedding", [])
        if not embedding:
            print(f"No embedding returned for text: {text[:50]}...")
            return []
        return embedding
    except Exception as e:
        print(f"Error getting embedding for text: {text[:50]}... - {e}")
        return []

###############################################################################
# 2. Cosine Similarity Helper
###############################################################################
def cosine_similarity(vec1, vec2):
    """
    Computes cosine similarity between two embedding vectors.
    """
    vec1 = np.array(vec1, dtype=np.float32)
    vec2 = np.array(vec2, dtype=np.float32)
    numerator = np.dot(vec1, vec2)
    denominator = (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)

###############################################################################
# 3. Semantic Chunking Function
###############################################################################
def semantic_chunking(
    text,
    max_chunk_size=200, 
    similarity_threshold=0.8
):
    """    Splits text into semantically coherent chunks by:
      1) Splitting text into sentences.
      2) Computing embeddings for each sentence.
      3) Grouping sentences into a chunk if they are sufficiently similar
         (cosine similarity >= similarity_threshold) AND the chunk size
         in words does not exceed max_chunk_size.
    Returns a list of chunk strings.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk_sentences = []
    current_chunk_embedding = None
    current_chunk_size = 0

    for sent in sentences:
        sentence_words = len(sent.split())
        # Get an embedding for this sentence
        sent_emb = get_ollama_embedding(sent)
        if not sent_emb:
            # If embedding failed, skip
            continue

        # If we're starting a new chunk, just initialize
        if current_chunk_embedding is None:
            current_chunk_sentences = [sent]
            current_chunk_embedding = sent_emb
            current_chunk_size = sentence_words
            continue

        # Compute similarity with the current chunk's average embedding
        similarity = cosine_similarity(current_chunk_embedding, sent_emb)

        # Check if we can add this sentence to the current chunk
        can_add_by_similarity = (similarity >= similarity_threshold)
        can_add_by_size = (current_chunk_size + sentence_words <= max_chunk_size)

        if can_add_by_similarity and can_add_by_size:
            # Merge this sentence into the existing chunk
            current_chunk_sentences.append(sent)
            current_chunk_size += sentence_words

            # Update the chunk embedding by simple averaging
            old_chunk_count = float(len(current_chunk_sentences) - 1)
            new_chunk_count = float(len(current_chunk_sentences))
            current_chunk_embedding = [
                (curr_val * old_chunk_count + new_val) / new_chunk_count
                for (curr_val, new_val) in zip(current_chunk_embedding, sent_emb)
            ]
        else:
            # Finalize the current chunk
            chunk_text = " ".join(current_chunk_sentences).strip()
            if chunk_text:
                chunks.append(chunk_text)
            
            # Start a new chunk with the current sentence
            current_chunk_sentences = [sent]
            current_chunk_embedding = sent_emb
            current_chunk_size = sentence_words

    # If any chunk remains, append it
    if current_chunk_sentences:
        chunk_text = " ".join(current_chunk_sentences).strip()
        if chunk_text:
            chunks.append(chunk_text)

    return chunks

###############################################################################
# 4. Load PDF content
###############################################################################
def load_pdf_content(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        # Add fallback for pages that might not extract text properly
        page_text = page.extract_text() or ""
        text += page_text + "\n"
    return text

###############################################################################
# 5. Gather text from PDFs and .txt files
###############################################################################
def load_data(data_folder):
    all_chunks = []
    file_paths = glob.glob(os.path.join(data_folder, "*.pdf")) + \
                 glob.glob(os.path.join(data_folder, "*.txt"))

    for path in file_paths:
        print(f"Processing file: {path}")
        if path.endswith(".pdf"):
            content = load_pdf_content(path)
            print(f"Loaded PDF content from {path}.")
        else:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                print(f"Loaded TXT content from {path}.")
            except Exception as e:
                print(f"Error reading {path}: {e}")
                continue

        # Chunk the content using semantic chunking
        chunks = semantic_chunking(content, max_chunk_size=200, similarity_threshold=0.8)
        print(f"Created {len(chunks)} semantic chunks from {path}.")
        for c in chunks:
            c = c.strip()
            if c:
                all_chunks.append(c)
    return all_chunks

###############################################################################
# 6. Ingest into Chroma with Testing 
###############################################################################
def create_chroma_collection(chunks, collection_name="university_docs", persist_dir="./chroma_db"):
    # Ensure persistence directory exists
    os.makedirs(persist_dir, exist_ok=True)

    # 1) Connect to Chroma with persistence
    try:
        client = chromadb.PersistentClient(path=persist_dir)
        print("Connected to ChromaDB with persistence.")
    except Exception as e:
        print(f"Error connecting to ChromaDB: {e}")
        return None

    # 2) Create (or load) the collection
    try:
        collection = client.get_or_create_collection(name=collection_name)
        print(f"Accessed collection '{collection_name}'.")
    except Exception as e:
        print(f"Error accessing collection '{collection_name}': {e}")
        return None

    # 3) Generate embeddings via Ollama + store
    ids = []
    embeddings = []
    documents = []
    for i, chunk in enumerate(chunks):
        emb = get_ollama_embedding(chunk)  # using nomic-embed-text
        if not emb:
            print(f"Skipping chunk {i} due to empty or invalid embedding.")
            continue

        embeddings.append(emb)
        documents.append(chunk)
        unique_id = str(uuid.uuid4())  # Generate a unique ID
        ids.append(unique_id)

    # 4) Add documents to the collection & test
    if embeddings:
        try:
            collection.add(
                documents=documents,
                embeddings=embeddings,
                ids=ids
            )
            print(f"Successfully added {len(embeddings)} document chunks to the collection '{collection_name}'.")

            # --- TEST 1: Check how many documents are now in the collection
            collection_count = collection.count()
            print(f"[TEST 1] The collection '{collection_name}' now has {collection_count} documents.")

            # --- TEST 2: Retrieve a sample from the stored documents
            sample = collection.get(ids=ids[:1])  # get first doc
            print(f"[TEST 2] A sample from the collection:\n{sample['documents']}")

        except Exception as e:
            print(f"Error adding documents to ChromaDB: {e}")
    else:
        print("No embeddings were generated. Collection not updated.")

    return collection

###############################################################################
# MAIN INGEST SCRIPT
###############################################################################
if __name__ == "__main__":
    data_folder = "./resources"  # folder with your .pdf and .txt
    print(f"Loading data from folder: {data_folder}")
    chunks = load_data(data_folder)
    print(f"Total number of chunks: {len(chunks)}")

    if chunks:
        # Create or update the Chroma collection
        collection = create_chroma_collection(chunks, "university_docs")
        if collection:
            print("Chroma ingestion completed successfully!")
    else:
        print("No valid chunks were found. Exiting.")
