import streamlit as st
import os
import requests
import chromadb
from chromadb.config import Settings
import uuid
from better_profanity import profanity

# Initialize profanity filter
profanity.load_censor_words()

# Streamlit Page Configuration
st.set_page_config(page_title="UMAI bot", layout="wide")

###############################################################################
# 1. Embedding & LLM Functions (via Ollama)
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
            st.warning(f"[Embedding Error] No embedding returned for text: {text[:50]}...")
            return []
        return embedding
    except Exception as e:
        st.error(f"[Embedding Error] Failed on text: {text[:50]}... - {e}")
        return []

def generate_ollama_response(
    question,
    context,
    server="http://127.0.0.1:11434",
    model="llama-3.2"
):
    """
    Calls Ollama's /api/generate endpoint using 'model' to produce an answer
    given a question and context. Adjust the prompt format as you like.
    """
    endpoint = f"{server}/api/generate"

    # A system or instruct-style prompt using the provided context
    prompt = f"""You are a helpful assistant. Use the following context to answer the user's question. 
If the answer cannot be found in the context, say \"I am sorry, I do not have an answer to your question, please contact the administration for more info.\"

Context:
{context}

Question: {question}
Answer:"""

    payload = {
        "model": model,
        "prompt": prompt
    }

    try:
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict) and "content" in data:
            return data["content"].strip()
        elif isinstance(data, dict) and "choices" in data:
            all_text = []
            for choice in data["choices"]:
                if "content" in choice:
                    all_text.append(choice["content"])
            return "\n".join(all_text).strip()
        else:
            return "I'm sorry, I couldn't parse the response from Ollama."
    except Exception as e:
        st.error(f"[LLM Error] Failed to generate response: {e}")
        return "I'm sorry, I encountered an error generating a response."

###############################################################################
# 2. Chroma Retrieval
###############################################################################
def retrieve_relevant_chunks(question, collection_name="university_docs", persist_dir="./chroma_db", top_k=3):
    """
    1. Connect to the local ChromaDB (persisted on disk).
    2. Embed the user question via Ollama.
    3. Query the collection for the top-k relevant chunks.
    4. Return them joined as a single context string, or as a list.
    """
    # Connect to Chroma
    try:
        client = chromadb.PersistentClient(path=persist_dir)
        collection = client.get_collection(name=collection_name)
    except Exception as e:
        st.error(f"[Chroma Error] Unable to connect to or fetch collection '{collection_name}': {e}")
        return ""

    # Embed the query
    query_embedding = get_ollama_embedding(question)
    if not query_embedding:
        return ""

    # Query top-k chunks
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        retrieved_docs = results.get("documents", [[]])[0]  # The first list in "documents"
        context = "\n\n".join(retrieved_docs)
        return context
    except Exception as e:
        st.error(f"[Chroma Error] Query failed: {e}")
        return ""

###############################################################################
# 3. Streamlit Interface
###############################################################################
def main():
    """
    Streamlit-based UI for UM AI Chatbot.
    """
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    selected_page = st.sidebar.radio("Go to", ["About Us", "Chat 1", "Chat 2"])

    if selected_page == "About Us":
        st.title("About UMAI bot")
        st.write("UMAI BOT is your personal assistant at the University of Malaya (UM). It helps students with various queries regarding their campus. Feel free to ask away.")
        st.markdown("<p style='font-size:16px;'>Developed with ❤️ by your Sudaarshan A/L Baskar.</p>", unsafe_allow_html=True)

    elif selected_page == "Chat 1":
        st.title("Chat 1")

        user_input = st.text_input("You (Chat for student Queries):", placeholder="Type your question here...")

        if user_input:
            if profanity.contains_profanity(user_input):
                st.warning("Your input contains inappropriate language. Please rephrase and try again.")
            else:
                with st.spinner("Processing your question..."):
                    context = retrieve_relevant_chunks(user_input)

                    if not context:
                        st.warning("No relevant context retrieved. The answer may be incomplete.")

                    response = generate_ollama_response(user_input, context)

                st.text_area("Chatbot Response:", value=response, height=200)

    elif selected_page == "Chat 2":
        st.title("Chat 2")
        st.markdown("<p style='font-size:16px; color:green;'> chat here .</p>", unsafe_allow_html=True)

        user_input = st.text_input("You :", placeholder="Type your question here...")

        if user_input:
            if profanity.contains_profanity(user_input):
                st.warning("Your input contains inappropriate language. Please rephrase and try again.")
            else:
                with st.spinner("Processing your question..."):
                    context = retrieve_relevant_chunks(user_input)

                    if not context:
                        st.warning("No relevant context retrieved. The answer may be incomplete.")

                    response = generate_ollama_response(user_input, context)

                st.text_area("Chatbot Response:", value=response, height=200)

if __name__ == "__main__":
    main()

