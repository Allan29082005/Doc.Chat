import os
import streamlit as st
import fitz  # PyMuPDF
import toml
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# ========== CONFIG ==========

# Load API key from secrets.toml
secrets = toml.load("secrets.toml")
GEMINI_API_KEY = secrets["AIzaSyCDk714DHDm04dCp2fbBgDsMy51T2DFkQw"]

CHUNK_SIZE = 500
TOP_K = 4

# ========== GEMINI SETUP ==========
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("models/gemini-1.5-flash")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ========== STREAMLIT UI ==========
st.set_page_config(page_title="DocBot (Gemini + RAG)", layout="wide")

st.title("DocBot (Gemini + RAG)")
st.markdown("Ask a question, and get answers from the document you've uploaded.")

# Sidebar for file upload
st.sidebar.title("📂 PDF Upload")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")

# Sidebar Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # (question, answer)

if uploaded_file:
    with st.spinner("📥 Reading and processing your PDF..."):
        os.makedirs("temp_files", exist_ok=True)
        file_path = os.path.join("temp_files", uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        # Extract PDF text
        doc = fitz.open(file_path)
        full_text = "\n".join(page.get_text() for page in doc)

        # Chunk the text
        def split_text(text, chunk_size=CHUNK_SIZE, overlap=50):
            return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]

        chunks = split_text(full_text)

        # Embed and index chunks
        chunk_embeddings = embedder.encode(chunks)
        index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
        index.add(np.array(chunk_embeddings))

        st.success("✅ Document indexed. Ask your question!")

    # Q&A
    query = st.text_input("Ask a question:")
    if query:
        query_embedding = embedder.encode([query])
        _, indices = index.search(np.array(query_embedding), TOP_K)
        retrieved_chunks = [chunks[i] for i in indices[0]]
        context = "\n\n".join(retrieved_chunks)

        prompt = f"""Answer the question using the context below.

        Context:
        {context}

        Question:
        {query}
        """

        with st.spinner("🤖 Thinking..."):
            response = gemini_model.generate_content(prompt)
            answer = response.text

        # Save to chat history
        st.session_state.chat_history.append((query, answer))

        # Show Answer
        st.markdown("### 🤖 Answer:")
        st.write(answer)

# ========== CHAT HISTORY ==========
st.sidebar.header("🗨️ Chat History")
with st.sidebar:
    for user_msg, gemini_msg in reversed(st.session_state.chat_history):
        with st.expander(
            label=f"""
                <div style="background-color:#D1C4E9; color:#311B92; padding:5px 10px; border-radius:6px;">
                    🧑 You: {user_msg}
                </div>
            """,
            expanded=False,
        ):
            st.markdown(
                f'<div style="background-color:#F3E5F5; border-radius:10px; padding:10px; margin:5px 0; color:#4A148C;"><b>Gemini:</b> {gemini_msg}</div>',
                unsafe_allow_html=True
            )
