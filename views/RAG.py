import streamlit as st
from openai import OpenAI
import pandas as pd
from typing import List
import PyPDF2
from io import BytesIO
import time
import re

class PDFProcessor:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
    
    def extract_text_from_pdf(self, pdf_file: BytesIO) -> str:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        text = re.sub(r'\s+', ' ', text).strip()
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            if end < len(text):
                for punct in ['. ', '? ', '! ']:
                    last_punct = text[start:end].rfind(punct)
                    if last_punct != -1:
                        end = start + last_punct + 2
                        break
            chunks.append(text[start:end].strip())
            start = end - overlap
        return chunks

    def get_embedding(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

    def similarity_search(self, query_embedding: List[float], embeddings_df: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
        embeddings_df['similarity'] = embeddings_df['embedding'].apply(
            lambda x: sum(a*b for a, b in zip(x, query_embedding)) / 
            (sum(a*a for a in x)**0.5 * sum(b*b for b in query_embedding)**0.5)
        )
        return embeddings_df.nlargest(top_k, 'similarity')

    def generate_response(self, query: str, context: str) -> str:
        messages = [
            {"role": "system", "content": """You are a helpful assistant analyzing PDF documents. 
             Use the provided context to answer questions accurately and concisely. 
             If the answer cannot be found in the context, say so clearly."""},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"}
        ]
        
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=messages,
            temperature=0.7,
        )
        return response.choices[0].message.content

# Initialize session state
if 'embeddings_df' not in st.session_state:
    st.session_state.embeddings_df = pd.DataFrame(columns=['text', 'embedding', 'page'])
if 'processor' not in st.session_state:
    st.session_state.processor = None
if 'pdf_name' not in st.session_state:
    st.session_state.pdf_name = None
if 'response' not in st.session_state:
    st.session_state.response = None
if 'context' not in st.session_state:
    st.session_state.context = None

# Page layout
st.title("📚 PDF Document Assistant")
st.caption("Upload a PDF and ask questions about its content")

st.markdown("""
    Please don't use it with too large documents, it will crash my free tier server 😔    
""")

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter OpenAI API Key:", type="password")
    if api_key:
        st.session_state.processor = PDFProcessor(api_key)
    
    uploaded_file = st.file_uploader("Upload a PDF file:", type=['pdf'])
    
    chunk_size = st.slider("Chunk Size (characters)", 500, 2000, 1000, 100)
    chunk_overlap = st.slider("Chunk Overlap (characters)", 50, 200, 100, 10)

# Process PDF when uploaded
if uploaded_file and api_key:
    if uploaded_file.name != st.session_state.pdf_name:
        with st.spinner("Processing PDF..."):
            try:
                st.session_state.pdf_name = uploaded_file.name
                text = st.session_state.processor.extract_text_from_pdf(uploaded_file)
                chunks = st.session_state.processor.chunk_text(
                    text, 
                    chunk_size=chunk_size, 
                    overlap=chunk_overlap
                )
                
                new_embeddings = []
                progress_bar = st.progress(0)
                
                for i, chunk in enumerate(chunks):
                    if st.session_state.processor:
                        embedding = st.session_state.processor.get_embedding(chunk)
                        new_embeddings.append({
                            'text': chunk, 
                            'embedding': embedding
                        })
                        progress_bar.progress((i + 1) / len(chunks))
                        time.sleep(0.1)
                
                st.session_state.embeddings_df = pd.DataFrame(new_embeddings)
                st.success(f"Processed {len(chunks)} chunks from PDF successfully!")
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")

# Main content area
if not api_key:
    st.warning("Please enter your OpenAI API key in the sidebar to begin.")
elif not uploaded_file:
    st.info("Please upload a PDF file in the sidebar to begin.")
else:
    # Query input and processing
    query = st.text_input("What would you like to know about the document?")
    
    if query and not st.session_state.embeddings_df.empty:
        with st.spinner("Searching document and generating response..."):
            query_embedding = st.session_state.processor.get_embedding(query)
            similar_chunks = st.session_state.processor.similarity_search(
                query_embedding, 
                st.session_state.embeddings_df
            )
            
            context = "\n".join(similar_chunks['text'].tolist())
            response = st.session_state.processor.generate_response(query, context)
            
            # Store response and context in session state
            st.session_state.response = response
            st.session_state.context = similar_chunks
    
    # Display results
    if st.session_state.response:
        st.write("### Answer:")
        st.write(st.session_state.response)
        
        with st.expander("View source context"):
            for idx, row in st.session_state.context.iterrows():
                st.markdown("---")
                st.info(f"Similarity Score: {row['similarity']:.4f}")
                st.write(row['text'])