import streamlit as st
from pinecone import Pinecone
from io import BytesIO
import base64

collection_name = "landscapeimage"

# Initialize session state
if 'selected_record' not in st.session_state:
    st.session_state.selected_record = None

def set_selected_record(new_record):
    st.session_state.selected_record = new_record

@st.cache_resource
def get_client():
    return Pinecone(
        api_key=st.secrets.get("PINECONE_API_KEY")
    )

def get_initial_records():
    pc = get_client()
    index = pc.Index(collection_name)
    # Fetch initial records using Pinecone's query
    # Since Pinecone doesn't have a direct scroll equivalent, 
    # we'll query with a neutral vector to get initial results
    response = index.query(
        vector=[0] * 512,  # Adjust vector dimension based on your setup
        top_k=12,
        include_metadata=True
    )
    return response.matches

def get_similar_records():
    if st.session_state.selected_record is not None:
        pc = get_client()
        index = pc.Index(collection_name)
        # Get the vector of the selected record
        selected_id = st.session_state.selected_record['id']
        response = index.query(
            id=selected_id,
            top_k=12,
            include_metadata=True
        )
        return response.matches
    return None

def get_bytes_from_base64(base64_string):
    return BytesIO(base64.b64decode(base64_string))

# Get records based on whether there's a selection
records = get_similar_records() if st.session_state.selected_record is not None else get_initial_records()

# Display selected image if there is one
if st.session_state.selected_record:
    image_bytes = get_bytes_from_base64(
        st.session_state.selected_record['metadata']['base64']
    )
    st.header("Image similar to:")
    st.image(
        image=image_bytes
    )
    st.divider()

# Create grid layout
columns = st.columns(3)

# Display images and buttons
for idx, record in enumerate(records):
    col_idx = idx % 3
    image_bytes = get_bytes_from_base64(record.metadata["base64"])
    with columns[col_idx]:
        st.image(
            image=image_bytes
        )
        st.button(
            label="Find similar images",
            key=record.id,
            on_click=set_selected_record,
            args=[record]
        )