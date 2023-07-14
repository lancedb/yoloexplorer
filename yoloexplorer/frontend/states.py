import streamlit as st

INDEX_PAGE_QUERY_FORM_KEY = "index_page_query_form"
INDEX_PAGE_SIMILARITY_FORM_KEY = "index_page_similarity_form"

def init_states():
    st.session_state.EXPLORER = None
    st.session_state.IMGS = []
    st.session_state.SELECTED_IMGS = 0
