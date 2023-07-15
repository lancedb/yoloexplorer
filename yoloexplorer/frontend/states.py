import streamlit as st

def init_states():
    st.session_state.EXPLORER = None
    st.session_state.IMGS = []

def update_state(state, value):
    st.session_state[state] = value