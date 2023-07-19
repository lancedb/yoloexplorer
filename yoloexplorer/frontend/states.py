import streamlit as st


def widget_key(action, data):
    return f"form_{action}_on_{data}"

def init_states(config_list):
    for config in config_list:
        data = config["data"].split(".")[0]
        st.session_state[f"EXPLORER_{data}"] = None
        st.session_state[f"IMGS_{data}"] = []
        st.session_state[f"SELECTED_IMGS_{data}"] = []
    st.session_state[f"STAGED_IMGS"] = []


def update_state(state, value, rerun=True):
    st.session_state[state] = value
    st.experimental_rerun() if rerun else None
