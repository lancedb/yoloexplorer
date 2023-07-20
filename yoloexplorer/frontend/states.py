import streamlit as st


def widget_key(action, data):
    return f"form_{action}_on_{data}"


def init_states(config_list):
    for config in config_list:
        data = config["data"].split(".")[0]
        st.session_state[f"EXPLORER_{data}"] = None
        st.session_state[f"IMGS_{data}"] = []
        st.session_state[f"SELECTED_IMGS_{data}"] = []
    st.session_state["STAGED_IMGS"] = set()
    st.session_state["PRIMARY_DATASET"] = config_list[0]["data"].split(".")[0]


def update_state(state, value):
    st.session_state[state] = value
