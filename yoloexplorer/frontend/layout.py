import sys
import json
import subprocess
import argparse
import streamlit as st
from streamlit_image_select import image_select
from typing import List
import streamlit.web.cli as stcli
from yoloexplorer import config

def init_states():
    st.session_state.EXPLORER = None
    st.session_state.IMGS = []

def update_state(state, value):
    st.session_state[state] = value

def layout():
    # function scope import
    from yoloexplorer import Explorer

    init_states()
    with open(config.TEMP_CONFIG_PATH) as json_file:
        data = json.load(json_file)
    exp = Explorer(**data)
    exp.build_embeddings()
    update_state("EXPLORER", exp)
    st.title("YOLOExplorer")

    with st.sidebar:
        dataset_info = st.session_state.EXPLORER.dataset_info
        st.write("**Dataset Information :**")

        for dataset_type in ["train", "val", "tets"]:
            st.text(f"{dataset_type.capitalize()} paths :")
            if isinstance(dataset_info['train'], str):
                st.text(dataset_info['train'])
            elif isinstance(dataset_info['train', List]):
                for p in dataset_info['train']:
                    st.text(p)

    st.write("**Dataset Visualization**")

    exp = st.session_state.EXPLORER
    st.session_state.IMGS = exp.table.to_pandas()["path"].to_list()
    if st.session_state.IMGS:
        clicked = image_select("Samples", images=st.session_state.IMGS)


def launch():
    cmd = ["streamlit", "run", __file__]
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(e)

if __name__ == "__main__":
    layout()