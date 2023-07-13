import streamlit as st
from streamlit_image_select import image_select
from typing import List

from .launch import run_streamlit
from .states import init_states, update_state

def layout():
    st.title("YOLOExplorer")

    with st.sidebar:
        dataset_info = st.session_state.EXPLORER.get_dataset_info()
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
    st.session_state.EXPLORER.IMGS = exp.to_pandas()["path"].to_list()
    if st.session_state.EXPLORER.IMGS:
        clicked = image_select("Samples", images=st.session_state.EXPLORER.IMGS)


def dash(explorer):
    run_streamlit(__file__)
    init_states()
    update_state("EXPLORER", explorer)
    layout()