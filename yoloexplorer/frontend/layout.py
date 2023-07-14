import sys
import json
import argparse
import streamlit as st
from streamlit_image_select import image_select
from typing import List
import streamlit.web.cli as stcli

from  yoloexplorer.frontend.states import init_states, update_state

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--info", type=str, required=True)
    return parser.parse_args()

def layout(info):
    # function scope import
    from yoloexplorer import Explorer
    import pdb; pdb.set_trace()
    init_states()
    update_state("EXPLORER", Explorer(**json.load(info)))
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
    st.session_state.IMGS = exp.to_pandas()["path"].to_list()
    if st.session_state.IMGS:
        clicked = image_select("Samples", images=st.session_state.IMGS)


def launch(info):
    sys.argv = ["streamlit", "run", __file__, "-- --info", info]
    stcli.main()

if __name__ == "__main__":
    args = argparser()
    layout(args.info)