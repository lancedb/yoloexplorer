import json
import subprocess
import streamlit as st
from streamlit_image_select import image_select
from yoloexplorer import config
from yoloexplorer.frontend.states import init_states, update_state

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


    st.write("**Dataset Visualization**")

    exp = st.session_state.EXPLORER
    st.session_state.IMGS = exp.table.to_pandas()["path"].to_list()
    if st.session_state.IMGS:
        num = min(4000, len(st.session_state.IMGS))

        clicked = image_select("Samples", images=st.session_state.IMGS[0:num]) #noqa


def launch():
    cmd = ["streamlit", "run", __file__, "--server.maxMessageSize", "1024"]
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(e)

if __name__ == "__main__":
    layout()