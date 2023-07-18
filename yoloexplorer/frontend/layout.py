import json
import subprocess

import streamlit as st
from streamlit_dash import image_select
from yoloexplorer import config
from yoloexplorer.frontend.states import init_states, update_state, INDEX_PAGE_QUERY_FORM_KEY, INDEX_PAGE_SIMILARITY_FORM_KEY

@st.cache_data
def _get_dataset():
    from yoloexplorer import Explorer # function scope import

    with open(config.TEMP_CONFIG_PATH) as json_file:
        data = json.load(json_file)
    exp = Explorer(**data)
    exp.build_embeddings()

    return exp

def reset_to_init_state():
    if st.session_state.get("EXPLORER") is None:
        init_states()
        exp = _get_dataset()
        update_state("EXPLORER", exp)
        update_state("IMGS", exp.table.to_pandas()["path"].to_list())

def query_form():
    with st.form(INDEX_PAGE_QUERY_FORM_KEY):
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            query = st.text_input("Query", "", label_visibility="collapsed")
        with col2:
            submitted = st.form_submit_button("Query")
        if submitted:
            if query:
                exp = st.session_state.EXPLORER
                df = exp.sql(query)
                update_state("IMGS", df["path"].to_list())

def similarity_form(selected_imgs):
    st.write("Similarity Search")
    with st.form(INDEX_PAGE_SIMILARITY_FORM_KEY):
        subcol1, subcol2 = st.columns([1,1])
        with subcol1:
            st.write("Limit")
            limit = st.number_input("limit", min_value=None, max_value=None, value=25, label_visibility="collapsed")
        
        with subcol2:
            st.write("Selected: ", len(selected_imgs))
            submitted = st.form_submit_button("Search")

        if submitted:
            find_similar_imgs(selected_imgs, limit=limit)

def find_similar_imgs(imgs, limit=25):
    exp = st.session_state.EXPLORER
    df = exp.table.to_pandas()
    _, idx = exp.get_similar_imgs(imgs, limit)
    paths = df["path"][idx].to_list()
    update_state("IMGS", paths)
    st.experimental_rerun() 
    print("updated IMGS")

            

def layout(): 
    st.set_page_config(layout='wide')
    col1, col2 = st.columns([0.75, 0.25], gap="small")

    reset_to_init_state()
    with col1: 
        subcol1, subcol2 = st.columns([0.2, 0.8])
        with subcol1:
            num = st.number_input("Max Images Displayed", min_value=0, max_value=len(st.session_state.IMGS), value=min(250, len(st.session_state.IMGS)))
        query_form() 

        if st.session_state.IMGS:
            selected_imgs = image_select(f"Total samples: {len(st.session_state.IMGS)}", images=st.session_state.IMGS[0:num], indices=st.session_state.SELECTED_IMGS, use_container_width=False) #noqa

    with col2:
        similarity_form(selected_imgs)
        display_labels = st.checkbox("Labels", value=False) #noqa
        st.write("Coming Soon: ")
        st.write("Export/Merge Dataset(s)")


def launch():
    cmd = ["streamlit", "run", __file__, "--server.maxMessageSize", "1024"]
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(e)

if __name__ == "__main__":
    layout()