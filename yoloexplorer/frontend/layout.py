import json
import subprocess

import streamlit as st
from streamlit_dash import image_select
from yoloexplorer import config
from yoloexplorer.frontend.states import init_states, update_state, widget_key


@st.cache_data
def _get_config():
    with open(config.TEMP_CONFIG_PATH) as json_file:
        data = json.load(json_file)
    return data


@st.cache_data
def _get_dataset(idx=0):
    from yoloexplorer import Explorer  # function scope import

    config = _get_config()[idx]
    exp = Explorer(**config)
    exp.build_embeddings()

    return exp


def reset_to_init_state():
    if st.session_state.get(f"STAGED_IMGS") is None:
        cfgs = _get_config()
        init_states(cfgs)
        for idx, cfg in enumerate(cfgs):
            data = cfg["data"].split(".")[0]
            exp = _get_dataset(idx)
            update_state(f"EXPLORER_{data}", exp)
            update_state(f"IMGS_{data}", exp.table.to_pandas()["path"].to_list())


def query_form(data):
    with st.form(widget_key("query", data)):
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            query = st.text_input("Query", "", label_visibility="collapsed")
        with col2:
            st.form_submit_button("Query", on_click=run_sql_query, args=(data, query))


def similarity_form(selected_imgs, selected_staged_imgs, data):
    st.write("Similarity Search")
    with st.form(widget_key("similarity", data)):
        subcol1, subcol2 = st.columns([1, 1])
        with subcol1:
            st.write("Limit")
            limit = st.number_input("limit", min_value=None, max_value=None, value=25, label_visibility="collapsed")

        with subcol2:
            disabled = len(selected_imgs) and len(selected_staged_imgs)
            st.write("Selected: ", len(selected_imgs))
            search = st.form_submit_button("Search", disabled=disabled)
            if disabled:
                st.error("Cannot search from staging and dataset")
            if search:
                find_similar_imgs(data, selected_staged_imgs or selected_imgs, limit)


def staging_area_form(data, selected_imgs):
    st.write("Staging Area")
    with st.form(widget_key("staging_area", data)):
        staged_imgs = set(st.session_state[f"STAGED_IMGS"]) - set(selected_imgs)
        st.form_submit_button(
            ":wastebasket:", disabled=len(selected_imgs) == 0, on_click=update_state, args=("STAGED_IMGS", staged_imgs)
        )
        st.form_submit_button("Clear", on_click=update_state, args=("STAGED_IMGS", set()))


def find_similar_imgs(data, imgs, limit=25):
    exp = st.session_state[f"EXPLORER_{data}"]
    _, idx = exp.get_similar_imgs(imgs, limit)
    paths = exp.table.to_pandas()["path"][idx].to_list()
    update_state(f"IMGS_{data}", paths)
    st.experimental_rerun()


def run_sql_query(data, query):
    if query.rstrip().lstrip():
        exp = st.session_state[f"EXPLORER_{data}"]
        df = exp.sql(query)
        update_state(f"IMGS_{data}", df["path"].to_list())


def remove_imgs(data, imgs):
    exp = st.session_state[f"EXPLORER_{data}"]
    ids = exp.table.to_pandas().set_index("path").loc[imgs]["id"].to_list()
    exp.remove_imgs(ids)
    update_state(f"IMGS_{data}", exp.table.to_pandas()["path"].to_list())


def layout():
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
    # staging area
    selected_staged_imgs = []
    if st.session_state.get(f"STAGED_IMGS"):
        staged_imgs = st.session_state[f"STAGED_IMGS"]
        total_staged_imgs = len(staged_imgs)
        col1, col2 = st.columns([0.8, 0.2], gap="small")
        with col1:
            selected_staged_imgs = image_select(
                f"Staged samples: {total_staged_imgs}", images=list(staged_imgs), use_container_width=False
            )
        with col2:
            staging_area_form(data="staging_area", selected_imgs=selected_staged_imgs)

    # Dataset tabs
    cfgs = _get_config()
    tabs = st.tabs([cfg["data"].split(".")[0] for cfg in cfgs])
    for idx, tab in enumerate(tabs):
        with tab:
            data = cfgs[idx]["data"].split(".")[0]
            col1, col2 = st.columns([0.75, 0.25], gap="small")

            reset_to_init_state()

            total_imgs = len(st.session_state[f"IMGS_{data}"])
            imgs = st.session_state[f"IMGS_{data}"]
            with col1:
                subcol1, subcol2 = st.columns([0.2, 0.8])
                with subcol1:
                    num = st.number_input(
                        "Max Images Displayed",
                        min_value=0,
                        max_value=total_imgs,
                        value=min(250, total_imgs),
                        key=widget_key("num_imgs_displayed", data),
                    )
                query_form(data)

                if total_imgs:
                    selected_imgs = image_select(
                        f"Total samples: {total_imgs}", images=imgs[0:num], use_container_width=False
                    )

            with col2:
                similarity_form(selected_imgs, selected_staged_imgs, data)
                total_staged_imgs = set(st.session_state["STAGED_IMGS"])
                total_staged_imgs.update(selected_imgs)

                display_labels = st.checkbox("Labels", value=False, key=widget_key("labels", data))
                st.button(
                    "Add to Staging",
                    key=widget_key("staging", data),
                    disabled=not selected_imgs,
                    on_click=update_state,
                    args=("STAGED_IMGS", total_staged_imgs),
                )

                if data == st.session_state["PRIMARY_DATASET"]:
                    st.button(
                        ":wastebasket:",
                        key=widget_key("delete", data),
                        on_click=remove_imgs,
                        args=(data, selected_imgs),
                        disabled=not selected_imgs or (len(selected_imgs) and len(selected_staged_imgs)),
                    )


def launch():
    cmd = ["streamlit", "run", __file__, "--server.maxMessageSize", "1024"]
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    layout()
