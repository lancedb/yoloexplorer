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
    return data["exps"]


@st.cache_data
def _get_dataset(idx=0):
    from yoloexplorer import Explorer  # function scope import

    config = _get_config()[idx]
    exp = Explorer(**config)
    exp.build_embeddings()

    return exp

def _get_primary_dataset():
    data = st.session_state["PRIMARY_DATASET"]
    exp = st.session_state[f"EXPLORER_{data}"]

    return exp

def reset_to_init_state():
    if st.session_state.get(f"STAGED_IMGS") is None: # if app is not initialized TODO: better check
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
            submit = st.form_submit_button("Query")
        if submit:
            run_sql_query(data, query)



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
            submit = st.form_submit_button("Search", disabled=disabled)
            if disabled:
                st.error("Cannot search from staging and dataset")
            if submit:
                find_similar_imgs(data, selected_imgs or selected_staged_imgs, limit)      

def staging_area_form(data, selected_imgs):
    st.write("Staging Area")
    with st.form(widget_key("staging_area", data)):
        col1, col2 = st.columns([1, 1])
        staged_imgs = set(st.session_state[f"STAGED_IMGS"]) - set(selected_imgs)
        with col1:
            st.form_submit_button(
            ":wastebasket:", disabled=len(selected_imgs) == 0, on_click=update_state, args=("STAGED_IMGS", staged_imgs)
            )
        with col2:
            st.form_submit_button("Clear", on_click=update_state, args=("STAGED_IMGS", set()))

def selected_options_form(data, selected_imgs, selected_staged_imgs, total_staged_imgs):
    with st.form(widget_key("selected_options", data)):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.form_submit_button(
                    "Add to Stage",
                    #key=widget_key("staging", data),
                    on_click=add_to_staging,
                    args=("STAGED_IMGS", total_staged_imgs),
                    disabled=not selected_imgs)

            
        with col2:
            if data == st.session_state["PRIMARY_DATASET"]:
                st.form_submit_button(
                        ":wastebasket:",
                        disabled=not selected_imgs or (len(selected_imgs) and len(selected_staged_imgs)),
                        on_click=remove_imgs,
                        args=(data, selected_imgs),
                    )

            else:
                st.form_submit_button(
                        f"Add to {st.session_state['PRIMARY_DATASET']}",
                        on_click=add_imgs,
                        args=(data, selected_imgs),
                        disabled=not selected_imgs,
                    )

def persist_reset_form():
    with st.form(widget_key("persist_reset", "PRIMARY_DATASET")):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.form_submit_button("Reset", on_click=reset)

        with col2:
            st.form_submit_button("Persist", on_click=update_state, args=("PERSISTING", True))
    
def find_similar_imgs(data, imgs, limit=25, rerun=False):
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
        st.experimental_rerun()

def add_to_staging(key, imgs):
    update_state(key, imgs)
    #st.experimental_rerun()

def remove_imgs(data, imgs):
    exp = st.session_state[f"EXPLORER_{data}"]
    idxs = exp.table.to_pandas().set_index("path").loc[imgs]["id"].to_list()
    exp.remove_imgs(idxs)
    update_state(f"IMGS_{data}", exp.table.to_pandas()["path"].to_list())
    #st.experimental_rerun()

def add_imgs(from_data, imgs):
    data = st.session_state["PRIMARY_DATASET"]
    exp = st.session_state[f"EXPLORER_{data}"]
    from_exp = st.session_state[f"EXPLORER_{from_data}"]
    idxs = from_exp.table.to_pandas().set_index("path").loc[imgs]["id"].to_list()
    exp.add_imgs(from_exp, idxs)
    update_state(f"IMGS_{data}", exp.table.to_pandas()["path"].to_list())
    update_state(f"SUCCESS_MSG", f"Added {len(imgs)} to {data}")

def reset():
    data = st.session_state["PRIMARY_DATASET"]
    exp = st.session_state[f"EXPLORER_{data}"]
    exp.reset()
    update_state("STAGED_IMGS", None)

def persist_changes():
    exp = _get_primary_dataset()
    with st.spinner('Creating new dataset...'):
        exp.persist()
    st.success("Dataset created successfully! Auto-reload in 30 seconds...")
    update_state("PERSISTING", False)
    st.button("Refresh", on_click=update_state, args=("STAGED_IMGS", None))

def rerender_button(data):
    col1, col2, col3 = st.columns([0.26, 0.3, 0.1])
    with col1:
        pass
    with col2:
        st.button(
        "Render Imgs :arrows_counterclockwise:",
        key=widget_key("render_imgs", data),
        help="""
        Imgs might not be rendered automatically in some cases to save memory when stage area is used.
        Click this button to force render imgs.
        """
        )
    with col3:
        pass

def layout():
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

    if st.session_state.get("PERSISTING"):
        persist_changes()
        return
    
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

            imgs = st.session_state[f"IMGS_{data}"]
            total_imgs = len(imgs)
            with col1:
                subcol1, subcol2, subcol3 = st.columns([0.2, 0.2, 0.6])
                with subcol1:
                    num = st.number_input(
                        "Max Images Displayed",
                        min_value=0,
                        max_value=total_imgs,
                        value=min(250, total_imgs),
                        key=widget_key("num_imgs_displayed", data),
                    )
                with subcol2:
                    start_idx = st.number_input('Start Index', min_value=0, max_value=total_imgs, value=0, key=widget_key("start_idx", data))
                with subcol3:
                    select_all = st.checkbox("Select All", value=False, key=widget_key("select_all", data))
    
                query_form(data)
                selected_imgs = []
                if total_imgs:
                    imgs_displayed = imgs[start_idx: start_idx+num]
                    selected_imgs = image_select(
                        f"Total samples: {total_imgs}", images=imgs_displayed, use_container_width=False,
                        indices=[i for i in range(num)] if select_all else None
                    )
                    if st.session_state.get(f"STAGED_IMGS"):
                        rerender_button(data)


            with col2:
                similarity_form(selected_imgs, selected_staged_imgs, data)
                total_staged_imgs = set(st.session_state["STAGED_IMGS"])
                total_staged_imgs.update(selected_imgs)

                display_labels = st.checkbox("Labels", value=False, key=widget_key("labels", data))
                selected_options_form(data, selected_imgs, selected_staged_imgs, total_staged_imgs)
                if data == st.session_state["PRIMARY_DATASET"]:
                    persist_reset_form()


def launch():
    cmd = ["streamlit", "run", __file__, "--server.maxMessageSize", "1024"]
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    layout()
