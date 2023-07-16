import streamlit as st
from yoloexplorer import Explorer
from streamlit_image_select import image_select
from typing import List

st.title("YOLOExplorer")

with st.sidebar:
    default_data_yaml = "coco128.yaml"
    st.text(f"Dataset: {default_data_yaml}")
    if default_data_yaml:
        if "data_yaml" not in st.session_state:
            st.session_state["data_yaml"] = default_data_yaml
        if "session" not in st.session_state:
            st.session_state["session"] = Explorer(default_data_yaml, model=None)
            dataset_info = st.session_state["session"].dataset_info

            st.write("**Dataset Information :**")

            for dataset_type in ["train", "val", "tets"]:
                st.text(f"{dataset_type.capitalize()} paths :")
                if isinstance(dataset_info["train"], str):
                    st.text(dataset_info["train"])
                elif isinstance(dataset_info["train", List]):
                    for p in dataset_info["train"]:
                        st.text(p)

st.write("**Dataset Visualization**")

# TODO: Only show one of the sample dataset at same time
img_train_checkbox_w, img_val_checkbox_w, img_test_checkbox_w, num_sample_w = st.columns(4)
with img_train_checkbox_w:
    img_train_checkbox = st.checkbox(label="Train")
with img_val_checkbox_w:
    img_val_checkbox = st.checkbox(label="Val")
with img_test_checkbox_w:
    img_test_checkbox = st.checkbox(label="Test")
with num_sample_w:
    num_samples_show = st.number_input(label="Numbers of Images", value=10)


if num_samples_show:
    if img_train_checkbox:
        img_url = "/home/hd/zidane.jpg"  # Put path to your image
        train_image_paths = [img_url] * int(num_samples_show)
        # st.image(train_image_paths, use_column_width="always")
        clicked = image_select("Training Samples", images=train_image_paths)
