import json

import numpy as np
import streamlit as st
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px

from yoloexplorer import config
from yoloexplorer.frontend.datasets import _get_primary_dataset


@st.cache_resource
def reduce_dim(df, alg):
    embeddings = np.array(df["vector"].to_list())
    if alg == "TSNE":
        tsne = TSNE(n_components=2, random_state=0)
        embeddings = tsne.fit_transform(embeddings)
    elif alg == "PCA":
        pca = PCA(n_components=2)
        embeddings = pca.fit_transform(embeddings)
    return embeddings


def embeddings():
    exp = _get_primary_dataset()
    df = exp.table.to_pandas()
    col1, col2 = st.columns([0.5, 0.5])
    with col1:
        st.dataframe(df, use_container_width=True)
    with col2:
        option = st.selectbox("Dim Reducer Algorithm", ("TSNE", "PCA", "UMAP (Coming soon)"))
        if option == "TSNE":
            embeddings = reduce_dim(df, "TSNE")
        elif option == "PCA":
            embeddings = reduce_dim(df, "TSNE")
        elif option == "UMAP (Coming soon)":
            st.write("Coming soon")

        fig = px.scatter(x=embeddings[:, 0], y=embeddings[:, 1])
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    embeddings()
