from pathlib import Path
from collections import defaultdict

import pandas as pd
import cv2
import duckdb
import matplotlib.pyplot as plt
import numpy as np
import yaml
from tqdm import tqdm

from ultralytics import YOLO
from ultralytics.yolo.utils import LOGGER, colorstr          
from ultralytics.yolo.utils.plotting import Annotator, colors
from torch import Tensor
import lancedb
import pyarrow as pa
from lancedb.embeddings import with_embeddings
from sklearn.decomposition import PCA

from yoloexplorer.dataset import get_dataset_info, Dataset
from yoloexplorer.yolo_predictor import YOLOEmbeddingsPredictor

SCHEMA = [
    "id",
    "img",
    "path",
    "cls",
    "labels",
    "bboxes",
    "segments",
    "keypoints",
    "meta",
]  # + "vector" with embeddings


def encode(img_path):
    img = cv2.imread(img_path)
    ext = Path(img_path).suffix
    img_encoded = cv2.imencode(ext, img)[1].tobytes()

    return img_encoded


def decode(img_encoded):
    nparr = np.frombuffer(img_encoded, np.byte)
    img = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)

    return img


class Explorer:
    """
    Dataset explorer
    """

    def __init__(self, data, model="yolov8n.pt", device="", project="run") -> None:
        """
        Args:
            data (str, optional): path to dataset file
            table (str, optional): path to LanceDB table to load embeddings Table from.
            model (str, optional): path to model. Defaults to None.
            device (str, optional): device to use. Defaults to ''. If empty, uses the default device.
            project (str, optional): path to project. Defaults to "runs/dataset".
        """
        self.data = data
        self.table = None
        self.model = model
        self.project = project
        self.dataset_info = None
        self.predictor = None
        self.trainset = None
        self.removed_img_count = 0
        self.verbose = False  # For embedding function
        self._sim_index = None
        self.version = None

        self.table_name = data
        self.temp_table_name = self.table_name + "_temp"

        # copy table to project if table is provided
        if model:
            self.predictor = self._setup_predictor(model, device)
        if data:
            self.dataset_info = get_dataset_info(self.data)

    def build_embeddings(self, batch_size=1000, verbose=False, force=False):
        """
        Builds the dataset in LanceDB table format

        Args:
            batch (int, optional): batch size. Defaults to 1000.
            verbose (bool, optional): verbose. Defaults to False.
            force (bool, optional): force rebuild. Defaults to False.
        """
        trainset = self.dataset_info["train"]
        trainset = trainset if isinstance(trainset, list) else [trainset]
        self.trainset = trainset
        self.verbose = verbose

        dataset = Dataset(
            img_path=trainset, data=self.dataset_info, augment=False, cache=False
        )
        batch_size = dataset.ni  # TODO: fix this hardcoding
        db = self._connect()
        if not force and self.table_name in db.table_names():
            LOGGER.info(
                "LanceDB embedding space already exists. Attempting to reuse it. Use force=True to overwrite."
            )
            self.table = self._open_table(self.table_name)
            self.version = self.table.version
            if len(self.table) == dataset.ni:
                return
            else:
                self.table = None
                LOGGER.info(
                    "Table length does not match the number of images in the dataset. Building embeddings..."
                )

        table_data = defaultdict(list)
        for idx, batch in enumerate(dataset):
            batch.pop("img")
            batch["id"] = idx
            batch["cls"] = batch["cls"].flatten().int().tolist()
            box_cls_pair = sorted(
                zip(batch["bboxes"].tolist(), batch["cls"]), key=lambda x: x[1]
            )
            batch["bboxes"] = [box for box, _ in box_cls_pair]
            batch["cls"] = [cls for _, cls in box_cls_pair]
            batch["labels"] = [self.dataset_info["names"][i] for i in batch["cls"]]
            batch["path"] = batch["im_file"]
            # batch["cls"] = batch["cls"].tolist()
            keys = (key for key in SCHEMA if key in batch)
            for key in keys:
                val = batch[key]
                if isinstance(val, Tensor):
                    val = val.tolist()
                table_data[key].append(val)

            table_data["img"].append(encode(batch["im_file"]))

            if len(table_data[key]) == batch_size or idx == dataset.ni - 1:
                df = pd.DataFrame(table_data)
                df = with_embeddings(
                    self._embedding_func, df, "img", batch_size=batch_size
                )
                if self.table:
                    self.table.add(table_data)
                else:
                    self.table = self._create_table(
                        self.table_name, data=df, mode="overwrite"
                    )
                self.version = self.table.version
                table_data = defaultdict(list)

        LOGGER.info(f'{colorstr("LanceDB:")} Embedding space built successfully.')

    def plot_embeddings(self):
        """
        Projects the embedding space to 2D using PCA

        Args:
            n_components (int, optional): number of components. Defaults to 2.
        """
        if self.table is None:
            LOGGER.error(
                "No embedding space found. Please build the embedding space first."
            )
            return None
        pca = PCA(n_components=2)
        embeddings = np.array(self.table.to_arrow()["vector"].to_pylist())
        embeddings = pca.fit_transform(embeddings)
        plt.scatter(embeddings[:, 0], embeddings[:, 1])
        plt.show()

    def get_similar_imgs(self, img, n=10):
        """
        Returns the n most similar images to the given image

        Args:
            img (int, str, Path): index of image in the table, or path to image
            n (int, optional): number of similar images to return. Defaults to 10.

        Returns:
            tuple: (list of paths, list of ids)
        """
        embeddings = None
        if self.table is None:
            LOGGER.error(
                "No embedding space found. Please build the embedding space first."
            )
            return None
        if isinstance(img, int):
            embeddings = self.table.to_pandas()["vector"][img]
        elif isinstance(img, (str, Path)):
            img = img
        elif isinstance(img, bytes):
            img = decode(img)
        else:
            LOGGER.error(
                "img should be index from the table(int) or path of an image (str or Path)"
            )
            return

        if embeddings is None:
            embeddings = self.predictor.embed(img).squeeze().cpu().numpy()
        sim = self.table.search(embeddings).limit(n).to_df()
        return sim["img"].to_list(), sim["id"].to_list()

    def plot_similar_imgs(self, img, n=10):
        """
        Plots the n most similar images to the given image

        Args:
            img (int, str, Path): index of image in the table, or path to image.
            n (int, optional): number of similar images to return. Defaults to 10.
        """
        _, ids = self.get_similar_imgs(img, n)
        self.plot_imgs(ids, n=n)

    def plot_imgs(self, ids=None, query=None, n=10, labels=True):
        if ids is None and query is None:
            ValueError("ids or query must be provided")

        # Resize the images to the minimum and maximum width and height
        resized_images = []
        df = (
            self.sql(query)
            if query
            else self.table.to_pandas().iloc[ids]
        )
        if n < len(df):
            df = df[0:n]
        for _, row in df.iterrows():
            img = decode(row["img"])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if labels:
                ann = Annotator(img)
                for box, label, cls in zip(row["bboxes"], row["labels"], row["cls"]):
                    ann.box_label(box, label, color=colors(cls, True))

                img = ann.result()
            resized_images.append(img)

        # Create a grid of the images
        fig, axes = plt.subplots(nrows=len(resized_images) // 2, ncols=2)
        for i, ax in enumerate(axes.ravel()):
            ax.imshow(resized_images[i])
            ax.axis("off")
        # Display the grid of images
        plt.show()

    def get_similarity_index(
        self, top_k=0.01, sim_thres=0.90, reduce=False, sorted=False
    ):
        """

        Args:
            sim_thres (float, optional): Similarity threshold to set the minimum similarity. Defaults to 0.9.
            top_k (float, optional): Top k fraction of the similar embeddings to apply the threshold on. Default 0.1.
            dim (int, optional): Dimension of the reduced embedding space. Defaults to 256.
            sorted (bool, optional): Sort the embeddings by similarity. Defaults to False.
        Returns:
            np.array: Similarity index
        """
        if self.table is None:
            LOGGER.error(
                "No embedding space found. Please build the embedding space first."
            )
            return None
        if top_k > 1.0:
            LOGGER.warning("top_k should be between 0 and 1. Setting top_k to 1.0")
            top_k = 1.0
        if top_k < 0.0:
            LOGGER.warning("top_k should be between 0 and 1. Setting top_k to 0.0")
            top_k = 0.0
        if sim_thres is not None:
            if sim_thres > 1.0:
                LOGGER.warning(
                    "sim_thres should be between 0 and 1. Setting sim_thres to 1.0"
                )
                sim_thres = 1.0
            if sim_thres < 0.0:
                LOGGER.warning(
                    "sim_thres should be between 0 and 1. Setting sim_thres to 0.0"
                )
                sim_thres = 0.0
        embs = np.array(self.table.to_arrow()["vector"].to_pylist())
        self._sim_index = np.zeros(len(embs))
        limit = max(int(len(embs) * top_k), 1)

        # create a new table with reduced dimensionality to speedup the search
        self._search_table = self.table
        if reduce:
            dim = min(256, embs.shape[1])  # TODO: make this configurable
            pca = PCA(n_components=min(dim, len(embs)))
            embs = pca.fit_transform(embs)
            dim = embs.shape[1]
            values = pa.array(embs.reshape(-1), type=pa.float32())
            table_data = pa.FixedSizeListArray.from_arrays(values, dim)
            table = pa.table(
                [table_data, self.table.to_arrow()["id"]], names=["vector", "id"]
            )
            self._search_table = self._create_table(
                "reduced_embs", data=table, mode="overwrite"
            )

        # with multiprocessing.Pool() as pool: # multiprocessing doesn't do much. Need to revisit
        #    list(tqdm(pool.imap(build_index, iterable)))

        for _, emb in enumerate(tqdm(embs)):
            df = self._search_table.search(emb).metric("cosine").limit(limit).to_df()
            if sim_thres is not None:
                df = df.query(f"score >= {1.0 - sim_thres}")
            for idx in df["id"][1:]:
                self._sim_index[idx] += 1
        self._drop_table("reduced_embs") if reduce else None

        return self._sim_index if not sorted else np.sort(self._sim_index)

    def plot_similarity_index(
        self, sim_thres=0.90, top_k=0.01, reduce=False, sorted=False
    ):
        """
        Plots the similarity index

        Args:
            threshold (float, optional): Similarity threshold to set the minimum similarity. Defaults to 0.9.
            top_k (float, optional): Top k fraction of the similar embeddings to apply the threshold on. Default 0.1.
            dim (int, optional): Dimension of the reduced embedding space. Defaults to 256.
            sorted (bool, optional): Whether to sort the index or not. Defaults to False.
        """
        index = self.get_similarity_index(top_k, sim_thres, reduce)
        if sorted:
            index = np.sort(index)
        plt.bar([i for i in range(len(index))], index)
        plt.xlabel("idx")
        plt.ylabel("similarity count")
        plt.show()

    def remove_imgs(self, idxs):
        """
        Works on temporary table. To apply the changes to the main table, call `persist()`

        Args:
            idxs (int or list): Index of the image to remove from the dataset.
        """
        if isinstance(idxs, int):
            idxs = [idxs]

        pa_table = self.table.to_arrow()
        mask = [True for _ in range(len(pa_table))]
        for idx in idxs:
            mask[idx] = False

        self.removed_img_count += len(idxs)

        table = pa_table.filter(mask)
        ids = [i for i in range(len(table))]
        table = table.set_column(0, 'id', [ids])  # TODO: Revisit this. This is a hack to fix the ids==dix
        self.table = self._create_table(
            self.temp_table_name, data=table, mode="overwrite"
        )  # work on a temporary table

        self.log_status()

    def add_imgs(self, exp, idxs):
        """
        Works on temporary table. To apply the changes to the main table, call `persist()`

        Args:
            data (pd.DataFrame or pa.Table): Table rows to add to the dataset.
        """
        table_df = self.table.to_pandas()
        data = exp.table.to_pandas().iloc[idxs]
        assert len(table_df["vector"].iloc[0]) == len(
            data["vector"].iloc[0]
        ), "Vector dimension mismatch"
        table_df = pd.concat([table_df, data], ignore_index=True)
        ids = [i for i in range(len(table_df))]
        table_df["id"] = ids
        self.table = self._create_table(
            self.temp_table_name, data=table_df, mode="overwrite"
        )  # work on a temporary table
        self.log_status()

    def reset(self):
        """
        Resets the dataset table to its original state or to the last persisted state.
        """
        if self.table is None:
            LOGGER.info("No changes made to the dataset.")
            return

        db = self._connect()
        if self.temp_table_name in db.table_names():
            self._drop_table(self.temp_table_name)

        self.table = self._open_table(self.table_name)
        self.removed_img_count = 0
        # self._sim_index = None # Not sure if we should reset this as computing the index is expensive
        LOGGER.info("Dataset reset to original state.")

    def persist(self, name=None):
        """
        Persists the changes made to the dataset. Available only if data is provided in the constructor.

        Args:
            name (str, optional): Name of the new dataset. Defaults to `data_updated.yaml`.
        """
        db = self._connect()
        if self.table is None or self.temp_table_name not in db.table_names():
            LOGGER.info("No changes made to the dataset.")
            return

        LOGGER.info("Persisting changes to the dataset...")
        self.log_status()

        if not name:
            name = self.data.split(".")[0] + "_updated"
        datafile_name = name + ".yaml"
        train_txt = "train_updated.txt"

        path = Path(name).resolve()  # add new train.txt file in the dataset parent path
        path.mkdir(parents=True, exist_ok=True)
        if (path / train_txt).exists():
            (path / train_txt).unlink()  # remove existing

        for img in tqdm(self.table.to_pandas()["path"].to_list()):
            with open(path / train_txt, "a") as f:
                f.write(f"{img}" + "\n")  # add image to txt file

        new_dataset_info = self.dataset_info.copy()
        new_dataset_info.pop("yaml_file")
        new_dataset_info.pop(
            "path"
        )  # relative paths will get messed up when merging datasets
        new_dataset_info.pop(
            "download"
        )  # Assume all files are present offline, there is no way to store metadata yet
        new_dataset_info["train"] = (path / train_txt).resolve().as_posix()
        for key, value in new_dataset_info.items():
            if isinstance(value, Path):
                new_dataset_info[key] = value.as_posix()

        yaml.dump(
            new_dataset_info, open(path / datafile_name, "w")
        )  # update dataset.yaml file

        # TODO: not sure if this should be called data_final to prevent overwriting the original data?
        self.table = self._create_table(
            self.table_name, data=self.table.to_arrow(), mode="overwrite"
        )
        db.drop_table(self.temp_table_name)

        LOGGER.info("Changes persisted to the dataset.")
        self._log_training_cmd(Path(path / datafile_name).relative_to(Path.cwd()).as_posix())

    def log_status(self):
        # TODO: Pretty print log status
        LOGGER.info("\n|-----------------------------------------------|")
        LOGGER.info(f"\t Number of images: {len(self.table.to_arrow())}")
        LOGGER.info("|------------------------------------------------|")

    def sql(self, query: str):
        """
        Executes a SQL query on the dataset table.

        Args:
            query (str): SQL query to execute.
        """
        if self.table is None:
            LOGGER.info("No table found. Please provide a dataset to work on.")
            return

        table = self.table.to_arrow()  # noqa
        result = duckdb.sql(query).to_df()

        return result

    def _log_training_cmd(self, data_path):
        LOGGER.info(
            f'{colorstr("LanceDB: ") }New dataset created successfully! Run the following command to train a model:'
        )
        LOGGER.info(f"yolo train data={data_path} epochs=10")

    def _connect(self):
        db = lancedb.connect(self.project)

        return db

    def _create_table(self, name, data=None, mode="overwrite"):
        db = lancedb.connect(self.project)
        table = db.create_table(name, data=data, mode=mode)

        return table

    def _open_table(self, name):
        db = lancedb.connect(self.project)
        table = db.open_table(name) if name in db.table_names() else None
        if table is None:
            raise ValueError(f'{colorstr("LanceDB: ") }Table not found.')
        return table

    def _drop_table(self, name):
        db = lancedb.connect(self.project)
        if name in db.table_names():
            db.drop_table(name)
            return True

        return False

    def _copy_table_to_project(self, table_path):
        if not table_path.endswith(".lance"):
            raise ValueError(f"{colorstr('LanceDB: ')} Table must be a .lance file")

        LOGGER.info(f"Copying table from {table_path}")
        path = Path(table_path).parent
        name = Path(table_path).stem  # lancedb doesn't need .lance extension
        db = lancedb.connect(path)
        table = db.open_table(name)
        return self._create_table(
            self.table_name, data=table.to_arrow(), mode="overwrite"
        )

    def _embedding_func(self, imgs):
        embeddings = []
        for img in tqdm(imgs):
            img = decode(img)
            embeddings.append(
                self.predictor.embed(img, verbose=self.verbose).squeeze().cpu().numpy()
            )
        return embeddings

    def _setup_predictor(self, model, device=""):
        model = YOLO(model)
        predictor = YOLOEmbeddingsPredictor(overrides={"device": device})
        predictor.setup_model(model.model)

        return predictor

    def create_index(self):
        # TODO: create index
        pass


if __name__ == "__main__":
    voc_table = Explorer("coco128.yaml")
    voc_table.build_embeddings(force=True)
    voc_table.remove_imgs([7,10,11,12,13,14,15,16,17,18,19,20,21,22,23])
    # import pdb;pdb.set_trace()
