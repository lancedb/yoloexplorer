import json

import numpy as np
import streamlit as st

from yoloexplorer import config
from yoloexplorer.frontend.datasets import _get_primary_dataset

INTEGRATION_IMPORT_ERROR = None

try:
    import data_gradients #noqa

    import data_gradients.feature_extractors.object_detection as detection
    from data_gradients.datasets.detection import YoloFormatDetectionDataset
except ImportError:
    INTEGRATION_IMPORT_ERROR = "data-gradients"

@st.cache_data
def _get_config():
    with open(config.TEMP_CONFIG_PATH) as json_file:
        data = json.load(json_file)
    return data["analysis"]

@st.cache_data
def _get_task_from_data(data):
    # TODO: support more tasks
    return "detection"

DETECTION = {
    "DetectionBoundingBoxArea": detection.DetectionBoundingBoxArea,
    "DetectionBoundingBoxPerImageCount": detection.DetectionBoundingBoxPerImageCount,
    "DetectionBoundingBoxSize": detection.DetectionBoundingBoxSize,
    "DetectionClassFrequency": detection.DetectionClassFrequency,
    "DetectionClassHeatmap": detection.DetectionClassHeatmap,
    "DetectionClassesPerImageCount": detection.DetectionClassesPerImageCount,
    "DetectionSampleVisualization": detection.DetectionSampleVisualization,
    "DetectionBoundingBoxIoU": detection.DetectionBoundingBoxIoU,
} if INTEGRATION_IMPORT_ERROR is None else {}

SEGMENTATION = {

} if INTEGRATION_IMPORT_ERROR is None else {}

TASK2MODULES = { "detection": DETECTION, "segmentation": SEGMENTATION}
TASK2LABELS = {"detection": "bboxes", "segmentation": "masks"}

@st.cache_resource
def analyse_dataset():
    exp = _get_primary_dataset()
    info = exp.dataset_info
    deci_ds = YoloFormatDetectionDataset(root_dir=info["path"], 
                                         images_dir=info["image_dir"],)

def analysis():
    if not _get_config():
        st.error("Enable analysis by passing `analysis=True` when launching the dashboard.")
        return

    if INTEGRATION_IMPORT_ERROR:
        st.error(f"The following package(s) are required to run this module: `{INTEGRATION_IMPORT_ERROR}`. Please install them and try again.")
        return
    
    task = _get_task_from_data(_get_primary_dataset())
    modules = TASK2MODULES[task]
    results = []
    for _, module in modules.items():
        pass


if __name__ == "__main__":
    analysis()

    

