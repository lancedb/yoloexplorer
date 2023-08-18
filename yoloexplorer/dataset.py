import os
import cv2
import numpy as np
import torch
import supervision as sv

from ultralytics.yolo.data.dataset import YOLODataset
from ultralytics.yolo.data.augment import Format
from ultralytics.yolo.data.utils import check_det_dataset
from ultralytics.yolo.utils import yaml_save


def get_dataset_info(data="coco128.yaml", task="detect"):
    # TODO: handle other tasks
    filepath = data
    data = check_det_dataset(data)
    yaml_save(data=data, file=filepath)
    return data


def get_label_directory(image_direcyory):
    return image_direcyory.replace("images", "labels")


def get_relative_path(path1, path2):
    """Gets the relative path of `path1` to `path2`.

    Args:
    path1: The absolute path of the first file.
    path2: The absolute path of the second file.

    Returns:
    The relative path of `path1` to `path2`.
    """

    relative_path = os.path.relpath(path1, os.path.dirname(path2))

    return relative_path


class Dataset(YOLODataset):
    def __init__(self, *args, data=None, **kwargs):
        super().__init__(*args, data=data, use_segments=False, use_keypoints=False, **kwargs)

    # NOTE: Load the image directly without any resize operations.
    def load_image(self, i):
        """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                im = np.load(fn)
            else:  # read image
                im = cv2.imread(f)  # BGR
                if im is None:
                    raise FileNotFoundError(f"Image Not Found {f}")
            h0, w0 = im.shape[:2]  # orig hw
            return im, (h0, w0), im.shape[:2]

        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    def build_transforms(self, hyp=None):
        transforms = Format(
            bbox_format="xyxy",
            normalize=False,
            return_mask=self.use_segments,
            return_keypoint=self.use_keypoints,
            batch_idx=True,
            mask_ratio=hyp.mask_ratio,
            mask_overlap=hyp.overlap_mask,
        )
        return transforms


class SupervisionDetectionDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dataset_info, data="coco128.yaml", task="detect"):

        trainsets = dataset_info["train"]
        trainsets = trainsets if isinstance(trainsets, list) else [trainsets]

        datasets = []
        for trainset in trainsets:
            _dataset = sv.DetectionDataset.from_yolo(images_directory_path=trainset,
                                                     annotations_directory_path=get_label_directory(trainset),
                                                     data_yaml_path=data)
            datasets.append(_dataset)

        self.ds = sv.DetectionDataset.merge(dataset_list=datasets)
        self.classes = self.ds.classes
        self.ni = len(self.ds)
        self.indices = range(len(self.ds.images.keys()))

    def __len__(self):
        return len(self.ds.images.keys())

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()[0]

        index = self.indices[idx]  # linear, shuffled, or image_weights
        image_name = list(self.ds.images.keys())[index]
        img = self.ds.images[image_name]
        detections = self.ds.annotations[image_name]
        batch = {}
        batch["im_file"] = image_name
        batch["img"] = torch.from_numpy(img)
        batch["ori_shape"] = img.shape[:2]
        batch["bboxes"] = torch.from_numpy(detections.xyxy)
        batch["cls"] = torch.from_numpy(detections.class_id)
        batch["path"] = image_name
        return batch

    def load_image(self, i):
        """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
        im = self.ds.images[i]
        h0, w0 = im.shape[:2]  # orig hw
        return im, (h0, w0), im.shape[:2]
