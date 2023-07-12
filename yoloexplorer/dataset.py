import os
import cv2
import numpy as np

from ultralytics.yolo.data.dataset import YOLODataset
from ultralytics.yolo.data.augment import Format
from ultralytics.yolo.data.utils import check_det_dataset


def get_dataset_info(data="coco128.yaml", task="detect"):
    # TODO: handle other tasks
    data = check_det_dataset(data)

    return data


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
        super().__init__(
            *args, data=data, use_segments=False, use_keypoints=False, **kwargs
        )

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
