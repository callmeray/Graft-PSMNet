import os
import random
from typing import Tuple
import cv2
import numpy as np
import torch
from loguru import logger
from path import Path
from PIL import Image
from skimage.feature import canny
from torch.utils.data import Dataset
from tqdm import tqdm

import pickle


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


class MessyTableDataset(Dataset):
    def __init__(
            self,
            mode: str,
            root_dir: str,
            split_file: str,
            height: int,
            width: int,
            meta_name: str,
            depth_name: str,
            left_name: str,
            right_name: str,
            label_name: str = "",
    ):
        self.mode = mode
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            logger.error(f"Not exists root dir: {self.root_dir}")

        self.split_file = split_file
        self.height, self.width = height, width
        self.meta_name = meta_name
        self.depth_name = depth_name
        self.left_name, self.right_name = left_name, right_name
        self.label_name = label_name

        self.img_dirs = self._gen_path_list()

        logger.info(
            f"MessyTableDataset: mode: {mode}, root_dir: {root_dir}, length: {len(self.img_dirs)},"
            f" left_name: {left_name}, right_name: {right_name},"
        )

    def _gen_path_list(self):
        img_dirs = []
        if not self.split_file:
            logger.warning(f"Split_file is not defined. The dataset is None.")
            return img_dirs
        with open(self.split_file, "r") as f_split_file:
            for l in f_split_file.readlines():
                img_dirs.append(self.root_dir / l.strip())

        check = False
        if check:
            print("Checking img dirs...")
            for d in tqdm(img_dirs):
                if not d.exists():
                    logger.error(f"{d} not exists.")

        return img_dirs

    def __getitem__(self, index):
        data_dict = {}
        img_dir = self.img_dirs[index]

        img_l = np.array(Image.open(img_dir / self.left_name).convert(mode="L")) / 255  # [H, W]
        img_r = np.array(Image.open(img_dir / self.right_name).convert(mode="L")) / 255

        origin_h, origin_w = img_l.shape[:2]  # (960, 540)
        if origin_h in (720, 1080):
            img_l = cv2.resize(img_l, (960, 540), interpolation=cv2.INTER_CUBIC)
            img_r = cv2.resize(img_r, (960, 540), interpolation=cv2.INTER_CUBIC)

        origin_h, origin_w = img_l.shape[:2]  # (960, 540)
        assert (
                origin_h == 540 and origin_w == 960
        ), f"Only support H=540, W=960. Current input: H={origin_h}, W={origin_w}"

        if self.meta_name:
            img_meta = load_pickle(img_dir / self.meta_name)
            extrinsic_l = img_meta["extrinsic_l"]
            extrinsic_r = img_meta["extrinsic_r"]
            intrinsic_l = img_meta["intrinsic_l"]
            intrinsic_l[:2] /= 2
            intrinsic_l[2] = np.array([0.0, 0.0, 1.0])
            baseline = np.linalg.norm(extrinsic_l[:, -1] - extrinsic_r[:, -1])
            focal_length = intrinsic_l[0, 0]
            if self.depth_name:
                img_depth_l = (
                        cv2.imread(img_dir / self.depth_name, cv2.IMREAD_UNCHANGED).astype(float) / 1000
                )  # convert from mm to m
                img_depth_l = cv2.resize(img_depth_l, (origin_w, origin_h), interpolation=cv2.INTER_NEAREST)
                mask = img_depth_l > 0
                img_disp_l = np.zeros_like(img_depth_l)
                img_disp_l[mask] = focal_length * baseline / img_depth_l[mask]

        if self.label_name:
            img_label_l = cv2.imread(img_dir / self.label_name, cv2.IMREAD_UNCHANGED).astype(int)
            img_label_l = cv2.resize(img_label_l, (origin_w, origin_h), interpolation=cv2.INTER_NEAREST)

        # random crop
        if self.mode == "test":
            x = 0
            y = -2
            assert self.height == 544 and self.width == 960, f"Only support H=544, W=960 for now"

            def crop(img):
                if img.ndim == 2:
                    img = np.concatenate(
                        [np.zeros((2, 960), dtype=img.dtype), img, np.zeros((2, 960), dtype=img.dtype)]
                    )
                else:
                    img = np.concatenate(
                        [
                            np.zeros((2, 960, img.shape[2]), dtype=img.dtype),
                            img,
                            np.zeros((2, 960, img.shape[2]), dtype=img.dtype),
                        ]
                    )
                return img

            def crop_label(img):
                img = np.concatenate(
                    [
                        np.ones((2, 960), dtype=img.dtype) * self.num_classes,
                        img,
                        np.ones((2, 960), dtype=img.dtype) * self.num_classes,
                    ]
                )
                return img

        else:
            x = np.random.randint(0, origin_w - self.width)
            y = np.random.randint(0, origin_h - self.height)

            def crop(img):
                return img[y: y + self.height, x: x + self.width]

            def crop_label(img):
                return img[y: y + self.height, x: x + self.width]

        img_l = crop(img_l)
        img_r = crop(img_r)
        if self.depth_name and self.meta_name:
            intrinsic_l[0, 2] -= x
            intrinsic_l[1, 2] -= y
            img_depth_l = crop(img_depth_l)
            img_disp_l = crop(img_disp_l)

        if self.label_name:
            img_label_l = crop_label(img_label_l)

        data_dict["dir"] = img_dir.name
        data_dict["full_dir"] = str(img_dir)
        data_dict["img_l"] = torch.from_numpy(img_l).float().expand(3, -1, -1)
        data_dict["img_r"] = torch.from_numpy(img_r).float().expand(3, -1, -1)
        if self.meta_name:
            data_dict["intrinsic_l"] = torch.from_numpy(intrinsic_l).float()
            data_dict["baseline"] = torch.tensor(baseline).float()
            data_dict["focal_length"] = torch.tensor(focal_length).float()
            data_dict["extrinsic_l"] = torch.from_numpy(img_meta["extrinsic_l"])
            data_dict["extrinsic"] = torch.from_numpy(img_meta["extrinsic"])
            data_dict["intrinsic"] = torch.from_numpy(img_meta["intrinsic"])
            if self.depth_name:
                data_dict["img_depth_l"] = torch.from_numpy(img_depth_l).float()
                data_dict["img_disp_l"] = torch.from_numpy(img_disp_l).float()

        if self.label_name:
            data_dict["img_label_l"] = torch.from_numpy(img_label_l).long()

        return data_dict

    def __len__(self):
        return len(self.img_dirs)
