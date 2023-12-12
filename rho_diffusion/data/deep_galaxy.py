from __future__ import annotations

import os
import re
from typing import Union

import h5py
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset

from rho_diffusion import utils
from rho_diffusion.registry import registry


@registry.register_dataset("DeepGalaxyDataset")
class DeepGalaxyDataset(Dataset):
    def __init__(
        self,
        path: str,
        use_emb_as_labels: bool = True,
        dset_name_pattern: str = "s_*",
        camera_pos: list = [0],
        t_lim: list = None,
        transform: torchvision.transforms.transforms.Compose = None,
        target_transform: torchvision.transforms.transforms.Compose = None,
    ):
        self.h5fn = path
        self.use_emb_labels = use_emb_as_labels

        self.data = None
        self.labels = None
        self.num_classes = 0
        self.dset_name_pattern = (dset_name_pattern,)
        self.camera_pos = camera_pos
        self.t_lim = t_lim
        self.parameter_space = {
            "s": [],
            "m": [],
            "t": [],
            "c": [],
        }

        if transform is None:
            transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.CenterCrop(256),
                    torchvision.transforms.Resize((128, 128)),
                    torchvision.transforms.Lambda(
                        lambda t: (t * 2) - 1,
                    ),  # Scale between [-1, 1]
                ],
            )
        self.transform = transform
        self.target_transform = target_transform

        self.load(dset_name_pattern, camera_pos, t_lim)

    def __len__(self):
        if self.data is None:
            self.data, self.labels = self.load()

        return self.data.shape[0]

    def __getitem__(self, idx):
        if self.data is None:
            self.data, self.labels = self.load()
        image = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def load(self, dset_name_pattern, camera_pos, t_lim):
        data, labels = self._load_all(
            dset_name_pattern=dset_name_pattern,
            camera_pos=camera_pos,
            t_lim=t_lim,
        )
        # self.data = torch.ByteTensor(data)
        self.data = torch.FloatTensor(data.swapaxes(1, 3))
        # self.data = data
        print(self.data.shape, labels.shape)

        if self.use_emb_labels:
            self.labels = torch.FloatTensor(labels)
        else:
            self.labels = torch.LongTensor(labels)

    @staticmethod
    def calculate_embeddings(
        s: int | list | np.ndarray,
        m: int | list | np.ndarray,
        t: int | list | np.ndarray,
        c: int | list | np.ndarray,
    ) -> torch.Tensor:
        assert (
            type(s) == type(m) == type(t) == type(c)
        ), "Data type of the arguments must agree"
        if type(s) == int:
            label_dict = {
                "m": s,
                "s": m,
                "t": t,
                "c": c,
            }
            return utils.calculate_sha512_embedding(label_dict, l=512)
        else:
            label_embs = []
            for i in range(len(s)):
                label_dict = {
                    "m": int(m[i]),
                    "s": int(s[i]),
                    "t": int(t[i]),
                    "c": int(c[i]),
                }
                print(label_dict)
                label_embs.append(utils.calculate_sha512_embedding(label_dict, l=128))
            return torch.FloatTensor(np.array(label_embs, dtype=np.float32))

    def _load_all(self, dset_name_pattern="*", camera_pos="*", t_lim=None):
        """
        Load all data that match the dataset name pattern.
        """
        self.h5f = h5py.File(self.h5fn, "r")
        full_dset_list = self.h5f.keys()

        r = re.compile(dset_name_pattern)
        matched_dset_list = list(filter(r.match, full_dset_list))
        print("Selected datasets: %s" % matched_dset_list)
        images_set = []
        labels_m_set = []
        labels_s_set = []
        labels_t_set = []
        labels_cpos_set = []
        labels_emb_set = []
        if isinstance(camera_pos, int):
            camera_pos = [camera_pos]  # convert to list
        elif isinstance(camera_pos, str):
            if camera_pos == "*":
                camera_pos = range(0, 14)
        print("Selected camera positions: %s" % camera_pos)
        for dset_name in matched_dset_list:
            for cpos in camera_pos:
                h5_path = "/%s/images_camera_%02d" % (dset_name, cpos)
                print("Loading dataset %s" % h5_path)
                images = self._load_dataset(h5_path)
                labels_m, labels_s, labels_t = self._get_labels(dset_name, cpos)
                labels_cpos = np.ones(labels_m.shape, dtype=np.int32) * cpos
                if t_lim is not None:
                    t_low, t_high = np.min(t_lim), np.max(t_lim)
                    flags = np.logical_and(labels_t >= t_low, labels_t <= t_high)
                    images = images[flags]
                    labels_t = labels_t[flags]
                    labels_m = labels_m[flags]
                    labels_s = labels_s[flags]
                    labels_cpos = labels_cpos[flags]
                labels_t_ = (labels_t / 5).astype(np.int32)
                labels_t_ = labels_t_ - np.min(labels_t_)
                # collapse classes
                # labels_t_ = labels_t_ // 10
                images_set.append(images)
                labels_m_set.append(labels_m)
                labels_s_set.append(labels_s)
                labels_t_set.append(labels_t_)
                labels_cpos_set.append(labels_cpos)
        if len(images_set) > 0:
            images_set = np.concatenate(images_set, axis=0)
            # labels_emb_set = np.concatenate(labels_emb_set, axis=0)
            labels_m_set = np.concatenate(labels_m_set, axis=0)
            labels_s_set = np.concatenate(labels_s_set, axis=0)
            labels_t_set = np.concatenate(labels_t_set, axis=0)
            labels_cpos_set = np.concatenate(labels_cpos_set, axis=0)

        # compute label dict and the sha512 embeddings
        for i in range(len(labels_m_set)):
            label_dict = {
                "m": int(labels_m_set[i]),
                "s": int(labels_s_set[i]),
                "t": int(labels_t_set[i]),
                "c": int(labels_cpos_set[i]),
            }

            label_emb = utils.calculate_sha512_embedding(label_dict, l=128)
            labels_emb_set.append(label_emb)

            # Add to the parameter space
            if int(labels_m_set[i]) not in self.parameter_space["m"]:
                self.parameter_space["m"].append(int(labels_m_set[i]))

            if int(labels_s_set[i]) not in self.parameter_space["s"]:
                self.parameter_space["s"].append(int(labels_s_set[i]))

            if int(labels_t_set[i]) not in self.parameter_space["t"]:
                self.parameter_space["t"].append(int(labels_t_set[i]))

            if int(labels_cpos_set[i]) not in self.parameter_space["c"]:
                self.parameter_space["c"].append(int(labels_cpos_set[i]))

        # labels_emb_set.append(labels_emb)
        num_classes = np.unique(labels_t_set).shape[0]
        print("Number of classes:", num_classes)

        self.parameter_space["m"] = sorted(self.parameter_space["m"])
        self.parameter_space["s"] = sorted(self.parameter_space["s"])
        self.parameter_space["t"] = sorted(self.parameter_space["t"])
        self.parameter_space["c"] = sorted(self.parameter_space["c"])

        print(self.parameter_space)

        self.num_classes = num_classes
        if self.use_emb_labels:
            return images_set, np.array(labels_emb_set)
        else:
            return (
                images_set,
                torch.LongTensor(labels_t_set),
            )

    def _load_dataset(self, h5_path):
        """
        Load a dataset according to the h5_path.
        """
        images = self.h5f[h5_path][()]
        images = images / np.max(images)
        # if images.dtype == np.uint8:
        #     # if uint8, then the range is [0, 255]. Normalize to [0, 1]
        #     # if float32, don't do anything since the range is already [0, 1]
        #     images = images.astype(np.float) / 255
        # elif images.shape[-1] == 1:
        #     # just gray scale float images. Repeat along the channel
        #     # old_shape = images.shape
        #     # images = np.repeat(images.astype(np.float32), 3, axis=(len(old_shape)-1))
        #     # print('Repeating...', old_shape, images.shape)
        #     pass
        return images

    def _get_labels(self, dset_name, camera_pos=0):
        print("Getting labels...")
        size_ratios = np.array([0.25, 0.5, 0.75, 1, 1.25, 1.5])
        mass_ratios = np.array([0.25, 0.5, 0.75, 1, 1.25, 1.5])
        s = float(dset_name.split("_")[1])
        m = float(dset_name.split("_")[3])
        cat_t = self.h5f["%s/t_myr_camera_%02d" % (dset_name, camera_pos)][()]
        cat_s = np.argwhere(size_ratios == s)[0, 0] * np.ones(
            cat_t.shape,
            dtype=np.int32,
        )
        cat_m = np.argwhere(mass_ratios == m)[0, 0] * np.ones(
            cat_t.shape,
            dtype=np.int32,
        )
        return cat_m, cat_s, cat_t
