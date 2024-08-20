# Copyright (C) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
#
#
# SPDX-License-Identifier: Apache-2.


from __future__ import annotations

import os
import re
from typing import Union

import h5py
import numpy as np
import torch
import torchvision


from rho_diffusion import utils
from rho_diffusion.registry import registry
from rho_diffusion.models.conditioning import MultiEmbeddings
from rho_diffusion.data.base import MultiVariateDataset
from rho_diffusion.data.parameter_space import DiscreteParameterSpace


@registry.register_dataset("DeepGalaxyDataset")
class DeepGalaxyDataset(MultiVariateDataset):

    parameter_space = DiscreteParameterSpace(
        param_dict={'s': [0.25, 0.5, 0.75, 1, 1.25, 1.5], 
                    'm': [0.25, 0.5, 0.75, 1, 1.25, 1.5], 
                    't': list(range(300, 655, 5)), 
                    'c': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
                    },
        sampler=None)



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
        self.loaded_parameter_space = DiscreteParameterSpace(
            param_dict = {
                "s": [],
                "m": [],
                "t": [],
                "c": [],
            }
        )
        self.attributes = ['s', 'm', 't', 'c']
        

    
        

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

        # if self.use_emb_labels:
        #     self.labels = torch.FloatTensor(labels)
        # else:
            # self.labels = torch.LongTensor(labels)
        self.labels = labels

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
                # labels_t_ = (labels_t / 5).astype(np.int32)
                # labels_t_ = labels_t_ - np.min(labels_t_)
                # collapse classes
                # labels_t_ = labels_t_ // 10
                images_set.append(images)
                labels_m_set.append(labels_m)
                labels_s_set.append(labels_s)
                labels_t_set.append(labels_t)
                labels_cpos_set.append(labels_cpos)
        if len(images_set) > 0:
            images_set = np.concatenate(images_set, axis=0)
            # labels_emb_set = np.concatenate(labels_emb_set, axis=0)
            labels_m_set = np.concatenate(labels_m_set, axis=0)
            labels_s_set = np.concatenate(labels_s_set, axis=0)
            labels_t_set = np.concatenate(labels_t_set, axis=0)
            labels_cpos_set = np.concatenate(labels_cpos_set, axis=0)

        # compute the parameter space dynamically according to the loaded data
        self.loaded_parameter_space['m'] = np.unique(labels_m_set)
        self.loaded_parameter_space['s'] = np.unique(labels_s_set)
        self.loaded_parameter_space['t'] = np.unique(labels_t_set)
        self.loaded_parameter_space['c'] = np.unique(labels_cpos_set)
        # for i in range(len(labels_m_set)):
        #     # label_dict = {
        #     #     "m": int(labels_m_set[i]),
        #     #     "s": int(labels_s_set[i]),
        #     #     "t": int(labels_t_set[i]),
        #     #     "c": int(labels_cpos_set[i]),
        #     # }

        #     # label_emb = utils.calculate_sha512_embedding(label_dict, l=128)
        #     # labels_emb_set.append(label_emb)

        #     # Add to the parameter space
        #     if int(labels_m_set[i]) not in self.loaded_parameter_space["m"]:
        #         self.loaded_parameter_space["m"].append(labels_m_set[i])

        #     if int(labels_s_set[i]) not in self.loaded_parameter_space["s"]:
        #         self.loaded_parameter_space["s"].append(labels_s_set[i])

        #     if int(labels_t_set[i]) not in self.loaded_parameter_space["t"]:
        #         self.loaded_parameter_space["t"].append(labels_t_set[i])

        #     if int(labels_cpos_set[i]) not in self.loaded_parameter_space["c"]:
        #         self.loaded_parameter_space["c"].append(labels_cpos_set[i])

        # labels_emb_set.append(labels_emb)
        num_classes = np.unique(labels_t_set).shape[0]

        self.loaded_parameter_space["m"] = sorted(self.loaded_parameter_space["m"])
        self.loaded_parameter_space["s"] = sorted(self.loaded_parameter_space["s"])
        self.loaded_parameter_space["t"] = sorted(self.loaded_parameter_space["t"])
        self.loaded_parameter_space["c"] = sorted(self.loaded_parameter_space["c"])

        print(self.loaded_parameter_space)

        # multi_label_embeddings = MultiEmbeddings(self.loaded_parameter_space, embedding_dim=128)
        loaded_parameters = {
            "m": labels_m_set,
            "s": labels_s_set,
            "t": labels_t_set,
            "c": labels_cpos_set
         }
        # cat_tensor = multi_label_embeddings.transform_to_categorical(loaded_parameters)

        # Pack the loaded parameters into a float tensor
        labels_tensor = torch.zeros((len(labels_m_set), len(self.attributes)), dtype=torch.float)
        for i, attr in enumerate(self.attributes):
            labels_tensor[:, i] = torch.tensor(loaded_parameters[attr], dtype=torch.float)


        self.num_classes = num_classes
        # if self.use_emb_labels is True:
        #     return images_set, np.array(labels_emb_set)
        # else:
        return (
            images_set,
            # torch.LongTensor(labels_t_set),
            labels_tensor
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
        # size_ratios = np.array([0.25, 0.5, 0.75, 1, 1.25, 1.5])
        # mass_ratios = np.array([0.25, 0.5, 0.75, 1, 1.25, 1.5])
        s = float(dset_name.split("_")[1])
        m = float(dset_name.split("_")[3])
        cat_t = self.h5f["%s/t_myr_camera_%02d" % (dset_name, camera_pos)][()]
        cat_s = np.array([s] * cat_t.shape[0])
        cat_m = np.array([m] * cat_t.shape[0])
        # cat_s = np.argwhere(size_ratios == s)[0, 0] * np.ones(
        #     cat_t.shape,
        #     dtype=np.int32,
        # )
        # cat_m = np.argwhere(mass_ratios == m)[0, 0] * np.ones(
        #     cat_t.shape,
        #     dtype=np.int32,
        # )
        return cat_m, cat_s, cat_t
