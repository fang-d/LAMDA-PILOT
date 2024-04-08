# -*- coding: utf-8 -*-
"""
Proper implementation of the ACIL [1].

This implementation refers to the official implementation https://github.com/ZHUANGHP/Analytic-continual-learning.

References:
[1] Zhuang, Huiping, et al.
    "ACIL: Analytic class-incremental learning with absolute memorization and privacy protection."
    Advances in Neural Information Processing Systems 35 (2022): 11602-11614.
"""

import torch
import logging
import numpy as np
from tqdm import tqdm
from models.base import BaseLearner
from utils.inc_net import BaseNet
from backbone.buffer import RandomBuffer
from typing import Dict, Any, Sized, List
from torch.utils.data import DataLoader, Sampler
from backbone.analytic_linear import RecursiveLinear
from utils.data_manager import DataManager, DummyDataset
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


__all__ = [
    "ACIL",
    "ACILNet",
    "InplaceRepeatSampler",
]


class InplaceRepeatSampler(Sampler):
    def __init__(self, data_source: Sized, num_repeats: int = 1):
        self.data_source = data_source
        self.num_repeats = num_repeats

    def __iter__(self):
        for i in range(len(self.data_source)):
            for _ in range(self.num_repeats):
                yield i

    def __len__(self):
        return len(self.data_source) * self.num_repeats


class ACILNet(BaseNet):
    def __init__(
        self,
        args: Dict[str, Any],
        buffer_size: int = 8192,
        gamma: float = 0.1,
        pretrained: bool = False,
        device=None,
        dtype=torch.double,
    ) -> None:
        super().__init__(args, pretrained)
        assert isinstance(
            self.backbone, torch.nn.Module
        ), "The backbone network must be a `torch.nn.Module`."
        self.backbone: torch.nn.Module = self.backbone.to(device, non_blocking=True)

        self.args = args
        self.buffer_size: int = buffer_size
        self.gamma: float = gamma
        self.device = device
        self.dtype = dtype

        config = resolve_data_config(self.backbone.pretrained_cfg, model=self.backbone)
        self.backbone_transform = create_transform(**config).transforms

    @torch.no_grad()
    def forward(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        X = self.backbone(X)
        X = self.buffer(X)
        X = self.fc(X)
        return {"logits": X}

    def update_fc(self, nb_classes: int) -> None:
        self.fc.update_fc(nb_classes)

    def generate_fc(self, *_) -> None:
        self.fc = RecursiveLinear(
            self.buffer_size,
            self.gamma,
            bias=False,
            device=self.device,
            dtype=self.dtype,
        )

    def generate_buffer(self) -> None:
        self.buffer = RandomBuffer(
            self.feature_dim, self.buffer_size, device=self.device, dtype=self.dtype
        )

    def after_task(self) -> None:
        self.fc.after_task()

    @torch.no_grad()
    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        X = self.backbone(X)
        X = self.buffer(X)
        Y: torch.Tensor = torch.nn.functional.one_hot(y, self.fc.out_features)
        self.fc.fit(X, Y)


class ACIL(BaseLearner):
    def __init__(self, args: Dict[str, Any]) -> None:
        if "memory_size" not in args:
            args["memory_size"] = 0
        elif args["memory_size"] != 0:
            raise ValueError(
                f"{self.__class__.__name__} is an exemplar-free method,"
                "so the `memory_size` must be 0."
            )
        super().__init__(args)
        self.parse_args(args)
        self.create_network()
        # As a simple example, we freeze the backbone network of the AL-based CIL methods.
        self._network.freeze()

    def parse_args(self, args: Dict[str, Any]) -> None:
        # Bigger batch size leads faster learning speed, >= 4096 for ImageNet.
        self.batch_size: int = args["batch_size"]
        # 8192 for CIFAR-100, and 16384 for ImageNet
        self.buffer_size: int = args["buffer_size"]
        # Regularization term of the regression
        self.gamma: float = args["gamma"]
        # Inplace repeat sampler for the training data loader during incremental learning
        self.inplace_repeat: int = args.get("inplace_repeat", 1)
        # Num workers
        self.num_workers: int = args.get("num_workers", 4)

    def create_network(self) -> None:
        self._network = ACILNet(
            self.args,
            buffer_size=self.buffer_size,
            pretrained=False,
            gamma=self.gamma,
            device=self._device,
        )

        self._network.generate_buffer()
        self._network.generate_fc()

        if len(self._multiple_gpus) > 1:
            self._network.backbone = torch.nn.DataParallel(
                self._network.backbone, self._multiple_gpus
            )

    def incremental_train(self, data_manager: DataManager) -> None:
        self._cur_task += 1
        if self._cur_task == 0:
            # As the AL-based methods for large pre-train models frozen the backbone network,
            # we replace the default transform with the transform provided by timm.
            data_manager._test_trsf = []
            data_manager._common_trsf = self._network.backbone_transform

        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        test_dataset: DummyDataset = data_manager.get_dataset(
            np.arange(0, self._total_classes),
            source="test",
            mode="test",
            ret_data=False,
        )  # type: ignore

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=(self.num_workers == 0),
        )

        self._network.to(self._device)

        train_dataset: DummyDataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="test",
            ret_data=False,
        )  # type: ignore

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=InplaceRepeatSampler(train_dataset, self.inplace_repeat),
        )

        self._train(
            train_loader,
            desc="Base Re-align" if self._cur_task == 0 else "Incremental Learning",
        )

    @torch.no_grad()
    def _train(
        self, train_loader: DataLoader, desc: str = "Incremental Learning"
    ) -> None:
        self._network.eval()
        self._network.update_fc(self._total_classes)
        for _, X, y in tqdm(train_loader, desc=desc):
            X: torch.Tensor = X.to(self._device, non_blocking=True)
            y: torch.Tensor = y.to(self._device, non_blocking=True)
            self._network.fit(X, y)
        self._network.after_task()

    def after_task(self) -> None:
        self._known_classes = self._total_classes
        self._network.after_task()
