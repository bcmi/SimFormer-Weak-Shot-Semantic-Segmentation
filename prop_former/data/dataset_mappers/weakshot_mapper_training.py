# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging

import numpy as np
import torch
from torch.nn import functional as F
import os
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Instances

__all__ = ["WeakShotMapperTraining"]


class WeakShotMapperTraining:
    @configurable
    def __init__(self, cfg=None, is_train=True, *,
                 augmentations, image_format, ignore_label, size_divisibility, meta, ):
        self.cfg = cfg
        self.is_train = is_train
        self.tfm_gens = augmentations
        self.img_format = image_format
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility
        self.meta = meta
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}")
        self.logger = logger

        self.existing_count = 0
        self.updated_count = 0

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        augs = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]
        if cfg.INPUT.CROP.ENABLED:
            augs.append(
                T.RandomCrop_CategoryAreaConstraint(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                )
            )
        if cfg.INPUT.COLOR_AUG_SSD:
            augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
        augs.append(T.RandomFlip())

        # Assume always applies to the training set.
        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])
        ignore_label = meta.ignore_label

        ret = {
            "cfg": cfg,
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
            "meta": meta,
        }
        return ret

    def get_valid_aug_image_mask(self, raw_image, raw_mask):

        aug_input = T.AugInput(raw_image, sem_seg=raw_mask)
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)

        return aug_input

    def __call__(self, dataset_dict):
        assert self.is_train, "DatasetMapper should only be used for training!"

        if self.cfg.ALL_EXISTING:
            dataset_dict['type'] = 'existing'

        if dataset_dict['type'] == 'existing':
            self.existing_count += 1
        else:
            self.updated_count += 1

        if self.updated_count >= 1000:
            ostr = f'Updated: {self.updated_count}; Existing: {self.existing_count}'
            logger = logging.getLogger('detectron2')
            logger.info(ostr)
            self.existing_count = 0
            self.updated_count = 0

        if self.cfg.NOVEL_HAS_MASK:
            novel_has_mask = True
        else:
            novel_has_mask = False

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        assert "sem_seg_file_name" in dataset_dict

        if self.cfg.PSEUDO_LABEL_PATH == 'none':
            raw_segm_gt = utils.read_image(dataset_dict['sem_seg_file_name']).astype("double")
        else:
            gt_labels_path = dataset_dict['sem_seg_file_name']
            pseudo_label_path = f'datasets/{self.cfg.PSEUDO_LABEL_PATH}/{os.path.basename(gt_labels_path)}'
            # assert os.path.exists(pseudo_label_path), f'Not found: {pseudo_label_path}'
            if os.path.exists(pseudo_label_path):
                label_path = pseudo_label_path
                # print(f'Use Pseudo for {os.path.basename(gt_labels_path)}!')
            else:
                # Because this image has no novel class, so there is no pseudo label for this image.
                print(f'Use GT for {os.path.basename(gt_labels_path)}!')
                label_path = gt_labels_path

            raw_segm_gt = utils.read_image(label_path).astype("double")

        max_try = 50
        for repeat_idx in range(max_try):
            if repeat_idx > 30:
                raw_segm_gt = utils.read_image(dataset_dict['sem_seg_file_name']).astype("double")

            aug_input = self.get_valid_aug_image_mask(image, raw_segm_gt)

            valid = False
            for did in np.unique(aug_input.sem_seg):
                if did in self.meta.c_base_dids + self.meta.c_novel_dids:
                    valid = True

            if valid:
                break
            else:
                pass
            repeat_idx


        if not valid:
            self.logger.info(f'\nFail for repeat {repeat_idx}')
            self.logger.info(f"{np.unique(raw_segm_gt)}\n" \
                             f"{np.unique(aug_input.sem_seg)}\n" \
                             f"Fail in {dataset_dict['sem_seg_file_name']}\n")
            print(f'\nFail for repeat {repeat_idx}')
            print(f"{np.unique(raw_segm_gt)}\n" \
                  f"{np.unique(aug_input.sem_seg)}\n" \
                  f"Fail in {dataset_dict['sem_seg_file_name']}\n")
            raise ValueError(f"Still invalid for {max_try}:\n"
                             f"{np.unique(raw_segm_gt)}\n" \
                             f"{np.unique(aug_input.sem_seg)}\n" \
                             f"Fail in {dataset_dict['sem_seg_file_name']}\n")

        image = torch.as_tensor(np.ascontiguousarray(aug_input.image.transpose(2, 0, 1)))
        segm_gt_raw_aug = torch.as_tensor(aug_input.sem_seg.astype("long"))

        if self.size_divisibility > 0:
            image_size = (image.shape[-2], image.shape[-1])
            padding_size = [0, self.size_divisibility - image_size[1], 0, self.size_divisibility - image_size[0], ]
            pad_region = F.pad(torch.zeros_like(segm_gt_raw_aug), padding_size, value=1).contiguous()

            image = F.pad(image, padding_size, value=128).contiguous()
            segm_gt_raw_aug = F.pad(segm_gt_raw_aug, padding_size, value=self.ignore_label).contiguous()
            dataset_dict['pad_region'] = pad_region

        image_shape = (image.shape[-2], image.shape[-1])  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = image
        dataset_dict["segm_gt_raw_aug"] = segm_gt_raw_aug.long()

        # Prepare per-category binary masks
        segm_gt_raw_aug = segm_gt_raw_aug.numpy()
        instances = Instances(image_shape)

        if dataset_dict['type'] == 'existing':
            base_has_mask = True
        elif dataset_dict['type'] == 'updated':
            base_has_mask = False
        else:
            raise NotImplementedError

        ###### We must move the NoMask class to the end of list ##############
        # We employ zero mask for the class having no annotated mask

        labels_A, labels_B = [], []
        has_masks_A, has_masks_B = [], []
        masks_A, masks_B = [], []

        for did in np.unique(segm_gt_raw_aug):
            if did in self.meta.c_base_dids:
                has_mask = base_has_mask
            elif did in self.meta.c_novel_dids:
                has_mask = novel_has_mask
            else:
                # ignore
                continue

            if has_mask:
                mask = segm_gt_raw_aug == did
            else:
                mask = np.zeros(image_shape)
            cid = self.meta.c_did_to_cid[did]

            if has_mask:
                labels_A.append(cid)
                has_masks_A.append(has_mask)
                masks_A.append(mask)
            else:
                labels_B.append(cid)
                has_masks_B.append(has_mask)
                masks_B.append(mask)

        labels = labels_A + labels_B
        has_masks = has_masks_A + has_masks_B
        masks = masks_A + masks_B

        ###### We must move the NoMask class to the end of list ##############

        assert len(labels) >= 1, f"\n{np.unique(raw_segm_gt)}\n" \
                                 f"{np.unique(segm_gt_raw_aug)}\n" \
                                 f"No labels in {dataset_dict['sem_seg_file_name']}"

        instances.gt_classes = torch.tensor(labels, dtype=torch.int64)
        instances.has_masks = torch.tensor(has_masks, dtype=torch.bool)
        instances.gt_masks = BitMasks(
            torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])).tensor

        dataset_dict["instances"] = instances

        return dataset_dict
