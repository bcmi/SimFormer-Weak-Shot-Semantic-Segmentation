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

__all__ = ["WeakShotMapperTrainingPair"]


class WeakShotMapperTrainingPair:
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

        dpath = f'datasets/imname_to_pair_list_dict_{self.meta.name}.pth'
        assert os.path.exists(dpath), f'Not Found: {dpath}'
        self.deconf_dict = torch.load(dpath)

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

    def get_data_dict_from_file_paths(self, img_path, ant_path):
        dataset_dict = {'file_name': img_path,
                        'sem_seg_file_name': ant_path}

        assert self.is_train, "DatasetMapper should only be used for training!"

        if self.cfg.ALL_EXISTING:
            dataset_dict['type'] = 'existing'

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
            pseudo_label_path = gt_labels_path.replace('annotations_detectron2/train', self.cfg.PSEUDO_LABEL_PATH)
            # assert os.path.exists(pseudo_label_path), f'Not found: {pseudo_label_path}'

            if os.path.exists(pseudo_label_path):
                label_path = pseudo_label_path
            else:
                # Because this image has no novel class.
                label_path = gt_labels_path

            raw_segm_gt = utils.read_image(label_path).astype("double")

        max_try = 50
        for repeat_idx in range(max_try):

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

        assert len(labels) >= 1, f"\n{np.unique(raw_segm_gt)}\n" \
                                 f"{np.unique(segm_gt_raw_aug)}\n" \
                                 f"No labels in {dataset_dict['sem_seg_file_name']}"

        instances.gt_classes = torch.tensor(labels, dtype=torch.int64)
        instances.has_masks = torch.tensor(has_masks, dtype=torch.bool)
        instances.gt_masks = BitMasks(
            torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])).tensor

        dataset_dict["instances"] = instances

        return dataset_dict

    def __call__(self, dataset_dict):
        img1_path = dataset_dict['file_name']
        ant1_path = dataset_dict['sem_seg_file_name']
        data1_dict = self.get_data_dict_from_file_paths(img1_path, ant1_path)

        key_name = os.path.basename(ant1_path)
        ant2_name = self.sample_pair_img(key_name)
        ant2_path = f'{os.path.dirname(ant1_path)}/{ant2_name}'
        img2_path = img1_path.replace(key_name.split('.')[0], ant2_name.split('.')[0])
        assert os.path.exists(ant2_path), f'File Not Found: {ant2_path}; {key_name}'
        assert os.path.exists(img2_path), f'File Not Found: {img2_path}: {key_name}'

        data2_dict = self.get_data_dict_from_file_paths(img2_path, ant2_path)
        data1_dict['paired_dict'] = data2_dict
        return data1_dict

    def sample_pair_img(self, key_name):
        deconf_d = self.deconf_dict[key_name]

        if self.cfg.CROSS_IMG_SIM.PAIR_TYPE == 'Rand':
            candi_imgs = list(self.deconf_dict.keys())
        elif self.cfg.CROSS_IMG_SIM.PAIR_TYPE == 'BInter':
            if len(deconf_d['base_comm_pair_list']) >= 2:
                candi_imgs = deconf_d['base_comm_pair_list']
            elif len(deconf_d['novel_comm_pair_list']) >= 2:
                candi_imgs = deconf_d['novel_comm_pair_list']
            else:
                print(f'{key_name}: {deconf_d}')
                return key_name
        elif self.cfg.CROSS_IMG_SIM.PAIR_TYPE == 'NInter':
            if len(deconf_d['novel_comm_pair_list']) >= 2:
                candi_imgs = deconf_d['novel_comm_pair_list']
            elif len(deconf_d['base_comm_pair_list']) >= 2:
                candi_imgs = deconf_d['base_comm_pair_list']
            else:
                print(f'{key_name}: {deconf_d}')
                return key_name
        elif self.cfg.CROSS_IMG_SIM.PAIR_TYPE.startswith('Deconf'):
            min_prob = float(self.cfg.CROSS_IMG_SIM.PAIR_TYPE[6:])

            if len(deconf_d['deconf_pair_list']) >= 2:
                keys = list(deconf_d['deconf_pair_list'].keys())
                counts = np.array([len(v) for v in deconf_d['deconf_pair_list'].values()])
                counts[counts == 0] = 500

                if np.random.rand() < min_prob:
                    focused_key = keys[counts.argmin()]
                else:
                    focused_key = np.random.choice(keys)

                candi_imgs = deconf_d['deconf_pair_list'][focused_key]

            else:
                if len(deconf_d['novel_comm_pair_list']) >= 2:
                    candi_imgs = deconf_d['novel_comm_pair_list']
                elif len(deconf_d['base_comm_pair_list']) >= 2:
                    candi_imgs = deconf_d['base_comm_pair_list']
                else:
                    # print(f'{key_name}: {deconf_d}')
                    return key_name

        else:
            raise NotImplementedError

        if len(candi_imgs) >= 2:
            if key_name in candi_imgs:
                candi_imgs.remove(key_name)

        if len(list(candi_imgs)) == 0:
            print(f'Key Name: {key_name}')
            return key_name

        pair_img_name = np.random.choice(list(candi_imgs))
        return pair_img_name
