import os

import torch
from detectron2.data import DatasetCatalog, MetadataCatalog
import prop_former.data.datasets.coco_stuff_10k.meta_files.info as INFO
# from detectron2.data.datasets import load_sem_seg
from detectron2.utils.file_io import PathManager
from detectron2.data import detection_utils as utils
import numpy as np
from tqdm import tqdm


def load_sem_seg(gt_root, image_root, s_name, gt_ext="png", image_ext="jpg"):
    def file2id(folder_path, file_path):
        image_id = os.path.normpath(os.path.relpath(file_path, start=folder_path))
        image_id = os.path.splitext(image_id)[0]
        return image_id

    input_files = sorted(
        (os.path.join(image_root, f) for f in PathManager.ls(image_root) if f.endswith(image_ext)),
        key=lambda file_path: file2id(image_root, file_path),
    )
    gt_files = sorted(
        (os.path.join(gt_root, f) for f in PathManager.ls(gt_root) if f.endswith(gt_ext)),
        key=lambda file_path: file2id(gt_root, file_path),
    )
    assert len(gt_files) > 0, "No annotations found in {}.".format(gt_root)
    if len(input_files) != len(gt_files):
        input_basenames = [os.path.basename(f)[: -len(image_ext)] for f in input_files]
        gt_basenames = [os.path.basename(f)[: -len(gt_ext)] for f in gt_files]
        intersect = list(set(input_basenames) & set(gt_basenames))
        intersect = sorted(intersect)
        input_files = [os.path.join(image_root, f + image_ext) for f in intersect]
        gt_files = [os.path.join(gt_root, f + gt_ext) for f in intersect]

    dataset_dicts = []
    for (img_path, gt_path) in tqdm(zip(input_files, gt_files)):
        if 'COCO_train2014_000000016680' in img_path:
            continue
        if 'COCO_train2014_000000230639' in img_path:
            continue
        if 'COCO_train2014_000000382127' in img_path:
            continue
        if 'COCO_train2014_000000429995' in img_path:
            continue
        if 'COCO_train2014_000000314646' in img_path:
            continue
        record = {}
        record["file_name"] = img_path
        record["sem_seg_file_name"] = gt_path

        dataset_dicts.append(record)

    return consider_updated_images(s_name, dataset_dicts)


def consider_updated_images(s_name, dataset_dicts):
    updated_ratio_dict = {
        'split10': 0.0,
        'split11': 0.1,
        'split12': 0.2,
        'split13': 0.3,
        'split14': 0.4,
        'split15': 0.5,
    }

    existing_ratio = 0.6

    existing_num = int(len(dataset_dicts) * existing_ratio)
    updated_num = int(len(dataset_dicts) * updated_ratio_dict[s_name])

    randn_permute = np.load('prop_former/data/datasets/coco_stuff_10k/meta_files/updated_rand_permute.npy')
    existing_idx = randn_permute[:existing_num].tolist()
    updated_idx = randn_permute[existing_num:(existing_num + updated_num)].tolist()

    updated_existing_data_list = []

    for i, data in enumerate(dataset_dicts):
        if i in existing_idx:
            img_type = 'existing'
        elif i in updated_idx:
            img_type = 'updated'
        else:
            continue

        data['type'] = img_type
        updated_existing_data_list.append(data)
        i

    # existing_N = len([i for i in updated_existing_data_list if i['type'] == 'existing'])
    # updated_N = len([i for i in updated_existing_data_list if i['type'] == 'updated'])

    # torch.save(updated_existing_data_list, f'output/Updated_images_split_COCO_{s_name}.pth')
    return updated_existing_data_list


def load_sem_seg_s10(gt_root='datasets/coco/coco_stuff_10k/annotations_detectron2/train',
                     image_root='datasets/coco/coco_stuff_10k/images_detectron2/train'):
    return load_sem_seg(gt_root, image_root, 'split10')


def load_sem_seg_s11(gt_root='datasets/coco/coco_stuff_10k/annotations_detectron2/train',
                     image_root='datasets/coco/coco_stuff_10k/images_detectron2/train'):
    return load_sem_seg(gt_root, image_root, 'split11')


def load_sem_seg_s12(gt_root='datasets/coco/coco_stuff_10k/annotations_detectron2/train',
                     image_root='datasets/coco/coco_stuff_10k/images_detectron2/train'):
    return load_sem_seg(gt_root, image_root, 'split12')


def load_sem_seg_s13(gt_root='datasets/coco/coco_stuff_10k/annotations_detectron2/train',
                     image_root='datasets/coco/coco_stuff_10k/images_detectron2/train'):
    return load_sem_seg(gt_root, image_root, 'split13')


def load_sem_seg_s14(gt_root='datasets/coco/coco_stuff_10k/annotations_detectron2/train',
                     image_root='datasets/coco/coco_stuff_10k/images_detectron2/train'):
    return load_sem_seg(gt_root, image_root, 'split14')


def load_sem_seg_s15(gt_root='datasets/coco/coco_stuff_10k/annotations_detectron2/train',
                     image_root='datasets/coco/coco_stuff_10k/images_detectron2/train'):
    return load_sem_seg(gt_root, image_root, 'split15')


updated_func_dict = {
    'split10': load_sem_seg_s10,
    'split11': load_sem_seg_s11,
    'split12': load_sem_seg_s12,
    'split13': load_sem_seg_s13,
    'split14': load_sem_seg_s14,
    'split15': load_sem_seg_s15,
}
