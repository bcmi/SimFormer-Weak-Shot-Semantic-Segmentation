import os
from prop_former.data.datasets.voc.meta_files.info import *
from tqdm import tqdm
from detectron2.data import detection_utils as utils
import numpy as np
from prop_former.data.datasets.shared import write_data_list_to_file

voc_training_file = name_to_file['voc_full_trainaug_seg']
existing_rate = 0.5
split_name = 'split1'
base_names = eval(f'{split_name}_base_names')
novel_names = eval(f'{split_name}_novel_names')
existing_save_file = name_to_existing_file[split_name]
updated_save_file = name_to_updated_file[split_name]


def split_file(root):
    data_root = os.path.join(root, "VOC2012")

    total_lines = open(voc_training_file).read().splitlines()
    total_num = len(total_lines)

    idx_perm = np.random.permutation([i for i in range(total_num)])

    existing_num = int(total_num * existing_rate)

    existing_idx_list = idx_perm[:existing_num].tolist()
    updated_idx_list = idx_perm[existing_num:].tolist()

    base_dids = [k for k, v in voc_did_to_names.items() if v in base_names]
    novel_dids = [k for k, v in voc_did_to_names.items() if v in novel_names]

    existing_data_list, updated_data_list = [], []

    for idx in tqdm(existing_idx_list):
        data = {}
        img_name, ant_name = total_lines[idx].split(' ')
        abs_img_path = f'{data_root}/{img_name}'
        abs_ant_path = f'{data_root}/{ant_name}'

        assert os.path.exists(abs_img_path), f'FileNotFound: {abs_img_path}'
        assert os.path.exists(abs_ant_path), f'FileNotFound: {abs_ant_path}'

        data['file_name'] = abs_img_path
        data['sem_seg_file_name'] = abs_ant_path

        raw_ant = utils.read_image(abs_ant_path)

        has_base = False
        for did in np.unique(raw_ant):
            if did in base_dids:
                has_base = True

        if has_base:
            existing_data_list.append(data)
        else:
            updated_data_list.append(data)

    for idx in tqdm(updated_idx_list):
        data = {}
        img_name, ant_name = total_lines[idx].split(' ')
        abs_img_path = f'{data_root}/{img_name}'
        abs_ant_path = f'{data_root}/{ant_name}'

        assert os.path.exists(abs_img_path), f'FileNotFound: {abs_img_path}'
        assert os.path.exists(abs_ant_path), f'FileNotFound: {abs_ant_path}'

        data['file_name'] = abs_img_path
        data['sem_seg_file_name'] = abs_ant_path

        updated_data_list.append(data)

    write_data_list_to_file(data_root, existing_data_list, existing_save_file)
    write_data_list_to_file(data_root, updated_data_list, updated_save_file)

    A = open(voc_training_file).read().splitlines()
    B = open(existing_save_file).read().splitlines()
    C = open(updated_save_file).read().splitlines()
    assert sorted(B + C) == sorted(A)
    return


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
split_file(_root)
