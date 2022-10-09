import os
from detectron2.data import detection_utils as utils
import numpy as np
from tqdm import tqdm
import torch
import pickle
import torch.nn.functional as F


def read_split_data_list_from_file(data_root, existing_file_path, updated_file_path):
    existing_data_list = []
    for line in open(existing_file_path).read().splitlines():
        data = {}
        img_name, ant_name = line.split(' ')
        abs_img_name = f'{data_root}/{img_name}'
        abs_ant_name = f'{data_root}/{ant_name}'

        assert os.path.exists(abs_img_name), f'FileNotFound: {abs_img_name}'
        assert os.path.exists(abs_ant_name), f'FileNotFound: {abs_ant_name}'

        data['file_name'] = abs_img_name
        data['sem_seg_file_name'] = abs_ant_name
        data['type'] = 'existing'
        existing_data_list.append(data)

    updated_data_list = []
    for line in open(updated_file_path).read().splitlines():
        data = {}
        img_name, ant_name = line.split(' ')
        abs_img_name = f'{data_root}/{img_name}'
        abs_ant_name = f'{data_root}/{ant_name}'

        assert os.path.exists(abs_img_name), f'FileNotFound: {abs_img_name}'
        assert os.path.exists(abs_ant_name), f'FileNotFound: {abs_ant_name}'

        data['file_name'] = abs_img_name
        data['sem_seg_file_name'] = abs_ant_name
        data['type'] = 'updated'
        updated_data_list.append(data)

    return existing_data_list + updated_data_list


def read_data_list_from_file(data_root, file_path):
    data_list = []
    for line in open(file_path).read().splitlines():
        data = {}
        img_name, ant_name = line.split(' ')
        abs_img_name = f'{data_root}/{img_name}'
        abs_ant_name = f'{data_root}/{ant_name}'

        assert os.path.exists(abs_img_name), f'FileNotFound: {abs_img_name}'
        assert os.path.exists(abs_ant_name), f'FileNotFound: {abs_ant_name}'

        data['file_name'] = abs_img_name
        data['sem_seg_file_name'] = abs_ant_name
        data_list.append(data)

    return data_list


def write_data_list_to_file(data_root, data_list, file_path):
    'images_detection2/2011_003276.jpg annotations_detection2/2011_003276.png'

    with open(file_path, 'w', encoding='utf-8') as f:
        for data in data_list:
            line = f"{data['file_name'].split(data_root + '/')[1]}" \
                   f" {data['sem_seg_file_name'].split(data_root + '/')[1]}\n"
            f.write(line)
