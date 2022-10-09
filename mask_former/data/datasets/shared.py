import os
from detectron2.data import detection_utils as utils
import numpy as np
from tqdm import tqdm
import torch
import pickle
import torch.nn.functional as F

def get_embedding(cfg):
    dataset_path = os.path.join(cfg['datadir'], cfg['dataset'])
    if cfg['embedding'] == 'word2vec':
        class_emb = pickle.load(open(dataset_path + '/word_vectors/word2vec.pkl', "rb"))
    elif cfg['embedding'] == 'fasttext':
        class_emb = pickle.load(open(dataset_path + '/word_vectors/fasttext.pkl', "rb"))
    elif cfg['embedding'] == 'fastnvec':
        class_emb = np.concatenate([pickle.load(open(dataset_path + '/word_vectors/fasttext.pkl', "rb")),
                                    pickle.load(open(dataset_path + '/word_vectors/word2vec.pkl', "rb"))], axis=1)
    else:
        print("invalid embedding: {0}".format(cfg['embedding']))

    if not cfg['emb_without_normal']:
        class_emb = F.normalize(torch.tensor(class_emb, dtype=torch.float32), p=2, dim=1)
        print("Class embedding map normalized!")
    else:
        class_emb = torch.tensor(class_emb, dtype=torch.float32)
    return class_emb


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


def split_data_list_from_file(data_root, file_path, split_meta, voc_dataset_id_to_names):
    splited_did_to_cid = split_meta['c_dataset_id_to_contiguous_id']

    base_dids = [k for k, v in splited_did_to_cid.items() if v != 255]
    novel_dids = [k for k in voc_dataset_id_to_names.keys() if k not in base_dids]

    base_list, novel_list = [], []
    for line in tqdm(open(file_path).read().splitlines()):
        data = {}
        img_name, ant_name = line.split(' ')
        abs_img_name = f'{data_root}/{img_name}'
        abs_ant_name = f'{data_root}/{ant_name}'

        assert os.path.exists(abs_img_name), f'FileNotFound: {abs_img_name}'
        assert os.path.exists(abs_ant_name), f'FileNotFound: {abs_ant_name}'

        raw_ant = utils.read_image(abs_ant_name)
        data['file_name'] = abs_img_name
        data['sem_seg_file_name'] = abs_ant_name

        has_novel = False
        for did in np.unique(raw_ant):
            if did in novel_dids:
                has_novel = True

        if has_novel:
            novel_list.append(data)
        else:
            base_list.append(data)

    return base_list, novel_list


def write_data_list_to_file(data_root, data_list, file_path):
    'images_detection2/2011_003276.jpg annotations_detection2/2011_003276.png'

    with open(file_path, 'w', encoding='utf-8') as f:
        for data in data_list:
            line = f"{data['file_name'].split(data_root + '/')[1]}" \
                   f" {data['sem_seg_file_name'].split(data_root + '/')[1]}\n"
            f.write(line)
