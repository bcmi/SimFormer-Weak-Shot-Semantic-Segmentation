import sys

sys.path.append('.')

import torch
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
import prop_former.data.datasets.coco_stuff_10k.meta_files.info as INFO
# from detectron2.data.datasets import load_sem_seg
from detectron2.utils.file_io import PathManager
from detectron2.data import detection_utils as utils
import numpy as np
from tqdm import tqdm
from detectron2.data import DatasetCatalog, MetadataCatalog
import prop_former.data.datasets.coco_stuff_10k.meta_files.info as INFO

eps = 1e-5


def get_imname_to_dids(split_name):
    meta = MetadataCatalog.get(split_name)
    data_list = DatasetCatalog.get(split_name)

    itd_path = f'datasets/imname_to_dids_{split_name}.pth'
    if os.path.exists(itd_path):
        imname_to_dids = torch.load(itd_path)
    else:
        imname_to_dids = {}
        for data_idx, data_item in tqdm(enumerate(data_list)):
            imname = os.path.basename(data_item['sem_seg_file_name'])
            raw_segm_gt = utils.read_image(data_item['sem_seg_file_name'])

            all_dids = np.unique(raw_segm_gt)
            novel_dids = [did for did in all_dids if did in meta.c_novel_dids]
            base_dids = [did for did in all_dids if did in meta.c_base_dids]
            imname_to_dids[imname] = {'base_dids': base_dids, 'novel_dids': novel_dids}

        torch.save(imname_to_dids, itd_path)

    return imname_to_dids


def limit_set_len(anyset, maxlen):
    if len(anyset) <= maxlen:
        return anyset
    else:
        thatlist = list(anyset)
        thatlist.__delitem__(np.random.randint(maxlen))
        return set(thatlist)


def get_deconf_dict(split_name, imname_to_dids):
    deconf_path = f'datasets/imname_to_pair_list_dict_{split_name}.pth'

    if os.path.exists(deconf_path):
        imname_to_pair_list_dict = torch.load(deconf_path)
    else:
        max_deconf_pair_len = 500
        max_common_pair_len = 50
        imname_to_pair_list_dict = {}
        for focused_imname, v in tqdm(imname_to_dids.items()):
            base_dids = v['base_dids']
            novel_dids = v['novel_dids']

            deconf_pair_list = {ndid: set() for ndid in novel_dids}
            novel_comm_pair_list = set()
            base_comm_pair_list = set()

            for candi_imname, candi_v in imname_to_dids.items():
                candi_novel_dids = candi_v['novel_dids']
                candi_base_dids = candi_v['base_dids']

                novel_inter = list(set(novel_dids).intersection(set(candi_novel_dids)))

                if len(novel_inter) == 1:
                    deconf_pair_list[novel_inter[0]].add(candi_imname)
                    deconf_pair_list[novel_inter[0]] = limit_set_len(deconf_pair_list[novel_inter[0]],
                                                                     max_deconf_pair_len)


                elif len(novel_dids) >= 2:
                    novel_comm_pair_list.add(candi_imname)
                    novel_comm_pair_list = limit_set_len(novel_comm_pair_list, max_common_pair_len)

                base_inter = list(set(base_dids).intersection(set(candi_base_dids)))

                if len(base_inter) >= 1:
                    base_comm_pair_list.add(candi_imname)
                    base_comm_pair_list = limit_set_len(base_comm_pair_list, max_common_pair_len)

            imname_to_pair_list_dict[focused_imname] = {'deconf_pair_list': deconf_pair_list,
                                                        'novel_comm_pair_list': novel_comm_pair_list,
                                                        'base_comm_pair_list': base_comm_pair_list}

        torch.save(imname_to_pair_list_dict, deconf_path)

    return imname_to_pair_list_dict


def check_deconf_dict(imname_to_dids, decon_dict):
    for imname, deconf in tqdm(decon_dict.items()):

        for cid, dlist in deconf['deconf_pair_list'].items():
            A = imname_to_dids[imname]['novel_dids']

            for pairname in dlist:
                B = imname_to_dids[pairname]['novel_dids']

                assert set(A).intersection(set(B)) == {cid}, f'{A}; {B}; {imname}; {pairname}'

            dlist

        imname

    return


def main(split_name='ADE_split1_train'):
    meta = MetadataCatalog.get(split_name)
    data_list = DatasetCatalog.get(split_name)

    imname_to_dids = get_imname_to_dids(split_name)
    decon_dict = get_deconf_dict(split_name, imname_to_dids)
    check_deconf_dict(imname_to_dids, decon_dict)
    return


import sys

'coco_stuff_split3_train'
'ADE_split1_train'

# python prop_former/modeling/cross_img_sim/compute_pairs.py coco_stuff_split3_train
if __name__ == '__main__':
    istr = sys.argv[1]
    print(istr)
    main(split_name=istr)
