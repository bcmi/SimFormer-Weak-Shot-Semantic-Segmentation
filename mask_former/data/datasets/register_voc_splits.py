# import os
# import torch
# from detectron2.data import DatasetCatalog, MetadataCatalog
# from .shared import read_data_list_from_file, write_data_list_to_file, split_data_list_from_file
# import numpy as np
# import pickle
#
# ignored_cid = 255
# ignored_dids = [255]
#
# CAT_LIST = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
#             'bottle', 'bus', 'car', 'cat', 'chair',
#             'cow', 'diningtable', 'dog', 'horse',
#             'motorbike', 'person', 'pottedplant',
#             'sheep', 'sofa', 'train',
#             'tvmonitor']
#
# CAT_COLOR = [
#     [255, 255, 255],
#     [220, 20, 60], [119, 11, 32], [0, 0, 142], [0, 0, 230], [106, 0, 228],
#     [0, 60, 100], [0, 80, 100], [0, 0, 70], [0, 0, 192], [250, 170, 30],
#     [100, 170, 30], [220, 220, 0], [175, 116, 175], [250, 0, 30], [165, 42, 42],
#     [255, 77, 255], [0, 226, 252], [182, 182, 255], [0, 82, 0], [120, 166, 157],
# ]
#
# voc_dataset_id_to_names = {k: v for k, v in enumerate(CAT_LIST)}
# voc_dataset_id_to_color = {k: v for k, v in enumerate(CAT_COLOR)}
#
# voc_dataset_ids = list(voc_dataset_id_to_names.keys())
# dataset_id_to_query_id = {did: i for i, did in enumerate(voc_dataset_ids)}
#
# word2vec = pickle.load(open('init_datasets/voc_meta/word_vectors/word2vec.pkl', "rb")).astype(np.float32)
# fasttext = pickle.load(open('init_datasets/voc_meta/word_vectors/fasttext.pkl', "rb")).astype(np.float32)
# fcweight = torch.load('init_datasets/voc_meta/trans_query.pth', map_location='cpu').numpy()
#
#
# # from mask_former.utils.viz_tools import viz_class_colors
# # viz_class_colors(voc_dataset_id_to_names, voc_dataset_id_to_color)
#
# def _get_voc_full_meta():
#     splited_dataset_ids = voc_dataset_ids
#     assert len(splited_dataset_ids) == 21, len(splited_dataset_ids)
#     splited_names = [voc_dataset_id_to_names[did] for did in splited_dataset_ids]
#     splited_did_to_cid = {k: i for i, k in enumerate(splited_dataset_ids)}
#
#     # from 0 to 20.
#     cid_to_did = {v: k for k, v in splited_did_to_cid.items() if v != ignored_cid}
#
#     splited_contiguous_id_to_color = {v: voc_dataset_id_to_color[k] for k, v in splited_did_to_cid.items()}
#
#     ret = {
#         "c_dataset_id_to_contiguous_id": splited_did_to_cid,
#         "c_cid_to_did": cid_to_did,
#         "c_class_names": splited_names,
#         "c_contiguous_id_to_color": splited_contiguous_id_to_color,
#     }
#     ret["word2vec"] = word2vec
#     ret["fasttext"] = fasttext
#     ret["fcweight"] = fcweight
#     return ret
#
#
# def _get_voc_split1_meta():
#     novel1_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle']
#     base1_names = [name for name in CAT_LIST if name not in novel1_names]
#     assert len(base1_names) + len(novel1_names) == len(CAT_LIST)
#
#     base_dataset_ids = [k for k, v in voc_dataset_id_to_names.items() if v in base1_names]
#     novel_dataset_ids = [k for k, v in voc_dataset_id_to_names.items() if v in novel1_names]
#
#     did_to_cid_full = {k: i for i, k in enumerate(voc_dataset_ids)}
#     contiguous_all_dataset_ids = list(did_to_cid_full.keys())
#
#     did_to_cid_training = {k: v if k in base_dataset_ids else ignored_cid for k, v in did_to_cid_full.items()}
#     did_to_cid_testing = did_to_cid_full
#
#     ret = {
#         "c_dataset_id_to_contiguous_id_training": did_to_cid_training,
#         "c_dataset_id_to_contiguous_id_testing": did_to_cid_testing,
#         "c_base_dataset_ids": base_dataset_ids,
#         "c_novel_dataset_ids": novel_dataset_ids,
#         "c_contiguous_all_dataset_ids": contiguous_all_dataset_ids,
#         "c_dataset_id_to_name": voc_dataset_id_to_names,
#         "c_dataset_id_to_color": voc_dataset_id_to_color,
#     }
#     ret["word2vec"] = word2vec
#     ret["fasttext"] = fasttext
#     ret["fcweight"] = fcweight
#     return ret
#
#
# name_to_file = {
#     'voc_full_trainaug_seg': 'init_datasets/voc_meta/train_aug.txt',
#     'voc_full_val_seg': 'init_datasets/voc_meta/val.txt',
#
#     'voc_split1_trainaug_seg': 'init_datasets/voc_meta/train_aug_base1.txt',
#     'voc_split1_val_seg': 'init_datasets/voc_meta/val.txt',
# }
# name_to_meta = {
#     'voc_full_trainaug_seg': _get_voc_full_meta,
#     'voc_full_val_seg': _get_voc_full_meta,
#
#     'voc_split1_trainaug_seg': _get_voc_split1_meta,
#     'voc_split1_val_seg': _get_voc_split1_meta,
# }
#
#
# def register_voc_splits(root):
#     print(f'Register VOC QTFormer...')
#
#     data_root = os.path.join(root, "VOC2012")
#
#     # Read&Save Base1 Split TXT
#     # base1_meta = _get_voc_base1_meta()
#     #
#     # trainaug_base_list, trainaug_novel_list = split_data_list_from_file(
#     #     data_root, name_to_file['voc_full_trainaug_seg'], base1_meta, voc_dataset_id_to_names)
#     #
#     # val_base_list, val_novel_list = split_data_list_from_file(
#     #     data_root, name_to_file['voc_full_val_seg'], base1_meta, voc_dataset_id_to_names)
#     #
#     # write_data_list_to_file(data_root, trainaug_base_list, 'init_datasets/voc_meta/train_aug_base1.txt')
#
#     for split_name in ['voc_full_trainaug_seg', 'voc_full_val_seg',
#                        'voc_split1_trainaug_seg', 'voc_split1_val_seg', ]:
#         split_meta = name_to_meta[split_name]()
#
#         DatasetCatalog.register(
#             split_name,
#             lambda x=data_root, y=name_to_file[split_name]:
#             read_data_list_from_file(x, y)
#         )
#
#         MetadataCatalog.get(split_name).set(
#             evaluator_type="weakshot_sem_seg",
#             ignore_label=ignored_cid,
#             **split_meta,
#         )
#
#     return
#
#
# _root = os.getenv("DETECTRON2_DATASETS", "datasets")
# register_voc_splits(_root)
