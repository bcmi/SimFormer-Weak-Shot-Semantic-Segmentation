from detectron2.data import MetadataCatalog
import torch.nn as nn
import torch
import numpy as np
import copy

import matplotlib.pyplot as plt
import numpy as np
from terminaltables import AsciiTable
import copy
import os
import torch.nn.functional as F


def c_print_csv_format(results, logger):
    col_num = 4

    for task, res in results.items():
        imp_keys = sorted([k for k in res.keys() if "-" not in k])
        summary_res = {k: res[k] for k in res.keys() if k in imp_keys}
        class_IoU_res = {k.split('-')[1]: res[k] for k in res.keys() if k not in imp_keys and 'IoU' in k}
        class_ACC_res = {k.split('-')[1]: res[k] for k in res.keys() if k not in imp_keys and 'ACC' in k}

        names = sorted(list(class_IoU_res.keys()))
        ml = min(max([len(name) for name in names]), 10)

        table_data = []
        title = [f'     Name: IoU / ACC' for i in range(col_num)]
        table_data.append(title)

        row_data = []
        for i, name in enumerate(names):
            row_data.append(f'{name.ljust(ml)}: {class_IoU_res[name]:.1f}/{class_ACC_res[name]:.1f}')
            if ((i + 1) % col_num == 0) | (i == len(names) - 1):
                table_data.append(copy.deepcopy(row_data))
                row_data = []

        table_ins = AsciiTable(table_data)
        for i in range(len(table_ins.justify_columns)):
            table_ins.justify_columns[i] = 'center'
        out_str = f'\n!! Class Result of \"{task}\":\n{table_ins.table}'
        logger.info(out_str)

        name, value = [], []
        for k, v in summary_res.items():
            name.append(f'{k.ljust(5)}')
            value.append(f'{v:.1f}')

        table_ins = AsciiTable([name, value])
        for i in range(len(table_ins.justify_columns)):
            table_ins.justify_columns[i] = 'center'
        out_str = f'\n!! Summary of \"{task}\":\n{table_ins.table}'

        logger.info(out_str)

    return

def print_pc(module_dict, printf=print):
    for name, module in module_dict.items():
        total_params = sum(p.numel() for p in module.parameters())
        total_trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)

        printf(f'{total_trainable_params / 1e6:.1f}M/{total_params / 1e6:.1f}M training/total params in {name}.')
    return


def crf_inference_for_segm(img, segm, t=10, pos_scale_factor=1, im_scale_factor=2):
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax

    score_maps = np.stack([segm == c for c in np.unique(segm)]).astype(np.float32)

    localcid_to_globalcid = {i: c for i, c in enumerate(np.unique(segm))}

    h, w = img.shape[:2]
    n_labels = score_maps.shape[0]

    d = dcrf.DenseCRF2D(w, h, n_labels)
    d.setUnaryEnergy(score_maps.reshape((n_labels, -1)))

    d.addPairwiseGaussian(sxy=3 / pos_scale_factor, compat=3)
    d.addPairwiseBilateral(sxy=80 / im_scale_factor, srgb=13, rgbim=np.copy(img), compat=10)
    Q = d.inference(t)
    res = np.array(Q).reshape((n_labels, h, w)).argmax(0)

    final_res = copy.deepcopy(segm)
    for localcid in np.unique(res):
        final_res[res == localcid] = localcid_to_globalcid[localcid]

    return final_res


def crf_inference_for_prob(img, segm, t=10, scale_factor=1, labels=21):
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax

    h, w = img.shape[:2]
    n_labels = labels

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3 / scale_factor, compat=3)
    d.addPairwiseBilateral(sxy=80 / scale_factor, srgb=13, rgbim=np.copy(img), compat=10)
    Q = d.inference(t)

    return np.array(Q).reshape((n_labels, h, w))
