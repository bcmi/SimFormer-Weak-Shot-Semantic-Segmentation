import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from detectron2.utils.file_io import PathManager
from shutil import copyfile


def generate_pseudo_label(pred_segm, gt_segm_raw, ant_file, output_dir, meta, ant_file_to_type=None):
    '''
        pred_segm is cid, while gt_segm_raw is did.
    '''

    # split_idx = int(meta.name.split('_')[2][5:])
    # if split_idx >= 10:
    #     img_type = ant_file_to_type[ant_file]
    # else:
    #     img_type = 'existing'
    img_type = 'existing'

    assert img_type in ['existing', 'updated']
    mixed_mask = np.ones_like(gt_segm_raw) * 255

    for gt_did in np.unique(gt_segm_raw):
        if gt_did == 255:
            continue
        if gt_did in meta.c_novel_dids:
            novel_cid = meta.c_did_to_cid[gt_did]
            mixed_mask[pred_segm == novel_cid] = gt_did

    if img_type == 'updated':
        for gt_did in np.unique(gt_segm_raw):
            if gt_did == 255:
                continue
            if gt_did in meta.c_base_dids:
                base_cid = meta.c_did_to_cid[gt_did]
                mixed_mask[pred_segm == base_cid] = gt_did
    else:
        for gt_did in np.unique(gt_segm_raw):
            if gt_did == 255:
                continue
            if gt_did in meta.c_base_dids:
                mixed_mask[gt_segm_raw == gt_did] = gt_did

    os.makedirs(output_dir, exist_ok=True)
    save_file = f'{output_dir}/{os.path.basename(ant_file)}'
    mixed_mask = mixed_mask.astype(np.uint8)

    mixed_mask_img = Image.fromarray(mixed_mask)
    mixed_mask_img.save(save_file)

    # with PathManager.open(save_file, "rb") as f:
    #     mixed_mask2 = np.array(Image.open(f), dtype=np.int)
    #
    # assert (mixed_mask2 == mixed_mask).min()

    # copyfile(ant_file, f'{output_dir}/{os.path.basename(ant_file).split(".")[0]}_GT.png')
    return mixed_mask
