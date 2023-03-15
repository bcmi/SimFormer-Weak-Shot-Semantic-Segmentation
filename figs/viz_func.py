import matplotlib.pyplot as plt
import numpy as np
import copy
import os
import torch.nn.functional as F
import torch

red = [255, 0, 0]
blue = [0, 0, 255]
black = [0, 0, 0]

ignored_color = black

def viz_props(data_dict,
              proposal_cls, proposal_masks,
              predicted_masks, meta,
              fpath='', dpi=125):
    dict1 = {}
    dict1['image'] = data_dict['image'].permute(1, 2, 0).cpu().numpy()
    dict1['gt'] = get_colored_mask(data_dict['sem_seg'].cpu().numpy(), meta, 'cid')
    dict1['pred'] = get_colored_mask(predicted_masks.argmax(dim=0).cpu().numpy(), meta, 'cid')
    dict1['B red; N blue'] = get_base_novel_colored_mask(data_dict['sem_seg'].cpu().numpy(), meta, 'did')

    prop_probs = proposal_cls[0].softmax(dim=-1)
    cls_probs, cls_ids = prop_probs.max(dim=1)

    valid_is = torch.where(cls_ids != 171)[0]

    if len(valid_is) == 0:
        return

    valid_probs = cls_probs[valid_is]

    sorted_vids = valid_probs.sort(descending=True)[1]

    dict_list = [dict1]

    proposals = {}
    for i, svi in enumerate(sorted_vids[:8]):
        vi = valid_is[svi]
        did = meta.c_cid_to_did[cls_ids[vi].item()]
        sname = 'B' if did in meta.c_base_dids else 'N'
        cls_name = meta.c_did_to_name[did]
        cls_prob = cls_probs[vi].item()
        proposal_mask = proposal_masks[0][vi].sigmoid().cpu().numpy()
        proposals[f'{sname}-{cls_name[:5]}: {cls_prob:.0%}'] = proposal_mask

        if (i + 1) % 4 == 0:
            dict_list.append(proposals)
            proposals = {}

    if len(proposals) != 0:
        dict_list.append(proposals)

    viz_dict_list(dict_list, fpath=fpath, dpi=dpi, font_unit=5)
    return


def get_base_novel_colored_mask(nmask, meta, mask_type='did'):
    colored_mask = np.zeros(list(nmask.shape) + [3])
    for id in np.unique(nmask):
        if mask_type == 'did':
            if id in meta.c_base_dids:
                color = red
            elif id in meta.c_novel_dids:
                color = blue
            else:
                color = ignored_color
        elif mask_type == 'cid':
            raise NotImplementedError
            if id in meta.c_cid_to_did:
                color = meta.c_did_to_color[meta.c_cid_to_did[id]]
            else:
                color = ignored_color
        else:
            raise NotImplementedError

        colored_mask[:, :, 0][nmask == id] = color[0]
        colored_mask[:, :, 1][nmask == id] = color[1]
        colored_mask[:, :, 2][nmask == id] = color[2]

    return colored_mask / 255.


def get_colored_mask(nmask, meta, mask_type):
    colored_mask = np.zeros(list(nmask.shape) + [3])
    for id in np.unique(nmask):
        if mask_type == 'did':
            if id in meta.c_did_to_color:
                color = meta.c_did_to_color[id]
            else:
                color = ignored_color
        elif mask_type == 'cid':
            if id in meta.c_cid_to_did:
                color = meta.c_did_to_color[meta.c_cid_to_did[id]]
            else:
                color = ignored_color
        else:
            raise NotImplementedError

        colored_mask[:, :, 0][nmask == id] = color[0]
        colored_mask[:, :, 1][nmask == id] = color[1]
        colored_mask[:, :, 2][nmask == id] = color[2]

    return colored_mask / 255.


def viz_detailed_preds(batched_inputs, global_sim_results, pixel_sim_results, sem_pred,
                       meta, pos_th=0.5, prefix="save_dir"):
    global_sim_result = global_sim_results[0]

    fpath = f"{prefix}/detailed/{os.path.basename(batched_inputs[0]['file_name'])}"
    fpath = fpath.replace('.jpg', f"_{str(global_sim_result.device).replace('cuda:', 'g')}.png")
    os.makedirs(os.path.dirname(fpath), exist_ok=True)

    global_class_sim = F.softmax(global_sim_result, dim=-1)[:, 1]

    basic_dict = {
        'Image': batched_inputs[0]['image'].permute(1, 2, 0).cpu().numpy() / 255.,
        'Pred': get_colored_mask(sem_pred.argmax(dim=0).cpu(), meta),
        'GT': get_colored_mask(batched_inputs[0]['sem_seg'], meta),
    }

    cid2did = {v: k for k, v in meta.c_dataset_id_to_contiguous_id_testing.items()}

    detailed_dict = {}
    for cid, gc_sim in enumerate(global_class_sim):
        if gc_sim > pos_th:
            name = meta.c_dataset_id_to_name[cid2did[cid]]
            detailed_dict[f'{name}_{gc_sim:.0%}'] = pixel_sim_results[0][cid].sigmoid().cpu().numpy()
        cid

    viz_dict_list([basic_dict, detailed_dict], fpath=fpath, dpi=100)
    return


def viz_all_proposals(batched_inputs, global_sim_results, mask_pred_results, meta,
                      prefix=""):
    fpath = f"{prefix}/viz/{os.path.basename(batched_inputs[0]['file_name'])}"
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    return


def viz_class_colors(did_to_names, did_to_colors, fpath='output/class_colors.jpg'):
    import copy
    dict_list = []
    lsize = 3

    row = {}
    for i, did in enumerate(list(did_to_names)):
        name = did_to_names[did]
        color = did_to_colors[did]

        patch = np.array(color)[np.newaxis, np.newaxis, :] * np.ones([100, 100, 3])
        row[f'{did}: {name}'] = patch / 255.

        if ((i + 1) % lsize == 0) | (i == len(did_to_names) - 1):
            dict_list.append(copy.deepcopy(row))
            row = {}
        i
    viz_dict_list(dict_list, fpath)
    return


def viz_dict_list(mask_dict_list, fpath, dpi=40, axoff=False, tight=True, font_unit=7):
    if len(mask_dict_list) == 1:
        mask_dict_list.append(mask_dict_list[0])

    size_unit = 5
    dict_num = len(mask_dict_list)
    mask_num = max(len(t) for t in mask_dict_list)

    fig, axes = plt.subplots(ncols=mask_num, nrows=dict_num,
                             figsize=(mask_num * size_unit, dict_num * size_unit))
    # fig.patch.set_facecolor('grey')

    for row in range(dict_num):
        for col in range(mask_num):
            if axoff:
                axes[row, col].axis('off')
            axes[row, col].imshow(np.ones(list(mask_dict_list[0].values())[0].shape[:2]) * 0.5,
                                  'gray', vmax=1., vmin=0.)

    for row, mask_dict in enumerate(mask_dict_list):
        for col, kv in enumerate(mask_dict.items()):
            axes[row, col].set_title(kv[0], fontsize=size_unit * font_unit)
            img = kv[1]

            if img is None:
                continue

            if len(img.shape) == 2:
                axes[row, col].imshow(img, 'gray', vmax=1., vmin=0.)
            elif len(img.shape) == 3:
                axes[row, col].imshow(img)
            else:
                raise NotImplementedError
    if tight:
        plt.tight_layout()

    plt.savefig(fpath.replace('jpg', 'png'), dpi=dpi)
    plt.close()
    return
