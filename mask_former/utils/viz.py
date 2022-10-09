import matplotlib.pyplot as plt
import numpy as np
from terminaltables import AsciiTable
import copy


def viz_data_ant(img, mask, meta, fpath='output/t.jpg'):
    nimg = img.permute(1, 2, 0).numpy() / 255

    colored_mask = np.ones_like(nimg)
    nmask = mask.numpy()

    for cid in np.unique(nmask):
        color = meta.voc_contiguous_id_to_color[cid]
        colored_mask[:, :, 0][nmask == cid] = color[0]
        colored_mask[:, :, 1][nmask == cid] = color[1]
        colored_mask[:, :, 2][nmask == cid] = color[2]

    size_unit = 5
    font_unit = 7

    fig, axes = plt.subplots(ncols=2, nrows=1,
                             figsize=(2 * size_unit, 1 * size_unit))

    axes[0].imshow(nimg)
    axes[0].axis('off')

    axes[1].imshow(colored_mask / 255.)
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(fpath, dpi=100)
    plt.close()

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


def viz_dict_list(mask_dict_list, fpath, dpi=40):
    size_unit = 5
    font_unit = 7
    dict_num = len(mask_dict_list)
    mask_num = max(len(t) for t in mask_dict_list)

    fig, axes = plt.subplots(ncols=mask_num, nrows=dict_num,
                             figsize=(mask_num * size_unit, dict_num * size_unit))

    for row in range(dict_num):
        for col in range(mask_num):
            axes[row, col].axis('off')

    for row, mask_dict in enumerate(mask_dict_list):
        for col, kv in enumerate(mask_dict.items()):
            axes[row, col].set_title(kv[0], fontsize=size_unit * font_unit)
            img = kv[1]
            if len(img.shape) == 2:
                axes[row, col].imshow(img, 'gray', vmax=1., vmin=0.)
            elif len(img.shape) == 3:
                axes[row, col].imshow(img)
            else:
                raise NotImplementedError

    plt.tight_layout()
    plt.savefig(fpath, dpi=dpi)
    plt.close()
    return


def c_print_csv_format(results, logger):
    col_num = 4

    for task, res in results.items():
        imp_keys = sorted([k for k in res.keys() if "-" not in k])
        summary_res = {k: res[k] for k in res.keys() if k in imp_keys}
        class_IoU_res = {k.split('-')[1]: res[k] for k in res.keys() if k not in imp_keys and 'IoU' in k}
        class_ACC_res = {k.split('-')[1]: res[k] for k in res.keys() if k not in imp_keys and 'ACC' in k}

        names = sorted(list(class_IoU_res.keys()))
        ml = max([len(name) for name in names])

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
