import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from prop_former.data.datasets.shared import read_data_list_from_file, read_split_data_list_from_file
from prop_former.data.datasets.voc.meta_files.info import *


# from mask_former.utils.viz_tools import viz_class_colors
# viz_class_colors(voc_dataset_id_to_names, voc_dataset_id_to_color)

def _get_voc_meta():
    did_to_cid = {k: i for i, k in enumerate(voc_did_list)}
    cid_to_did = {v: k for k, v in did_to_cid.items()}
    ret = {
        "c_did_to_cid": did_to_cid,
        "c_cid_to_did": cid_to_did,
        "c_class_names": [voc_did_to_names[did] for did in did_to_cid.keys()],
        "c_did_to_name": voc_did_to_names,
        "stuff_classes": [voc_did_to_names[did] for did in did_to_cid.keys()]
    }
    return ret


def _get_voc_split_meta(split_name):
    # Only used in Training
    base_names = eval(f'{split_name}_base_names')
    novel_names = eval(f'{split_name}_novel_names')
    assert len(base_names) + len(novel_names) == len(CAT_LIST)

    base_dids = [k for k, v in voc_did_to_names.items() if v in base_names]
    novel_dids = [k for k, v in voc_did_to_names.items() if v in novel_names]
    did_to_cid = {k: i for i, k in enumerate(voc_did_list)}
    cid_to_did = {v: k for k, v in did_to_cid.items() if v != ignored_cid}

    ret = {
        "c_did_to_cid": did_to_cid,
        "c_cid_to_did": cid_to_did,
        "c_class_names": [voc_did_to_names[did] for did in did_to_cid.keys()],
        "c_did_to_name": voc_did_to_names,

        "c_base_dids": base_dids,
        "c_novel_dids": novel_dids,
        "stuff_classes": [voc_did_to_names[did] for did in did_to_cid.keys()]
    }
    return ret


def register_voc_splits(root):
    print(f'Register VOC PropFormer...')
    data_root = os.path.join(root, "VOC2012")

    for typical_split_name in ['voc_val_seg', 'voc_trainaug_seg']:
        split_meta = _get_voc_meta()

        DatasetCatalog.register(
            typical_split_name,
            lambda x=data_root, y=name_to_file[typical_split_name]:
            read_data_list_from_file(x, y))

        MetadataCatalog.get(typical_split_name).set(
            evaluator_type="weakshot_sem_seg",
            ignore_label=ignored_cid,
            **split_meta,
        )

    for s_name in ['split1']:
        split_meta = _get_voc_split_meta(s_name)
        train_split_name = f'voc_{s_name}_trainaug'

        DatasetCatalog.register(
            train_split_name,
            lambda x=data_root, y=name_to_existing_file[train_split_name], z=name_to_updated_file[train_split_name]:
            read_split_data_list_from_file(x, y, z)
        )

        MetadataCatalog.get(train_split_name).set(
            evaluator_type="weakshot_sem_seg",
            ignore_label=ignored_cid,
            **split_meta,
        )

        eval_split_name = f'voc_{s_name}_val'
        DatasetCatalog.register(
            eval_split_name,
            lambda x=data_root, y=name_to_file[eval_split_name]:
            read_data_list_from_file(x, y))

        MetadataCatalog.get(eval_split_name).set(
            evaluator_type="weakshot_sem_seg",
            ignore_label=ignored_cid,
            **split_meta,
        )
    return


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_voc_splits(_root)
