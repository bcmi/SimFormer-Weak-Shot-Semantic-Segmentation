import os
from detectron2.data import DatasetCatalog, MetadataCatalog
import prop_former.data.datasets.ADE_20k.info as INFO
from detectron2.utils.file_io import PathManager
from detectron2.data import detection_utils as utils
import numpy as np
from tqdm import tqdm


def load_sem_seg(gt_root, image_root, gt_ext="png", image_ext="jpg"):
    # We match input images with ground truth based on their relative filepaths (without file
    # extensions) starting from 'image_root' and 'gt_root' respectively.
    def file2id(folder_path, file_path):
        # extract relative path starting from `folder_path`
        image_id = os.path.normpath(os.path.relpath(file_path, start=folder_path))
        # remove file extension
        image_id = os.path.splitext(image_id)[0]
        return image_id

    input_files = sorted(
        (os.path.join(image_root, f) for f in PathManager.ls(image_root) if f.endswith(image_ext)),
        key=lambda file_path: file2id(image_root, file_path),
    )
    gt_files = sorted(
        (os.path.join(gt_root, f) for f in PathManager.ls(gt_root) if f.endswith(gt_ext)),
        key=lambda file_path: file2id(gt_root, file_path),
    )

    assert len(gt_files) > 0, "No annotations found in {}.".format(gt_root)

    # Use the intersection, so that val2017_100 annotations can run smoothly with val2017 images
    if len(input_files) != len(gt_files):
        input_basenames = [os.path.basename(f)[: -len(image_ext)] for f in input_files]
        gt_basenames = [os.path.basename(f)[: -len(gt_ext)] for f in gt_files]
        intersect = list(set(input_basenames) & set(gt_basenames))
        # sort, otherwise each worker may obtain a list[dict] in different order
        intersect = sorted(intersect)
        input_files = [os.path.join(image_root, f + image_ext) for f in intersect]
        gt_files = [os.path.join(gt_root, f + gt_ext) for f in intersect]

    dataset_dicts = []

    all_255_list = ['ADE_train_00005149',
                    'ADE_train_00005150',
                    'ADE_train_00005152',
                    'ADE_train_00005333',
                    'ADE_train_00005905',
                    'ADE_train_00006510',
                    'ADE_train_00013298',
                    'ADE_train_00014634',
                    'ADE_train_00014636',
                    'ADE_train_00014884',
                    'ADE_train_00015320',
                    'ADE_train_00015330',
                    'ADE_train_00015928',
                    'ADE_train_00019743',
                    'ADE_train_00019385',
                    'ADE_train_00019873']
    for (img_path, gt_path) in tqdm(zip(input_files, gt_files)):
        if os.path.basename(img_path).split('.')[0] in all_255_list:
            continue
        record = {}
        record["file_name"] = img_path
        record["sem_seg_file_name"] = gt_path
        record["type"] = 'exisitng'

        # raw_segm_gt = utils.read_image(gt_path)
        # if raw_segm_gt.mean() == 255:
        #     print(f'')
        #     print(f'ALL 255 in')
        #     print(f'{gt_path}')
        #     print(f'{np.unique(raw_segm_gt)}')
        #     print(f'')
        #     all_255_list.append(gt_path)
        # else:
        #     dataset_dicts.append(record)
        dataset_dicts.append(record)

    return dataset_dicts


# from mask_former.utils.viz_tools import viz_class_colors
# viz_class_colors(voc_dataset_id_to_names, voc_dataset_id_to_color)

def _get_ADE_20k_split_meta(s_name):
    # Only used in Training
    base_names = eval(f'INFO.{s_name}_base_names')
    novel_names = eval(f'INFO.{s_name}_novel_names')
    assert len(base_names) + len(novel_names) == 150

    base_dids = [k for k, v in INFO.did_to_name.items() if v in base_names]
    novel_dids = [k for k, v in INFO.did_to_name.items() if v in novel_names]
    did_to_cid = {k: i for i, k in enumerate(INFO.did_list)}
    cid_to_did = {v: k for k, v in did_to_cid.items()}

    ret = {
        "c_did_to_cid": did_to_cid,
        "c_cid_to_did": cid_to_did,
        "c_class_names": [INFO.did_to_name[did] for did in did_to_cid.keys()],
        "c_did_to_name": INFO.did_to_name,

        "c_base_dids": base_dids,
        "c_novel_dids": novel_dids,

        "c_did_to_color": INFO.did_to_color,
        "stuff_classes": [INFO.did_to_name[did] for did in did_to_cid.keys()]
    }
    return ret


def register_ADE_20k_splits(root):
    print(f'Register ADE 20K PropFormer...')
    root = os.path.join(root, "ADEChallengeData2016")

    for s_name in ['split1', 'split2', 'split3', 'split4']:
        split_meta = _get_ADE_20k_split_meta(s_name)
        for name, image_dirname, sem_seg_dirname in [
            ("train", "images_detectron2/train", "annotations_detectron2/train"),
            ("val", "images_detectron2/test", "annotations_detectron2/test"),
        ]:
            split_name = f'ADE_{s_name}_{name}'
            image_dir = os.path.join(root, image_dirname)
            gt_dir = os.path.join(root, sem_seg_dirname)
            DatasetCatalog.register(
                split_name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
            )
            MetadataCatalog.get(split_name).set(
                image_root=image_dir,
                sem_seg_root=gt_dir,
                evaluator_type="weakshot_sem_seg",
                ignore_label=INFO.ignored_cid,
                **split_meta,
            )
    return


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_ADE_20k_splits(_root)
