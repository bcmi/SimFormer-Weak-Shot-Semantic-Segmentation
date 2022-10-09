USE_BACKGROUND = True
# USE_BACKGROUND = False # Change NUM_CLASSES to 20!!! Change MIL Mask Loss to 1e-7 !!!

ignored_cid = 255

name_to_file = {
    'voc_trainaug_seg': 'prop_former/data/datasets/voc/meta_files/train_aug.txt',
    'voc_val_seg': 'prop_former/data/datasets/voc/meta_files/val.txt'
}

name_to_existing_file = {}
name_to_updated_file = {}
for i in range(1):
    name_to_existing_file[
        f'voc_split{i + 1}_trainaug'] = f'prop_former/data/datasets/voc/meta_files/split{i + 1}_existing.txt'
    name_to_updated_file[
        f'voc_split{i + 1}_trainaug'] = f'prop_former/data/datasets/voc/meta_files/split{i + 1}_updated.txt'

    name_to_file[f'voc_split{i + 1}_val'] = 'prop_former/data/datasets/voc/meta_files/val.txt'

if USE_BACKGROUND:
    CAT_LIST = ['background',
                'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                'bus', 'car', 'cat', 'chair', 'cow',
                'diningtable', 'dog', 'horse', 'motorbike', 'person',
                'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    CAT_COLOR = [
        [255, 255, 255],
        [220, 20, 60], [119, 11, 32], [0, 0, 142], [0, 0, 230], [106, 0, 228],
        [0, 60, 100], [0, 80, 100], [0, 0, 70], [0, 0, 192], [250, 170, 30],
        [100, 170, 30], [220, 220, 0], [175, 116, 175], [250, 0, 30], [165, 42, 42],
        [255, 77, 255], [0, 226, 252], [182, 182, 255], [0, 82, 0], [120, 166, 157]]

    voc_did_to_names = {k: v for k, v in enumerate(CAT_LIST)}
    voc_did_to_color = {k: v for k, v in enumerate(CAT_COLOR)}
else:
    CAT_LIST = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                'bus', 'car', 'cat', 'chair', 'cow',
                'diningtable', 'dog', 'horse', 'motorbike', 'person',
                'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    CAT_COLOR = [
        [220, 20, 60], [119, 11, 32], [0, 0, 142], [0, 0, 230], [106, 0, 228],
        [0, 60, 100], [0, 80, 100], [0, 0, 70], [0, 0, 192], [250, 170, 30],
        [100, 170, 30], [220, 220, 0], [175, 116, 175], [250, 0, 30], [165, 42, 42],
        [255, 77, 255], [0, 226, 252], [182, 182, 255], [0, 82, 0], [120, 166, 157]]

    voc_did_to_names = {k + 1: v for k, v in enumerate(CAT_LIST)}
    voc_did_to_color = {k + 1: v for k, v in enumerate(CAT_COLOR)}

voc_did_list = list(voc_did_to_names.keys())

# SPLIT 1
split1_novel_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle']
split1_base_names = [name for name in CAT_LIST if name not in split1_novel_names]

# SPLIT 2
split2_novel_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle']
split2_base_names = [name for name in CAT_LIST if name not in split1_novel_names]

voc_did_to_color_ex = {k: v for k, v in voc_did_to_color.items()}
voc_did_to_color_ex[0] = [255, 255, 255]
voc_did_to_color_ex[255] = [0, 0, 0]
