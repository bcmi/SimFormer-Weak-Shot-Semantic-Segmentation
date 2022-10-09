import logging
from detectron2.evaluation import SemSegEvaluator
import copy
import numpy as np
from collections import OrderedDict
import PIL.Image as Image
import torch
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager
from .pseudo_labeling import generate_pseudo_label
import os


class WeakShotSemSegEvaluator(SemSegEvaluator):
    def __init__(
            self,
            cfg,
            dataset_name,
            eval_type,
            eval_bias,
            distributed=True,
            output_dir=None,
            *,
            num_classes=None,
            ignore_label=None,
            eval_name='weakshot',
    ):
        self._logger = logging.getLogger(__name__)
        if num_classes is not None:
            self._logger.warn(
                "SemSegEvaluator(num_classes) is deprecated! It should be obtained from metadata."
            )
        if ignore_label is not None:
            self._logger.warn(
                "SemSegEvaluator(ignore_label) is deprecated! It should be obtained from metadata."
            )
        self.cfg = cfg
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")

        self.input_file_to_gt_file = {
            dataset_record["file_name"]: dataset_record["sem_seg_file_name"]
            for dataset_record in DatasetCatalog.get(dataset_name)
        }
        self.eval_type = eval_type

        meta = MetadataCatalog.get(dataset_name)

        self.eval_name = eval_name
        if eval_type == 'full':
            self._did_to_eid = meta.c_did_to_cid
            self._class_names = meta.c_class_names
        else:
            raise NotImplementedError

        self._num_classes = len(self._class_names)
        self.meta = meta
        if num_classes is not None:
            assert self._num_classes == num_classes, f"{self._num_classes} != {num_classes}"
        self._ignore_label = ignore_label if ignore_label is not None else meta.ignore_label

        self.eval_iter_count = 0

        # split_idx = int(meta.name.split('_')[2][5:])
        # if split_idx >= 10:
        #     all_files = DatasetCatalog.get(meta.name)
        #     self.ant_file_to_type = {d['sem_seg_file_name']: d['type'] for d in all_files}
        # else:
        #     self.ant_file_to_type = None
        self.ant_file_to_type = None

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            ant_file = self.input_file_to_gt_file[input["file_name"]]
            with PathManager.open(ant_file, "rb") as f:
                gt_segm_raw = np.array(Image.open(f), dtype=np.int)
            gt_segm = self.change_raw_ant_from_did_to_eid(gt_segm_raw)
            gt_segm[gt_segm == self._ignore_label] = self._num_classes

            if self.cfg.GeneratePseudoLabel:
                pred_segm = output["sem_seg"].argmax(dim=0).to(self._cpu_device)
                pred_segm_np = np.array(pred_segm, dtype=np.int)
                mixed_mask_raw = generate_pseudo_label(pred_segm_np, gt_segm_raw, ant_file,
                                                       self._output_dir, self.meta, self.ant_file_to_type)
                mixed_segm = self.change_raw_ant_from_did_to_eid(mixed_mask_raw)
                mixed_segm[mixed_segm == self._ignore_label] = self._num_classes
                pred_segm_np = mixed_segm

            elif self.cfg.EvalPseudoLabel:
                from prop_former.shared import crf_inference_for_segm
                pseudo_path = f'datasets/{self.cfg.PSEUDO_LABEL_PATH}/{os.path.basename(ant_file)}'
                if not os.path.exists(pseudo_path):
                    print(f'Not Found: {pseudo_path}')
                    pseudo_path = ant_file
                with PathManager.open(pseudo_path, "rb") as f:
                    pseudo_mask_raw = np.array(Image.open(f), dtype=np.int)

                dense_crf = False
                if dense_crf:
                    raw_img = np.asarray(Image.open(input['file_name']).convert("RGB"))

                    im_scale_factor = 100
                    pseudo_mask_raw_crfed = crf_inference_for_segm(raw_img, pseudo_mask_raw,
                                                                   im_scale_factor=im_scale_factor)

                    pseudo_mask_raw = pseudo_mask_raw_crfed

                pseudo_segm = self.change_raw_ant_from_did_to_eid(pseudo_mask_raw)
                pseudo_segm[pseudo_segm == self._ignore_label] = self._num_classes
                pred_segm_np = pseudo_segm
            else:
                pred_segm = output["sem_seg"].argmax(dim=0).to(self._cpu_device)
                pred_segm_np = np.array(pred_segm, dtype=np.int)

            # pred_segm_np = gt_segm

            self._conf_matrix += np.bincount(
                (self._num_classes + 1) * pred_segm_np.reshape(-1) + gt_segm.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)

            self.eval_iter_count += 1

    def change_raw_ant_from_did_to_eid(self, ant_did):
        res = copy.deepcopy(ant_did)
        for did in np.unique(ant_did):
            if did == self._ignore_label:
                continue
            elif did not in self._did_to_eid:
                res[ant_did == did] = self._ignore_label
                print(f'{did} not in self._did_to_eid')
            else:
                res[ant_did == did] = self._did_to_eid[did]

        return res

    def evaluate(self):
        if self._distributed:
            synchronize()
            conf_matrix_list = all_gather(self._conf_matrix)
            # self._predictions = all_gather(self._predictions)
            # self._predictions = list(itertools.chain(*self._predictions))
            if not is_main_process():
                return

            self._conf_matrix = np.zeros_like(self._conf_matrix)
            for conf_matrix in conf_matrix_list:
                self._conf_matrix += conf_matrix

        # if self._output_dir:
        #     PathManager.mkdirs(self._output_dir)
        #     file_path = os.path.join(self._output_dir, "sem_seg_predictions.json")
        #     with PathManager.open(file_path, "w") as f:
        #         f.write(json.dumps(self._predictions))

        acc = np.full(self._num_classes, np.nan, dtype=np.float)
        iou = np.full(self._num_classes, np.nan, dtype=np.float)
        tp = self._conf_matrix.diagonal()[:-1].astype(np.float)
        pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(np.float)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(np.float)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        iou_valid = (pos_gt + pos_pred) > 0
        union = pos_gt + pos_pred - tp
        iou[acc_valid] = tp[acc_valid] / union[acc_valid]
        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
        miou = np.sum(iou[acc_valid]) / np.sum(iou_valid)
        fiou = np.sum(iou[acc_valid] * class_weights[acc_valid])
        pacc = np.sum(tp) / np.sum(pos_gt)

        res = {}
        res["mIoU"] = 100 * miou
        res["fwIoU"] = 100 * fiou
        for i, name in enumerate(self._class_names):
            res["IoU-{}".format(name)] = 100 * iou[i]
        res["mACC"] = 100 * macc
        res["pACC"] = 100 * pacc
        for i, name in enumerate(self._class_names):
            res["ACC-{}".format(name)] = 100 * acc[i]

        # if self._output_dir:
        #     file_path = os.path.join(self._output_dir, "sem_seg_evaluation.pth")
        #     with PathManager.open(file_path, "wb") as f:
        #         torch.save(res, f)

        res = self.add_metric_for_base_novel(res)

        results = OrderedDict({f"sem_seg {self.eval_name}": res})

        # results_bg = self.evaluate_bg()
        return results

    def add_metric_for_base_novel(self, res):
        base_acc_iou_list = [[], []]
        novel_acc_iou_list = [[], []]

        base_names = [self.meta.c_did_to_name[did] for did in self.meta.c_base_dids]
        novel_names = [self.meta.c_did_to_name[did] for did in self.meta.c_novel_dids]

        outlier = []
        for k, v in res.items():
            if np.isnan(v):
                continue

            metric_name = k.split('-')
            if len(metric_name) >= 2:
                # metric, name = metric_name
                metric = k[:3]
                name = k[4:]

                if metric == 'ACC':
                    tid = 0
                elif metric == 'IoU':
                    tid = 1
                else:
                    raise NotImplementedError

                if name in base_names:
                    base_acc_iou_list[tid].append(v)
                elif name in novel_names:
                    novel_acc_iou_list[tid].append(v)
                else:
                    raise NotImplementedError
            else:
                outlier.append(metric_name)

        res['BaseACC'] = np.mean(base_acc_iou_list[0])
        res['BaseIoU'] = np.mean(base_acc_iou_list[1])
        res['NovelACC'] = np.mean(novel_acc_iou_list[0])
        res['NovelIoU'] = np.mean(novel_acc_iou_list[1])

        print('Outlier')
        print(outlier)
        return res
