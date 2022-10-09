from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList

from prop_former.modeling.hungarian_matcher import PropHungarianMatcher
from prop_former.modeling.prop_criterion import PropSetCriterion
import numpy as np
import logging
import os
import copy

eps = 1e-5


@META_ARCH_REGISTRY.register()
class PropFormer(nn.Module):
    @configurable
    def __init__(
            self,
            *,
            backbone: Backbone,
            sem_seg_head: nn.Module,
            criterion: nn.Module,
            num_queries: int,
            panoptic_on: bool,
            object_mask_threshold: float,
            overlap_threshold: float,
            metadata,
            size_divisibility: int,
            sem_seg_postprocess_before_inference: bool,
            pixel_mean: Tuple[float],
            pixel_std: Tuple[float],
            cfg: None,
    ):
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.panoptic_on = panoptic_on
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.cfg = cfg
        self.eval_iter_count = -1
        self.training_iter_count = -1

        # self.confusion_IoU_matrix = np.load('prop_former/data/datasets/coco_stuff_10k/'
        #                                     'meta_files/IoU_matrix_coco_stuff_split1_train.npy')

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        cls_asm_weight = cfg.LOSS.AssignCls
        dice_weight = cfg.LOSS.AssignMaskDICE
        mask_weight = cfg.LOSS.AssignMaskMASK
        comp_sup_novel_weight = cfg.LOSS.CompSupNovel

        cross_img_sim_base_weight = cfg.CROSS_IMG_SIM.BASE_LOSS
        cross_img_sim_distill_weight = cfg.CROSS_IMG_SIM.DISTILL_LOSS

        matcher = PropHungarianMatcher(cfg)

        weight_dict = {
            "loss_cls_a": cls_asm_weight,
            "loss_mask_m": mask_weight,
            "loss_mask_d": dice_weight,
            "loss_comp": comp_sup_novel_weight,
            "loss_cro_sim": cross_img_sim_base_weight,
            "loss_cro_dit": cross_img_sim_distill_weight,
        }
        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["masks_asm"]

        if cls_asm_weight != 0:
            losses.append("labels_asm")
        if (cross_img_sim_base_weight + cross_img_sim_distill_weight) != 0:
            losses.append("cross_img_base_sim_novel_distill")

        criterion = PropSetCriterion(cfg, sem_seg_head.num_classes,
                                     matcher=matcher, weight_dict=weight_dict,
                                     eos_coef=no_object_weight, losses=losses)

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                    or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "cfg": cfg,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs_raw):
        if self.cfg.EvalPseudoLabel:
            return [{'sem_seg': None}]

        if 'paired_dict' in batched_inputs_raw[0]:
            batched_inputs = []
            for b_input in batched_inputs_raw:
                batched_inputs.append(b_input)
                batched_inputs.append(b_input['paired_dict'])
        else:
            batched_inputs = batched_inputs_raw

        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features)

        if self.training:
            self.training_iter_count += 1
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)

                for b, (target, binput) in enumerate(zip(targets, batched_inputs)):
                    # Only used for BaseSimNovelReg
                    target['file_name'] = binput['file_name']
                    target['segm_gt_raw_aug'] = binput['segm_gt_raw_aug']
                    target['type'] = binput['type']
                    target['pad_region'] = binput['pad_region']
                    target['img'] = images.tensor[b]
            else:
                targets = None

            outputs['backbone_feature_dict'] = features
            losses = self.criterion(outputs, targets)

            sample_weight = 1

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= (self.criterion.weight_dict[k] * sample_weight)
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
                    print(k)

            self.eval_iter_count = 0
            return losses
        else:
            self.eval_iter_count += 1


            base_region_preds = [None]
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]

            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            processed_results = []

            for b in range(mask_cls_results.size(0)):
                mask_cls_result = mask_cls_results[b]
                mask_pred_result = mask_pred_results[b]
                input_per_image = batched_inputs[b]
                image_size = images.image_sizes[b]
                base_region_pred = base_region_preds[b]

                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])

                # semantic segmentation inference
                if base_region_pred is None:
                    sem_seg = self.semantic_inference(mask_cls_result, mask_pred_result)
                else:
                    sem_seg = self.semantic_inference_deseenbias(mask_cls_result, mask_pred_result, base_region_pred)

                if not self.sem_seg_postprocess_before_inference:
                    sem_seg_final = sem_seg_postprocess(sem_seg, image_size, height, width)
                else:
                    sem_seg_final = sem_seg

                p_res = {"sem_seg": sem_seg_final}

                if base_region_pred is not None:
                    if not self.sem_seg_postprocess_before_inference:
                        base_region_final = sem_seg_postprocess(base_region_pred, image_size, height, width)
                    else:
                        base_region_final = base_region_pred

                    p_res['base_region'] = base_region_final

                processed_results.append(p_res)

                if self.eval_iter_count == 1:
                    logger = logging.getLogger('detectron2')
                    for line in os.popen('nvidia-smi').readlines():
                        logger.info(line)

            return processed_results

    def prepare_targets(self, targets, images):
        h, w = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks

            mask_h, mask_w = gt_masks.shape[-2:]

            assert h == mask_h, f'ImgH {h} v.s. MaskH {mask_h}'
            assert w == mask_w, f'ImgW {w} v.s. MaskW {mask_w}'

            padded_masks = torch.zeros((gt_masks.shape[0], h, w), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, :gt_masks.shape[1], :gt_masks.shape[2]] = gt_masks

            new_targets.append({"labels": targets_per_image.gt_classes,
                                "masks": padded_masks,
                                "has_masks": targets_per_image.has_masks})

        return new_targets

    def semantic_inference(self, in_mask_cls, in_mask_pred):
        query_num = in_mask_pred.size(0)

        if in_mask_cls is None:
            mask_cls = torch.diag_embed(in_mask_pred.new_ones(query_num))
            mask_pred = in_mask_pred.sigmoid()
            semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        else:
            mask_cls = F.softmax(in_mask_cls, dim=-1)[..., :-1]
            mask_pred = in_mask_pred.sigmoid()

            semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)

        return semseg