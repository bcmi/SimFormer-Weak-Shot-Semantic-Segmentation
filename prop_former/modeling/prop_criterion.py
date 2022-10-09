import torch
import torch.nn.functional as F
from torch import nn
from detectron2.data import MetadataCatalog
from detectron2.utils.comm import get_world_size

from mask_former.utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list
from .loss_manager import get_cls_loss_on_pooling, get_cls_loss_on_assigned
from .loss_manager import get_mask_loss_on_pooling, get_mask_loss_on_assigned, activate_top_R_loss

import logging
import numpy as np
from .cross_img_sim.cro_simnet import get_cro_simnet, BalanceBinaryWeightManager
from .cross_img_sim.func import get_regions, rand_sample_points_within_the_region, sample_on_any_map
from .cross_img_sim.func import get_grid_pair_from_AB, get_shuffle_idx

eps = 1e-5
log_eps = 1e-2


class PropSetCriterion(nn.Module):
    def __init__(self, cfg, num_classes, matcher, weight_dict, eos_coef, losses):
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)
        losses_not_in_aux = ['base_sim_novel_reg', 'base_region_seg', 'cross_img_base_sim_novel_distill']
        self.aux_losses = [l for l in losses if l not in losses_not_in_aux]
        self.meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        self.cls_loss_type = cfg.MODEL.MASK_FORMER.CLS_LOSS_TYPE
        self.logger = logging.getLogger('d2.engine.train_loop')

        self.all_count = 0
        self.iter_count = 0

        if cfg.LOSS.IgnoreInit > -50:
            self.ingore_th = nn.Parameter(torch.tensor(cfg.LOSS.IgnoreInit), requires_grad=cfg.LOSS.IgnoreLearnable)
        else:
            self.ingore_th = None

        if cfg.CROSS_IMG_SIM.BASE_LOSS != 0:
            self.cro_pixel_sim_net = get_cro_simnet(cfg, dim_in=256 * 2, dim_mid=256)
            self.cro_weighter = BalanceBinaryWeightManager()

        return

    def loss_cross_img_base_sim_novel_distill(self, outputs, targets, indices, num_masks):
        '''
            pixel_label_map is did map!
        '''
        pixel_feat_map = outputs['pixel_features']

        B, C, H, W = pixel_feat_map.size()
        base_K = self.cfg.CROSS_IMG_SIM.BASE_POINT_NUM
        # Get pixel GT map from segm_gt_raw_aug (did map)!
        pixel_label_map = torch.stack([t['segm_gt_raw_aug'] for t in targets]).type_as(pixel_feat_map)
        pixel_label_map = F.interpolate(pixel_label_map[:, None], size=(H, W), mode="nearest")

        base_region, pad_region, novel_region, ignore_region = get_regions(pixel_label_map, targets, self.meta)

        shuffle_idx = get_shuffle_idx(B)

        assert B == 2, f'only support BS=2 for shuffle yet'
        img1_classes, img2_classes = [t['labels'].tolist() for t in targets]

        img1_base_classes = [c for c in img1_classes if c in self.meta.c_base_dids]
        img2_base_classes = [c for c in img2_classes if c in self.meta.c_base_dids]
        common_base_classes = list(set(img1_base_classes).intersection(set(img2_base_classes)))

        if common_base_classes:
            sampled_common_class = int(np.random.choice(common_base_classes))
            common_base_region = (pixel_label_map == sampled_common_class).float()
        else:
            common_base_region = base_region

        pos_points, _ = rand_sample_points_within_the_region(common_base_region, int(base_K * 0.8))
        # neg_points, _ = rand_sample_points_within_the_region(base_region * (1 - common_base_region), int(base_K * 0.2))
        neg_points, _ = rand_sample_points_within_the_region((1 - pad_region) * (1 - common_base_region), int(base_K * 0.2))
        # neg_points, _ = rand_sample_points_within_the_region(novel_region + base_region * 0.5, int(base_K * 0.2))
        base_points = torch.cat([pos_points, neg_points], dim=1)
        valid_map = base_region

        base_pixel_labels = sample_on_any_map(base_points, pixel_label_map, mode='nearest')
        base_pixel_valids = sample_on_any_map(base_points, valid_map, mode='nearest')
        base_pixel_feats = sample_on_any_map(base_points, pixel_feat_map)

        base_pixel_labels_shuffled = base_pixel_labels[shuffle_idx]
        base_pixel_feats_shuffled = base_pixel_feats[shuffle_idx]
        base_pixel_valids_shuffled = base_pixel_valids[shuffle_idx]

        base_pair_valids = get_grid_pair_from_AB(base_pixel_valids, base_pixel_valids_shuffled)
        base_pair_feats = get_grid_pair_from_AB(base_pixel_feats, base_pixel_feats_shuffled)

        if self.cfg.CROSS_IMG_SIM.BASE_DETACH:
            base_pair_feats = base_pair_feats.detach()

        base_pair_sim_pred = self.cro_pixel_sim_net(base_pair_feats.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        base_pair_sim_GT = (base_pixel_labels.unsqueeze(2) == base_pixel_labels_shuffled.unsqueeze(1)).long()
        base_pair_sim_valids = base_pair_valids.max(dim=3, keepdim=True)[0]

        sim_loss_map = F.binary_cross_entropy(base_pair_sim_pred, base_pair_sim_GT.float(), reduction='none')

        self.cro_weighter.update(base_pair_sim_GT)
        balance_weight = self.cro_weighter.get_balance_weight()
        balance_weight_map = sim_loss_map.new(balance_weight)[base_pair_sim_GT]

        balanced_sim_loss_map = balance_weight_map * sim_loss_map
        sim_loss = (balanced_sim_loss_map * base_pair_sim_valids).sum() / base_pair_sim_valids.sum().clamp(min=eps)

        NK = self.cfg.CROSS_IMG_SIM.NOVEL_POINT_NUM

        img1_classes, img2_classes = [t['labels'].tolist() for t in targets]
        img1_novel_classes = [c for c in img1_classes if c in self.meta.c_novel_dids]
        img2_novel_classes = [c for c in img2_classes if c in self.meta.c_novel_dids]
        common_novel_classes = list(set(img1_novel_classes).intersection(set(img2_novel_classes)))

        if len(common_novel_classes) >= 1:
            distill_region = novel_region + ignore_region # enable for training
            # distill_region = novel_region             # enable for eval
            distill_points, distill_points_scores = rand_sample_points_within_the_region(distill_region, NK)

            distill_pixel_valids = sample_on_any_map(distill_points, distill_region, mode='nearest')
            distill_pixel_feats = sample_on_any_map(distill_points, pixel_feat_map)

            distill_pixel_feats_shuffled = distill_pixel_feats[shuffle_idx]
            distill_pixel_valids_shuffled = distill_pixel_valids[shuffle_idx]

            distill_pair_feats = get_grid_pair_from_AB(distill_pixel_feats, distill_pixel_feats_shuffled)
            distill_pair_valids = get_grid_pair_from_AB(distill_pixel_valids, distill_pixel_valids_shuffled)

            if self.cfg.CROSS_IMG_SIM.TEACH_DETACH:
                with torch.no_grad():
                    distill_pair_feats = distill_pair_feats.detach()
                    distill_pair_sim_tea = self.cro_pixel_sim_net(
                        distill_pair_feats.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
                    distill_pair_sim_tea = distill_pair_sim_tea.squeeze(-1)
            else:
                distill_pair_sim_tea = self.cro_pixel_sim_net(
                    distill_pair_feats.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
                distill_pair_sim_tea = distill_pair_sim_tea.squeeze(-1)

            if self.cfg.CROSS_IMG_SIM.DISTILL_TO.endswith('Score'):
                if self.cfg.CROSS_IMG_SIM.DISTILL_TO.startswith('Full'):
                    full_cls = F.softmax(outputs['pred_logits'], dim=-1)[..., :-1]
                elif self.cfg.CROSS_IMG_SIM.DISTILL_TO.startswith('Novel'):
                    novel_cids = [self.meta.c_did_to_cid[did] for did in self.meta.c_novel_dids]
                    full_cls = F.softmax(outputs['pred_logits'], dim=-1)[..., :-1][:, :, novel_cids]
                else:
                    raise NotImplementedError
                full_proposal = outputs['pred_masks'].sigmoid()
                distill_to_map = torch.einsum("bqc,bqhw->bchw", full_cls, full_proposal)
            elif self.cfg.CROSS_IMG_SIM.DISTILL_TO.startswith('FullLogit'):
                if self.cfg.CROSS_IMG_SIM.DISTILL_TO == 'FullLogitC':
                    full_cls = torch.relu(outputs['pred_logits'][..., :-1])
                else:
                    full_cls = outputs['pred_logits'][..., :-1]
                full_proposal = outputs['pred_masks'].sigmoid()
                distill_to_map = torch.einsum("bqc,bqhw->bchw", full_cls, full_proposal)
            else:
                raise NotImplementedError

            distill_pixel_stu_feats = sample_on_any_map(distill_points, distill_to_map)
            distill_pixel_sim_stu = torch.relu(F.cosine_similarity(
                distill_pixel_stu_feats.unsqueeze(2), distill_pixel_stu_feats.unsqueeze(1), dim=-1))

            if self.cfg.CROSS_IMG_SIM.DISTILL_FUNC.startswith('b'):
                threshold = float(self.cfg.CROSS_IMG_SIM.DISTILL_FUNC[1:])
                distill_loss_map = -(distill_pair_sim_tea - threshold) * distill_pixel_sim_stu
            elif self.cfg.CROSS_IMG_SIM.DISTILL_FUNC == 'ce':
                distill_pair_sim_tea = distill_pair_sim_tea.clamp(min=log_eps, max=1 - log_eps)
                distill_pixel_sim_stu = distill_pixel_sim_stu.clamp(min=log_eps, max=1 - log_eps)
                distill_loss_map = -distill_pair_sim_tea * torch.log(distill_pixel_sim_stu) \
                                   - (1 - distill_pair_sim_tea) * torch.log(1 - distill_pixel_sim_stu)
            elif self.cfg.CROSS_IMG_SIM.DISTILL_FUNC == 'cce':
                threshold = 0.5
                distill_loss_map = \
                    -torch.relu(distill_pair_sim_tea - threshold) * torch.log(distill_pixel_sim_stu + 1) \
                    - torch.relu(threshold - distill_pair_sim_tea) * torch.log(1 - distill_pixel_sim_stu + 1)
            else:
                raise NotImplementedError

            if self.cfg.CROSS_IMG_SIM.DISTILL_VALID:
                valid_map = distill_pair_valids.max(dim=3, keepdim=False)[0]
                distill_loss = (distill_loss_map * valid_map).sum() / valid_map.sum().clamp(min=eps)
            else:
                distill_loss = distill_loss_map.mean()

        else:
            distill_loss = pixel_feat_map.new_zeros(1)[0]

        return {'loss_cro_sim': sim_loss, 'loss_cro_dit': distill_loss}

    def loss_labels_asm(self, outputs, targets, indices, num_masks):
        src_logits = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)

        return {"loss_cls_a": loss_ce}

    def loss_masks_asm(self, outputs, targets, indices, num_masks):
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        target_masks = target_masks[tgt_idx]

        image_H = target_masks.shape[-2:][0]
        src_masks = F.interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                  mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)

        has_mask_idx = target_masks.mean(1) != 0
        no_mask_idx = ~has_mask_idx

        loss_mask, loss_dice = get_mask_loss_on_assigned(
            src_masks[has_mask_idx], target_masks[has_mask_idx], num_masks)

        loss_dict = {"loss_mask_m": loss_mask,
                     "loss_mask_d": loss_dice}

        if (self.cfg.LOSS.CompSupNovel) != 0:
            loss_comp_sup_novel_batch = src_masks.new_zeros(1)[0]

            class_num_batch = [len(v['labels']) for v in targets]
            pred_mask_batch = src_masks.split(class_num_batch)
            gt_mask_batch = target_masks.split(class_num_batch)
            has_mask_idx_batch = has_mask_idx.split(class_num_batch)
            no_mask_idx_batch = no_mask_idx.split(class_num_batch)
            prop_cls_logits_batch = outputs['pred_logits'][src_idx].split(class_num_batch)

            self.novel_proposals_batch = []
            self.novel_proposals_cls_logit_batch = []
            self.comp_item_count = eps
            self.entroy_item_count = eps

            for b in range(len(targets)):
                if no_mask_idx_batch[b].sum() == 0:
                    # All base masks, so no need to add sup.
                    self.novel_proposals_batch.append(None)
                    self.novel_proposals_cls_logit_batch.append(None)
                else:
                    novel_proposals = pred_mask_batch[b][no_mask_idx_batch[b]]
                    self.novel_proposals_batch.append(novel_proposals.reshape(no_mask_idx_batch[b].sum(), image_H, -1))
                    self.novel_proposals_cls_logit_batch.append(prop_cls_logits_batch[b][no_mask_idx_batch[b]])
                    if self.ingore_th is None:
                        notbase_proposals = novel_proposals
                    else:
                        ignore_proposal = self.ingore_th[None, None].expand(-1, novel_proposals.size(1))
                        notbase_proposals = torch.cat((novel_proposals, ignore_proposal))
                        # print(self.ingore_th)

                    if has_mask_idx_batch[b].sum() >= 1:
                        notbase_region_GT = 1 - gt_mask_batch[b][has_mask_idx_batch[b]].max(0)[0]
                    else:
                        notbase_region_GT = src_masks.new_ones(src_masks.size(1))

                    pad_region = targets[b]['pad_region'].flatten(0)
                    notbase_region_GT *= (1 - pad_region.type_as(notbase_region_GT))

                    assert notbase_region_GT.sum() != 0, f'Seem no novel here'

                    if self.cfg.LOSS.CompSupNovel != 0:
                        notbase_region_pred = notbase_proposals.max(0)[0]
                        if self.cfg.LOSS.CompSupNovelType == 'EQ':
                            loss_comp_sup_novel_mask, loss_comp_sup_novel_dice = \
                                get_mask_loss_on_assigned(notbase_region_pred[None],
                                                          notbase_region_GT[None], num_masks=1)

                            loss_comp_sup_novel_per = self.cfg.LOSS.AssignMaskMASK * loss_comp_sup_novel_mask + \
                                                      self.cfg.LOSS.AssignMaskDICE * loss_comp_sup_novel_dice
                        else:
                            raise NotImplementedError
                        loss_comp_sup_novel_batch += loss_comp_sup_novel_per
                        self.comp_item_count += 1

            if self.cfg.LOSS.CompSupNovel != 0:
                loss_dict['loss_comp'] = loss_comp_sup_novel_batch / self.comp_item_count

        return loss_dict

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def forward(self, outputs, targets):
        device = outputs['pred_masks'].device

        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        indices = self.matcher(outputs_without_aux, targets)

        all_mask_num = sum(len(t["labels"]) for t in targets)
        has_mask_num = torch.cat([t["has_masks"] for t in targets]).sum().item()

        all_mask_num = torch.as_tensor([all_mask_num], dtype=torch.float, device=device)
        has_mask_num = torch.as_tensor([has_mask_num], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(all_mask_num)
            torch.distributed.all_reduce(has_mask_num)
        all_mask_num = torch.clamp(all_mask_num / get_world_size(), min=1).item()
        has_mask_num = torch.clamp(has_mask_num / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, has_mask_num))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.aux_losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, has_mask_num)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        self.iter_count += 1
        return losses

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            "labels_asm": self.loss_labels_asm,
            "masks_asm": self.loss_masks_asm,
            'cross_img_base_sim_novel_distill': self.loss_cross_img_base_sim_novel_distill,
        }

        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)
