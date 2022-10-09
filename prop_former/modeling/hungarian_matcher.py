import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn


def batch_mask_loss_novel(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    assert targets.sum() == 0

    T = inputs.new_ones(targets.size(0))[None, :]

    return -torch.log(inputs) * T


def batch_dice_loss(inputs, targets):
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


def batch_sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    hw = inputs.shape[1]

    prob = inputs.sigmoid()
    focal_pos = ((1 - prob) ** gamma) * F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    focal_neg = (prob ** gamma) * F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )
    if alpha >= 0:
        focal_pos = focal_pos * alpha
        focal_neg = focal_neg * (1 - alpha)
    loss = torch.einsum("nc,mc->nm", focal_pos, targets) + torch.einsum(
        "nc,mc->nm", focal_neg, (1 - targets)
    )
    return loss / hw


class PropHungarianMatcher(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    @torch.no_grad()
    def my_assignment(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        indices = []
        for b in range(bs):
            out_prob = outputs["pred_logits"][b].softmax(-1)  # [num_queries, num_classes]
            out_mask = outputs["pred_masks"][b]  # [num_queries, H_pred, W_pred]

            tgt_ids = targets[b]["labels"]
            tgt_mask = targets[b]["masks"].to(out_mask)
            tgt_mask = F.interpolate(tgt_mask[:, None], size=out_mask.shape[-2:], mode="nearest")

            # assert ((tgt_mask.mean([1, 2, 3]) != 0) == targets[b]['has_masks']).min(), \
            #     f"{targets[b]['file_name']}" \
            #     f"{(tgt_mask.mean([1, 2, 3]) != 0), targets[b]['has_masks']}"
            #
            # hasmask_idx = targets[b]['has_masks']
            hasmask_idx = tgt_mask.mean([1, 2, 3]) != 0
            nomask_idx = ~hasmask_idx

            out_mask_f = out_mask.flatten(1)  # [num_queries, H*W]
            tgt_mask_f = tgt_mask[:, 0].flatten(1)  # [num_total_targets, H*W]

            hasmask_cls_cost = -out_prob[:, tgt_ids[hasmask_idx]]
            nomask_cls_cost = -out_prob[:, tgt_ids[nomask_idx]]

            hasmask_mask_cost_mask = batch_sigmoid_focal_loss(out_mask_f, tgt_mask_f[hasmask_idx])
            hasmask_mask_cost_dice = batch_dice_loss(out_mask_f, tgt_mask_f[hasmask_idx])

            hasmask_mask_cost = self.cfg.LOSS.AssignMaskMASK * hasmask_mask_cost_mask \
                                + self.cfg.LOSS.AssignMaskDICE * hasmask_mask_cost_dice

            pMask = F.adaptive_max_pool2d(out_mask[:, None].sigmoid(), 1).squeeze(2).squeeze(2)
            nomask_mask_cost = 0 * batch_mask_loss_novel(pMask, tgt_mask[nomask_idx])

            # Final cost matrix
            A = self.cfg.ASM.HasMaskCls * hasmask_cls_cost
            B = self.cfg.ASM.HasMaskMask * hasmask_mask_cost

            C = self.cfg.ASM.NoMaskCls * nomask_cls_cost
            D = self.cfg.ASM.NoMaskMask * nomask_mask_cost

            cost_matrix = torch.cat((A + B, C + D), dim=1)

            indices.append(linear_sum_assignment(cost_matrix.cpu()))

        return [(torch.as_tensor(i, dtype=torch.int64),
                 torch.as_tensor(j, dtype=torch.int64))
                for i, j in indices]

    @torch.no_grad()
    def forward(self, outputs, targets):
        return self.my_assignment(outputs, targets)

    def __repr__(self):
        head = "Matcher " + self.__class__.__name__
        return head
