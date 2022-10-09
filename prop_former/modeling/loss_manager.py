import torch
import torch.nn.functional as F

from .loss_func import my_softmax_bce
from .loss_func import dice_loss, sigmoid_focal_loss, bce_loss_without_reduction, dice_loss_without_reduction


def get_cls_loss_on_assigned(pred_logits, targets, indices, idx):
    '''
    Args:
        pred_logits: [:,N,K]
        labels_full: [:,N_b]
        indices: [:,N_b/N_b]

    For each sample in the mini-batch:
        There is list_S, list_T in indices, which shows the s-th proposal is assigned to the t-th target.
    '''
    target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
    target_classes = torch.full(
        pred_logits.shape[:2], pred_logits.size(-1) - 1, dtype=torch.int64, device=pred_logits.device
    )
    target_classes[idx] = target_classes_o
    loss_ce = F.cross_entropy(pred_logits.transpose(1, 2), target_classes, reduction='none')

    return loss_ce.mean()


def get_cls_loss_on_pooling(pred_logits, labels_full, ltype='SoftmaxBCE'):
    # Note that there is no ignore class in labels_full.
    if ltype == 'MSM':
        raise NotImplementedError
    elif ltype == 'SigmoidBCE':
        raise NotImplementedError
    elif ltype == 'SoftmaxBCE':
        loss_cls = my_softmax_bce(pred_logits, labels_full).mean()
    elif ltype == 'RIB':
        pass
    else:
        raise NotImplementedError

    '''
        pooled_logits = outputs["pred_logits"].max(1)[0]

        mbce_targets = torch.ones_like(pooled_logits) * -1

        # Note that there is no ignore class in target['labels_full'].
        labels_full = [target['labels_full'] for target in targets]

        for i, target in enumerate(targets):
            mbce_t = target['labels_full']
            mbce_targets[i][:len(mbce_t)] = mbce_t

        if self.cls_loss_type == 'MSM':
            loss_cls = F.multilabel_soft_margin_loss(pooled_logits, mbce_targets)
        elif self.cls_loss_type == 'SigmoidBCE':
            loss_cls = my_sigmoid_bce(pooled_logits, labels_full, mbce_targets).mean()
        elif self.cls_loss_type == 'SoftmaxBCE':
            loss_cls = my_softmax_bce(outputs["pred_logits"], labels_full).mean()
        elif self.cls_loss_type == 'RIB':
            pass
        else:
            raise NotImplementedError
    '''
    return loss_cls


def get_mask_loss_on_assigned(inputs, targets, num_masks):
    if inputs.size(0) == 0:
        return inputs.new_zeros(1)[0], inputs.new_zeros(1)[0]
    else:
        # CHENS CHECK
        # assert (targets.max(1)[0]).min() == 1, f'Should not use zero mask as GT'
        return sigmoid_focal_loss(inputs, targets, num_masks), dice_loss(inputs, targets, num_masks)


def get_mask_loss_on_pooling(inputs, targets, num_masks):
    if inputs.size(0) == 0:
        return inputs.new_zeros(1)[0]
    else:
        # CHENS CHECK
        assert (targets.max(1)[0]).min() == 0
        pooled_pred = inputs.max(1, keepdim=True)[0]
        loss = F.binary_cross_entropy_with_logits(pooled_pred,
                                                  torch.ones_like(pooled_pred),
                                                  reduction="none")
        loss = loss.sum() / inputs.size(0)
        return loss


def activate_top_R_loss(inputs, targets, rate=0.1):
    if inputs.size(0) == 0:
        return inputs.new_zeros(1)[0]
    else:
        # CHENS CHECK
        assert (targets.max(1)[0]).min() == 0

        if inputs.size(0) >= 2:
            d = 1

        topk_region = torch.topk(inputs, k=int(inputs.size(1) * rate), dim=1)[0]
        loss = F.binary_cross_entropy_with_logits(topk_region,
                                                  torch.ones_like(topk_region),
                                                  reduction="none")
        return loss.mean()