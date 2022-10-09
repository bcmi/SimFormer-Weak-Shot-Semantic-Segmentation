import torch
import torch.nn.functional as F

eps = 1e-5


def my_sigmoid_bce(preds, targets, targets_):
    gts = torch.zeros_like(preds)
    for b, target in enumerate(targets):
        gts[b][target] = 1

    loss = -(gts * torch.log(preds.sigmoid()) + (1 - gts) * torch.log(1 - preds.sigmoid()))
    return loss

    batch_res = []
    for y, t in zip(preds, targets_):
        res = []
        for i in range(len(y)):
            if i in t:
                r = -y[i].sigmoid().log()
            else:
                r = -(1 - y[i].sigmoid()).log()
            res.append(r)

        batch_res.append(torch.stack(res))
    batch_res = torch.stack(batch_res)
    return


def my_softmax_bce(multi_preds, targets):
    '''
        multi_preds:    [B,N,K+1]
        targets:        [B,K+1]
    '''

    preds = torch.softmax(multi_preds, -1).max(1)[0]

    gts = torch.zeros_like(preds)
    for b, target in enumerate(targets):
        gts[b][target] = 1

    loss = -(gts * torch.log(preds) + (1 - gts) * torch.log(1 - preds))
    return loss


def dice_loss(inputs, targets, num_masks):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def sigmoid_focal_loss(inputs, targets, num_masks, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_masks


def dice_loss_without_reduction(inputs, targets):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


def bce_loss_without_reduction(inputs, targets):
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return ce_loss