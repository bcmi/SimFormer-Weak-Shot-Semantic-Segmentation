import torch
import torch.nn.functional as F


def get_shuffle_idx(B):
    # shuffle_idx = torch.randperm(B)

    seq_idx = torch.range(0, B - 1).long()
    shuffle_idx = torch.range(0, B - 1).long()
    shuffle_idx[::2] = seq_idx[1::2]
    shuffle_idx[1::2] = seq_idx[::2]
    return shuffle_idx


def get_grid_pair_from_AB(X, Y):
    assert X.dim() == 3
    assert Y.dim() == 3
    B, Ka, d = X.size()
    B, Kb, d = Y.size()

    pair = torch.cat([X.unsqueeze(2).expand(-1, -1, Kb, -1),
                      Y.unsqueeze(1).expand(-1, Ka, -1, -1)], dim=-1)
    return pair


def get_regions(pixel_labels, targets, meta):
    ignore_region = pixel_labels == 255

    novel_region_per = []
    for n_did in meta.c_novel_dids:
        novel_region_per.append(pixel_labels == n_did)

    novel_region_float = torch.stack(novel_region_per).sum(0)
    assert novel_region_float.max() <= 1
    novel_region = novel_region_float.bool()

    pad_region = torch.stack([t['pad_region'] for t in targets]).type_as(pixel_labels)
    pad_region = F.interpolate(pad_region[:, None], size=pixel_labels.size()[-2:], mode="nearest").bool()

    base_region = ~ignore_region * ~novel_region

    assert (ignore_region.float() + novel_region.float() + base_region.float()).max() == 1
    assert (ignore_region.float() + novel_region.float() + base_region.float()).min() == 1

    return base_region.float(), pad_region.float(), novel_region.float(), ignore_region.float()


def rand_sample_points_within_the_region(valid_region, point_num, rand_max=0.1):
    B, _, H, W = valid_region.size()

    point_positions = valid_region.new_ones(B, point_num, 2) * -10
    point_scores = valid_region.new_ones(B, point_num, 1) * -10

    # random score for random topk
    score_map = valid_region + torch.rand_like(valid_region) * rand_max

    score_map_f = score_map.reshape(B, H * W)
    point_probs_f, point_indices_f = torch.topk(score_map_f, k=point_num, dim=1)
    point_probs_per = point_probs_f.reshape(B, point_num)
    point_indices = point_indices_f.reshape(B, point_num)

    ws = (point_indices % W).to(torch.float) * 2 / (W - 1) - 1
    hs = (point_indices // W).to(torch.float) * 2 / (H - 1) - 1

    point_positions[:, :, 0] = ws
    point_positions[:, :, 1] = hs

    point_scores[:, :, 0] = point_probs_per

    assert point_positions.min() >= -1
    assert point_positions.max() <= 1

    return point_positions, point_scores


def sample_on_any_map(points, any_map, mode='bilinear'):
    assert points.dim() == 3
    assert any_map.dim() == 4

    B, K, _ = points.size()
    B, C, H, W = any_map.size()

    points_map = points.reshape(B, K, 1, 2)

    sampled_feature_map = F.grid_sample(any_map, points_map, mode=mode, align_corners=True)
    sampled_feature = sampled_feature_map.squeeze(-1).permute(0, 2, 1)

    return sampled_feature

# def get_regions(pixel_labels, meta):
#     ignore_region = pixel_labels == 255
#
#     novel_region_per = []
#     for n_did in meta.c_novel_dids:
#         novel_region_per.append(pixel_labels == n_did)
#
#     novel_region_float = torch.stack(novel_region_per).sum(0)
#     assert novel_region_float.max() <= 1
#     novel_region = novel_region_float.bool()
#
#     base_region = ~ignore_region * ~novel_region
#     return base_region, novel_region, ignore_region
