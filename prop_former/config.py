# -*- coding: utf-8 -*-

from detectron2.config import CfgNode as CN

inf = 1e8


def add_prop_former_config(cfg):
    print(f'adding PropFormer cfg')

    cfg.SEED = 6

    cfg.OUTPUT_PREFIX = ''
    cfg.MODEL.OUT_TASK = 'SEG'

    # For Proposal Line:
    cfg.MODEL.MASK_FORMER.MAKE_CLS = True
    cfg.MODEL.MASK_FORMER.CLS_WEIGHT = 1.

    cfg.MODEL.MASK_FORMER.FIXED_MATCHER = False
    cfg.MODEL.MASK_FORMER.FREEZE_QUERY = False
    cfg.MODEL.MASK_FORMER.TRANS_QUERY = 'RAND'  # FCWT256 / WDVT1 / WDVT2

    cfg.MODEL.MASK_FORMER.CLS_LOSS_TYPE = 'SoftmaxBCE'  # SoftmaxBCE / SigmoidBCE / RIB / SMS

    ####################################
    cfg.CROSS_IMG_SIM = CN()
    cfg.CROSS_IMG_SIM.BASE_LOSS = 0.
    cfg.CROSS_IMG_SIM.BASE_DETACH = True
    cfg.CROSS_IMG_SIM.BASE_POINT_NUM = 100
    cfg.CROSS_IMG_SIM.LayerNum = 3
    cfg.CROSS_IMG_SIM.BN = True

    cfg.CROSS_IMG_SIM.PAIR_TYPE = 'Deconf0.01'  # [Rand, BInter, NInter, Deconf]

    cfg.CROSS_IMG_SIM.TEACH_DETACH = True
    cfg.CROSS_IMG_SIM.DISTILL_LOSS = 0.
    cfg.CROSS_IMG_SIM.NOVEL_POINT_NUM = 100
    cfg.CROSS_IMG_SIM.DISTILL_TO = 'NovelScore'  # [NovelScore, FullScore, FullLogit, FullLogitC]
    cfg.CROSS_IMG_SIM.DISTILL_FUNC = 'ce'  # [ce, ce, b0.5]
    cfg.CROSS_IMG_SIM.FOCUS_K = 0.
    cfg.CROSS_IMG_SIM.DISTILL_VALID = False

    ############################
    cfg.ALL_EXISTING = True
    cfg.NOVEL_HAS_MASK = False
    ####################################
    cfg.ASM = CN()
    cfg.ASM.HasMaskCls = 5.
    cfg.ASM.NoMaskCls = 5.
    cfg.ASM.HasMaskMask = 1.
    cfg.ASM.NoMaskMask = 0.

    ####################################
    cfg.LOSS = CN()
    cfg.LOSS.AssignCls = 5.
    cfg.LOSS.MILCls = 0.

    cfg.LOSS.AssignMaskDICE = 1.
    cfg.LOSS.AssignMaskMASK = 20.
    cfg.LOSS.CompSupNovel = 0.

    cfg.LOSS.CompSupNovelType = 'EQ'  # [EQ, IN]
    cfg.LOSS.IgnoreInit = -2.9444  # Disable by <=-50
    cfg.LOSS.IgnoreLearnable = False

    ####################################
    cfg.EVAL = CN()
    cfg.EVAL.BIAS = ('1_1_1',)
    return
