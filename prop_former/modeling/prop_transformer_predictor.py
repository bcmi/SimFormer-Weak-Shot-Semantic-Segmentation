import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d

from mask_former.modeling.transformer.position_encoding import PositionEmbeddingSine
from mask_former.modeling.transformer.transformer import Transformer
from mask_former.modeling.transformer.transformer_predictor import MLP
from detectron2.data import MetadataCatalog


class PropTransformerPredictor(nn.Module):
    @configurable
    def __init__(self, in_channels, mask_classification=True, cfg=None, *, num_classes: int, hidden_dim: int,
                 num_queries: int, nheads: int, dropout: float, dim_feedforward: int, enc_layers: int, dec_layers: int,
                 pre_norm: bool, deep_supervision: bool, mask_dim: int, enforce_input_project: bool, ):
        super().__init__()
        self.mask_classification = mask_classification

        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        transformer = Transformer(
            d_model=hidden_dim,
            dropout=dropout,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            normalize_before=pre_norm,
            return_intermediate_dec=deep_supervision,
        )

        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model

        if cfg.MODEL.MASK_FORMER.TRANS_QUERY == 'RAND':
            self.query_embed = nn.Embedding(num_queries, hidden_dim)
        else:
            if cfg.MODEL.MASK_FORMER.TRANS_QUERY == 'FCWT256':
                transferrable_query = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).fcweight
            elif cfg.MODEL.MASK_FORMER.TRANS_QUERY == 'WDVT1':
                transferrable_query = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).word2vec
            elif cfg.MODEL.MASK_FORMER.TRANS_QUERY == 'WDVT2':
                transferrable_query = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).fasttext
            else:
                raise NotImplementedError

            trans_num, trans_dim = transferrable_query.shape
            self.query_embed = nn.Embedding(num_queries, trans_dim)
            self.query_embed.weight.data = torch.tensor(transferrable_query)
            assert trans_num == num_queries
            if trans_dim != hidden_dim:
                self.query_sqz = nn.Linear(trans_dim, hidden_dim, bias=True)

        if cfg.MODEL.MASK_FORMER.FREEZE_QUERY:
            self.query_embed.weight.requires_grad = False

        if in_channels != hidden_dim or enforce_input_project:
            self.input_proj = Conv2d(in_channels, hidden_dim, kernel_size=1)
            weight_init.c2_xavier_fill(self.input_proj)
        else:
            self.input_proj = nn.Sequential()
        self.aux_loss = deep_supervision

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)

        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        self.cfg = cfg

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification

        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dropout"] = cfg.MODEL.MASK_FORMER.DROPOUT
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD
        ret["enc_layers"] = cfg.MODEL.MASK_FORMER.ENC_LAYERS
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["deep_supervision"] = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["cfg"] = cfg
        return ret

    def forward(self, x, mask_features):
        pos = self.pe_layer(x)

        src = x
        mask = None

        if hasattr(self, 'query_sqz'):
            query = self.query_sqz(self.query_embed.weight)
        else:
            query = self.query_embed.weight

        query_embed, memory = self.transformer(self.input_proj(src), mask, query, pos)

        out = {}

        if self.mask_classification:
            outputs_class = self.class_embed(query_embed)
            out["pred_logits"] = outputs_class[-1]

        if self.aux_loss:
            # [l, bs, queries, embed]
            mask_embed = self.mask_embed(query_embed)
            outputs_seg_masks = torch.einsum("lbqc,bchw->lbqhw", mask_embed, mask_features)
            out["pred_masks"] = outputs_seg_masks[-1]
            out["aux_outputs"] = self._set_aux_loss(outputs_class if self.mask_classification else None,
                                                    outputs_seg_masks)
        else:
            # FIXME h_boxes takes the last one computed, keep this in mind
            # [bs, queries, embed]
            mask_embed = self.mask_embed(query_embed[-1])
            outputs_seg_masks = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
            out["pred_masks"] = outputs_seg_masks

        ####

        if self.cfg.CROSS_IMG_SIM.BASE_LOSS != 0:
            out['pixel_features'] = mask_features
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        if self.mask_classification:
            return [{"pred_logits": a, "pred_masks": b} for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]
