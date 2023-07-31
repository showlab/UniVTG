import pdb
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from model.transformer_encoder_droppath import build_transformer
from model.matcher import build_matcher
from model.position_encoding import build_position_encoding
from utils.span_utils import generalized_temporal_iou, span_cxw_to_xx

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

def mask_logits(inputs, mask, mask_value=-1e30):
    mask = mask.type(torch.float32)
    return inputs + (1.0 - mask) * mask_value

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

class WeightedPool(nn.Module):
    def __init__(self, dim):
        super(WeightedPool, self).__init__()
        weight = torch.empty(dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def forward(self, x, mask):
        alpha = torch.tensordot(x, self.weight, dims=1)  # shape = (batch_size, seq_length, 1)
        alpha = mask_logits(alpha, mask=mask.unsqueeze(2))
        alphas = nn.Softmax(dim=1)(alpha)
        pooled_x = torch.matmul(x.transpose(1, 2), alphas)  # (batch_size, dim, 1)
        pooled_x = pooled_x.squeeze(2)
        return pooled_x

class Model(nn.Module):
    """ This is the UniVTG module that performs moment localization. """

    def __init__(self, transformer, position_embed, txt_position_embed, txt_dim, vid_dim,
                 input_dropout, aux_loss=False,
                 max_v_l=75, span_loss_type="l1", use_txt_pos=False, n_input_proj=2):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture. See transformer.py
            position_embed: torch module of the position_embedding, See position_encoding.py
            txt_position_embed: position_embedding for text
            txt_dim: int, text query input dimension
            vid_dim: int, video feature input dimension
            max_v_l: int, maximum #clips in videos
            span_loss_type: str, one of [l1, ce]
                l1: (center-x, width) regression.
                ce: (st_idx, ed_idx) classification.
            # foreground_thd: float, intersection over prediction >= foreground_thd: labeled as foreground
            # background_thd: float, intersection over prediction <= background_thd: labeled background
        """
        super().__init__()
        self.transformer = transformer
        self.position_embed = position_embed
        self.txt_position_embed = txt_position_embed
        hidden_dim = transformer.d_model
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        span_pred_dim = 2 if span_loss_type == "l1" else max_v_l * 2

        self.token_type_embeddings = nn.Embedding(2, hidden_dim)
        self.token_type_embeddings.apply(init_weights)

        # Conv projector
        self.span_embed = Conv(hidden_dim, hidden_dim, span_pred_dim, 3, kernel_size=3)
        self.class_embed = Conv(hidden_dim, hidden_dim, 1, 3, kernel_size=3)  # 0: background, 1: foreground

        self.use_txt_pos = use_txt_pos
        self.n_input_proj = n_input_proj
        relu_args = [True] * 3
        relu_args[n_input_proj-1] = False
        self.input_txt_proj = nn.Sequential(*[
            LinearLayer(txt_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])
        self.input_vid_proj = nn.Sequential(*[
            LinearLayer(vid_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])

        # MLP Projector
        self.weightedpool = WeightedPool(hidden_dim)

    def forward(self, src_txt, src_txt_mask, src_vid, src_vid_mask, src_cls=None, src_cls_mask=None):
        bs = src_vid.shape[0]
        src_vid = self.input_vid_proj(src_vid)
        src_txt = self.input_txt_proj(src_txt)
        if src_cls is not None:
            src_cls = self.input_txt_proj(src_cls)

        # type token.
        src_vid = src_vid + self.token_type_embeddings(torch.full_like(src_vid_mask.long(), 1))
        src_txt = src_txt + self.token_type_embeddings(torch.zeros_like(src_txt_mask.long()))
        if src_cls is not None:
            src_cls = src_cls + self.token_type_embeddings(torch.zeros_like(src_cls_mask.long()))

        src = torch.cat([src_vid, src_txt], dim=1)  # (bsz, L_vid+L_txt, d)
        mask = torch.cat([src_vid_mask, src_txt_mask], dim=1).bool()  # (bsz, L_vid+L_txt)

        pos_vid = self.position_embed(src_vid, src_vid_mask)  # (bsz, L_vid, d)
        pos_txt = self.txt_position_embed(src_txt) if self.use_txt_pos else torch.zeros_like(src_txt)  # (bsz, L_txt, d)
        pos = torch.cat([pos_vid, pos_txt], dim=1)

        memory = self.transformer(src, ~mask, pos)
        vid_mem = memory[:, :src_vid.shape[1], :]  # (bsz, L_vid, d)

        outputs_class = self.class_embed(vid_mem).sigmoid()  # (#layers, batch_size, #queries, #classes)
        outputs_coord = self.span_embed(vid_mem)  # (#layers, bsz, #queries, 2 or max_v_l * 2)

        if self.span_loss_type == "l1":
            outputs_coord = outputs_coord.sigmoid()
            idx_mask = torch.tensor((-1, 1)).unsqueeze(0).unsqueeze(0).cuda()
            idx_mask = idx_mask.repeat(outputs_coord.shape[0], outputs_coord.shape[1], 1)
            outputs_coord = outputs_coord * idx_mask
        else:
            raise NotImplementedError

        out = {'pred_logits': outputs_class, 'pred_spans': outputs_coord,
               'src_vid_mask': src_vid_mask}

        vid_mem_proj = src_vid

        # word-level -> sentence-level
        txt_mem_proj = self.weightedpool(src_txt, src_txt_mask).unsqueeze(1)
        sim = F.cosine_similarity(vid_mem_proj, txt_mem_proj, dim=-1) + (src_vid_mask + 1e-45).log()

        out["vid_mem_proj"] = vid_mem_proj
        out["txt_mem_proj"] = txt_mem_proj
        if src_cls is not None:
            cls_mem_proj = self.weightedpool(src_cls, src_cls_mask)
            out["cls_mem_proj"] = cls_mem_proj
        out["saliency_scores"] = sim
        return out

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, matcher, weight_dict, eos_coef, losses, temperature, span_loss_type, max_v_l,
                 saliency_margin=1):
        """ Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            temperature: float, temperature for NCE loss
            span_loss_type: str, [l1, ce]
            max_v_l: int,
            saliency_margin: float
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.temperature = temperature
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        self.saliency_margin = saliency_margin
        self.temperature = 0.07

        # foreground and background classification
        self.foreground_label = 0
        self.background_label = 1
        self.eos_coef = eos_coef
        empty_weight = torch.ones(2)
        empty_weight[-1] = self.eos_coef  # lower weight for background (index 1, foreground index 0)
        self.register_buffer('empty_weight', empty_weight)

    def loss_spans(self, outputs, targets, indices):
        assert 'pred_spans' in outputs

        start_spans = targets['timestamp']
        pred_spans = outputs['pred_spans']
        src_spans = start_spans + pred_spans
        gt_spans = targets['span_labels_nn']

        mask =  targets['timestamp_mask'].bool()
        mask_full = targets['timestamp_mask'].unsqueeze(2).repeat(1, 1, 2)
        mask_valid =  targets['timestamp_window'].bool()
        mask_valid_full = targets['timestamp_window'].unsqueeze(2).repeat(1, 1, 2)

        loss_span = F.smooth_l1_loss(src_spans, gt_spans, reduction='none') * mask_valid_full
        loss_giou = 1 - torch.diag(generalized_temporal_iou(src_spans[mask_valid], gt_spans[mask_valid]))

        losses = {}
        losses['loss_b'] = loss_span.sum() / mask_valid.sum()
        losses['loss_g'] = loss_giou.mean()
        return losses

    def loss_labels(self, outputs, targets, indices, log=True):
        saliency_scores = targets["saliency_scores"]
        if saliency_scores.sum() == 0:
            return {"loss_f": 0.}
        
        src_logits = outputs['pred_logits'].squeeze(-1)  # (batch_size, #queries, #classes=2)
        target_classes = targets["saliency_scores"].squeeze() 

        weights = torch.ones_like(target_classes).float() * self.empty_weight[1]
        weights[target_classes.bool()] = self.empty_weight[0]

        loss_ce = F.binary_cross_entropy(src_logits, target_classes.float(),  reduction="none")
        return {"loss_f": loss_ce.sum() / target_classes.sum()}
        # return {"loss_f": loss_ce.sum() / len(target_classes)}

        # mask = targets['timestamp_mask'].bool()
        # mask_valid = targets['timestamp_window'].bool()
        # target_classes = torch.full(src_logits.shape[:2], 0, dtype=torch.int64, device=src_logits.device)  # (batch_size, #queries)
        # target_classes[mask_valid] = 1
        # # target_classes = targets['timestamp_window']  # soft cls.
        # target_classes.float()
        # # pdb.set_trace()

        # weights = torch.zeros_like(target_classes).float()
        # weights[mask] = self.empty_weight[1]
        # weights[mask_valid] = self.empty_weight[0]

        # loss_ce = F.binary_cross_entropy(src_logits, target_classes.float(), weight=weights,  reduction="none") * mask
        # # return {"loss_f": loss_ce.sum() / mask.sum()}
        # return {"loss_f": loss_ce.sum() / mask_valid.sum()}

    def loss_saliency(self, outputs, targets, indices, log=True):
        """higher scores for positive clips"""
        if "saliency_pos_labels" not in targets:
            return {"loss_s_inter": 0., "loss_s_intra": 0.}
        saliency_scores = targets["saliency_scores"]
        if saliency_scores.sum() == 0:
            return {"loss_s_inter": 0., "loss_s_intra": 0.}

        # * qfvs mil-nce mode
        pos_indices = saliency_scores.squeeze() > 0

        sim = outputs['saliency_scores']
        sim_soft = F.softmax(sim / self.temperature, dim=0)
        sim_log = torch.log(sim_soft[pos_indices])
        loss_saliency_intra = -sim_log.sum() / len(sim_log)
        return {"loss_s_inter": 0., "loss_s_intra": loss_saliency_intra}

        # * inter-vid mode
        # vid_mem_proj = outputs["vid_mem_proj"]
        # pos_indices = targets["saliency_pos_labels"][:,0].long()  # (N, #pairs)
        # batch_indices = torch.arange(len(vid_mem_proj)).to(vid_mem_proj.device)

        # vid_feats = vid_mem_proj[batch_indices, pos_indices]
        # txt_feats = outputs["txt_mem_proj"].squeeze(1)
        # sim = sim_matrix(vid_feats, txt_feats)

        # i_logsm = F.log_softmax(sim / self.temperature, dim=1)
        # j_logsm = F.log_softmax(sim.t() /self.temperature, dim=1)

        # # sum over positives
        # idiag = torch.diag(i_logsm)
        # jdiag = torch.diag(j_logsm)
        # loss_i = idiag.sum() / len(idiag)
        # loss_j = jdiag.sum() / len(jdiag)

        # loss_saliency_inter = - loss_i - loss_j

        # # * intra-vid mode
        # mask = targets['timestamp_mask']
        # selected_scores = saliency_scores[batch_indices, pos_indices].unsqueeze(-1)
        # neg_indices_in = (saliency_scores < selected_scores)
        # neg_indices_in[batch_indices, pos_indices] = True
        # mask_invalid = neg_indices_in * mask.bool()

        # sim_in = F.cosine_similarity(vid_mem_proj, txt_feats.unsqueeze(1), dim=-1)
        # sim_in = sim_in + (mask_invalid + 1e-45).log()
        # logsm_in_i = F.log_softmax(sim_in / self.temperature, dim=1)
        # logsm_in_j = F.log_softmax(sim_in.t() / self.temperature, dim=1)

        # pos_logsm_in_i = logsm_in_i[batch_indices, pos_indices]
        # pos_logsm_in_j = logsm_in_j[pos_indices, batch_indices]
        # loss_in_i = pos_logsm_in_i.sum() / len(pos_logsm_in_i)
        # loss_in_j = pos_logsm_in_j.sum() / len(pos_logsm_in_j)

        # loss_saliency_intra = - loss_in_i - loss_in_j

        # return {"loss_s_inter": loss_saliency_inter, "loss_s_intra": loss_saliency_intra}

    def loss_saliency_cls(self, outputs, targets, indices, log=True):
        """higher scores for positive clips"""
        if "saliency_pos_labels" not in targets:
            return {"loss_s_inter": 0., "loss_s_intra": 0.}
        saliency_scores = targets["saliency_scores"]
        if saliency_scores.sum() == 0:
            return {"loss_s_inter": 0., "loss_s_intra": 0.}

        # * inter-vid mode
        vid_mem_proj = outputs["vid_mem_proj"]
        pos_indices = targets["saliency_pos_labels"][:,0].long()  # (N, #pairs)
        batch_indices = torch.arange(len(vid_mem_proj)).to(vid_mem_proj.device)

        vid_feats = vid_mem_proj[batch_indices, pos_indices]
        txt_feats = outputs["txt_mem_proj"].squeeze(1)
        sim = sim_matrix(vid_feats, txt_feats)

        i_logsm = F.log_softmax(sim / self.temperature, dim=1)
        j_logsm = F.log_softmax(sim.t() /self.temperature, dim=1)

        # sum over positives
        idiag = torch.diag(i_logsm)
        jdiag = torch.diag(j_logsm)
        loss_i = idiag.sum() / len(idiag)
        loss_j = jdiag.sum() / len(jdiag)

        loss_saliency_inter = - loss_i - loss_j

        # * intra-vid mode
        if 'cls_idx' not in targets.keys(): # eval
            return {"loss_s_inter": loss_saliency_inter}

        cls_indices = targets['cls_idx'].bool()
        cls_feats = outputs["cls_mem_proj"].squeeze(1)
        sim_cls = sim_matrix(vid_feats, cls_feats)

        i_logsm_cls = F.log_softmax(sim_cls / self.temperature, dim=1)
        idiag_cls = i_logsm_cls[cls_indices]
        loss_cls_i = idiag_cls.sum() / len(idiag_cls)

        loss_saliency_intra = - loss_cls_i

        return {"loss_s_inter": loss_saliency_inter, "loss_s_intra": loss_saliency_intra}

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {
            "spans": self.loss_spans,
            "labels": self.loss_labels,
            "saliency": self.loss_saliency,
            "saliency_cls": self.loss_saliency_cls,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def forward(self, outputs, targets, mask_GT=None):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        indices = None
        # Compute all the requested losses
        losses = {}
        outputs['pred_logits'] = outputs['pred_logits'].reshape(1, -1).masked_select(mask_GT[0])
        count = mask_GT.sum()
        outputs['saliency_scores'] = outputs['saliency_scores'].reshape(1, -1).masked_select(mask_GT[0])
        # targets['saliency_scores'] = targets['saliency_scores'].masked_select(mask_GT[0])
        targets['saliency_scores'] = targets['saliency_scores'][0,:count]

        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices))

        return losses

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class Conv(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, kernel_size):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        # self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.layers = nn.ModuleList(
            nn.Conv1d(n, k, kernel_size=kernel_size, stride=1, padding=kernel_size//2, dilation=1, groups=1, bias=True, padding_mode='zeros')
                                    for n, k in zip([input_dim] + h, h + [output_dim]))
    def forward(self, x):
        x = x.permute(0,2,1)
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x.permute(0, 2, 1)

class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""

    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz)
        layers = [
            nn.Dropout(dropout),
            nn.Linear(in_hsz, out_hsz)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x  # (N, L, D)


def build_model(args):
    device = torch.device(args.device)

    transformer = build_transformer(args)
    position_embedding, txt_position_embedding = build_position_encoding(args)

    model = Model(
        transformer,
        position_embedding,
        txt_position_embedding,
        txt_dim=args.t_feat_dim,
        vid_dim=args.v_feat_dim,
        input_dropout=args.input_dropout,
        span_loss_type=args.span_loss_type,
        use_txt_pos=args.use_txt_pos,
        n_input_proj=args.n_input_proj,
    )

    matcher = build_matcher(args)
    weight_dict = {"loss_b": args.b_loss_coef,
                   "loss_g": args.g_loss_coef,
                   "loss_f": args.f_loss_coef,
                   "loss_s_intra": args.s_loss_intra_coef,
                   "loss_s_inter": args.s_loss_inter_coef}

    if args.dset_type in ['mr', 'vlp']:
        if 'tal' not in args.train_path:
            losses = ['spans', 'labels', 'saliency']
        else:
            losses = ['spans', 'labels', 'saliency_cls']
    elif args.dset_type in ['hl', 'vs']:
        losses = ['labels', 'saliency']

    criterion = SetCriterion(
        matcher=matcher,
        weight_dict=weight_dict, losses=losses,
        eos_coef=args.eos_coef, temperature=args.temperature,
        span_loss_type=args.span_loss_type, max_v_l=args.max_v_l,
        saliency_margin=args.saliency_margin,
    )
    criterion.to(device)
    return model, criterion
