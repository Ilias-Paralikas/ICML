"""The three SSL baselines, reimplemented against a generic (image, one-hot label)
interface so they run on every dataset in this repo. Faithful to the papers'
mechanisms; hyperparameter defaults follow SSL4MIS.

  MeanTeacher  - Tarvainen & Valpola 2017 (segmentation form)
  UAMT         - Yu et al., MICCAI 2019 (uncertainty-aware Mean Teacher)
  BCP          - Bai et al., CVPR 2023 (bidirectional copy-paste)

Each class:
  step(x_l, y_l, x_u, it) -> {'loss': tensor, 'sup': float, 'cons': float, ...}
  after_step(it)          -> EMA teacher update (call after optimizer.step())
  eval_model()            -> the network to evaluate (student, or EMA if configured)

x_l, x_u : (B, C, H, W) float in [0, 1]
y_l      : (B, K, H, W) binary one-hot
"""
import copy

import torch

from .ssl_utils import (bcp_mix, dice_ce_loss, ema_update, enable_dropout, mc_uncertainty,
                        onehot, rand_box_mask, sigmoid_rampup, softmax_mse)


class _EMABase:
    def __init__(self, student, teacher, num_classes, device, cfg):
        self.student = student
        self.teacher = teacher
        self.num_classes = num_classes
        self.device = device
        self.cfg = cfg
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    def after_step(self, it):
        # ramp the EMA decay up (0.99 -> ema_decay) over the first 1k steps, as in SSL4MIS
        decay = min(1.0 - 1.0 / (it + 1), self.cfg['ema_decay'])
        ema_update(self.student, self.teacher, decay)

    def eval_model(self):
        return self.teacher if self.cfg.get('eval_ema') else self.student

    def _cons_weight(self, it):
        return self.cfg['consistency'] * sigmoid_rampup(it, self.cfg['rampup'])


class MeanTeacher(_EMABase):
    def step(self, x_l, y_l, x_u, it):
        sup = dice_ce_loss(self.student(x_l), y_l)

        noise_s = torch.clamp(torch.randn_like(x_u) * 0.1, -0.2, 0.2)
        noise_t = torch.clamp(torch.randn_like(x_u) * 0.1, -0.2, 0.2)
        stu = self.student(x_u + noise_s)
        with torch.no_grad():
            tea = self.teacher(x_u + noise_t)
        cons = softmax_mse(stu, tea)

        w = self._cons_weight(it)
        return {'loss': sup + w * cons, 'sup': sup.item(), 'cons': cons.item(), 'w': w}


class UAMT(_EMABase):
    def step(self, x_l, y_l, x_u, it):
        sup = dice_ce_loss(self.student(x_l), y_l)

        noise_s = torch.clamp(torch.randn_like(x_u) * 0.1, -0.2, 0.2)
        noise_t = torch.clamp(torch.randn_like(x_u) * 0.1, -0.2, 0.2)
        stu = self.student(x_u + noise_s)
        with torch.no_grad():
            tea_probs, ent = mc_uncertainty(self.teacher, x_u + noise_t,
                                            passes=self.cfg['mc_passes'])
        self.teacher.eval()  # restore (mc_uncertainty left dropout on)

        # keep only low-uncertainty pixels; threshold ramps 0.75->1.0 * ln(C)
        import math
        max_ent = math.log(self.num_classes)
        thr = (0.75 + 0.25 * sigmoid_rampup(it, self.cfg['rampup'])) * max_ent
        cert = (ent < thr).float()                       # (B, 1, H, W)
        se = (stu.softmax(1) - tea_probs) ** 2
        cons = (cert * se).sum() / (cert.sum() * self.num_classes + 1e-6)

        w = self._cons_weight(it)
        return {'loss': sup + w * cons, 'sup': sup.item(), 'cons': cons.item(), 'w': w}


class BCP(_EMABase):
    """Two phases keyed off `it`:
      it <  pretrain_iters : supervised copy-paste pre-training on labelled data only
      it >= pretrain_iters : bidirectional copy-paste self-training with the EMA
                             teacher supplying pseudo-labels on the unlabelled crop
    """

    def _pretrain_step(self, x_l, y_l):
        if x_l.shape[0] >= 2:
            perm = torch.randperm(x_l.shape[0], device=x_l.device)
            mask = rand_box_mask(x_l.shape[0], x_l.shape[2], x_l.shape[3],
                                 self.cfg['mask_frac'], x_l.device)
            xi, yi, xo, yo = bcp_mix(x_l, y_l, x_l[perm], y_l[perm], mask)
            loss = dice_ce_loss(self.student(xi), yi) + dice_ce_loss(self.student(xo), yo)
        else:
            loss = dice_ce_loss(self.student(x_l), y_l)          # <2 labels: plain sup
        return {'loss': loss, 'sup': loss.item(), 'cons': 0.0, 'phase': 'pretrain'}

    def _selftrain_step(self, x_l, y_l, x_u, it):
        with torch.no_grad():
            p_u = onehot(self.teacher(x_u).argmax(1), self.num_classes)

        mask = rand_box_mask(x_l.shape[0], x_l.shape[2], x_l.shape[3],
                             self.cfg['mask_frac'], x_l.device)
        # A = labelled (weight 1), B = unlabelled/pseudo (weight u_weight)
        xi, yi, xo, yo = bcp_mix(x_l, y_l, x_u, p_u, mask)
        uw = self.cfg['u_weight']
        w_in = mask + (1 - mask) * uw          # labelled centre full, pseudo surround down-weighted
        w_out = (1 - mask) + mask * uw
        loss = (dice_ce_loss(self.student(xi), yi, pix_weight=w_in)
                + dice_ce_loss(self.student(xo), yo, pix_weight=w_out))
        return {'loss': loss, 'sup': loss.item(), 'cons': 0.0, 'phase': 'selftrain'}

    def step(self, x_l, y_l, x_u, it):
        if it < self.cfg['pretrain_iters']:
            return self._pretrain_step(x_l, y_l)
        return self._selftrain_step(x_l, y_l, x_u, it)


METHODS = {'meanteacher': MeanTeacher, 'uamt': UAMT, 'bcp': BCP}


def default_cfg(method):
    cfg = dict(ema_decay=0.99, consistency=0.1, rampup=200, eval_ema=False)
    if method == 'uamt':
        cfg['mc_passes'] = 8
    if method == 'bcp':
        cfg.update(mask_frac=0.667, u_weight=0.5, pretrain_iters=2000)
    return cfg
