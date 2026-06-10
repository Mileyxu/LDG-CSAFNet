import random
from random import sample
import argparse
import numpy as np
import os
import pickle
import time
from torch import Tensor
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.covariance import LedoitWolf
from scipy.ndimage import gaussian_filter
from skimage import morphology
from scipy.ndimage import uniform_filter
from skimage.segmentation import mark_boundaries
from numpy import ndarray
from skimage import measure
from sklearn.metrics import auc
import pandas as pd
from tqdm import tqdm
from module.backbone.ResNet import ResNet
import torchvision.models as models
from module.cbam import ChannelGate, SpatialGate
import matplotlib.pyplot as plt
import matplotlib
import random
from torch.utils.data import DataLoader, Subset
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models import wide_resnet50_2, resnet18, resnet50
import datasets.mvtec as mvtec

# device setup
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')


def fit_ledoit_wolf_mean_cov(X, reg_covar=1e-6):
    X = np.asarray(X, dtype=np.float64)
    n_samples, n_dims = X.shape

    if n_samples < 2:
        mean = X[0] if n_samples == 1 else np.zeros(n_dims, dtype=np.float64)
        cov = np.eye(n_dims, dtype=np.float64)
    else:
        estimator = LedoitWolf().fit(X)
        mean = estimator.location_
        cov = estimator.covariance_

    cov = 0.5 * (cov + cov.T)
    cov = cov + reg_covar * np.eye(n_dims, dtype=np.float64)
    return mean, cov


def covariance_to_precision_logdet(cov, reg_covar=1e-6):
    cov = np.asarray(cov, dtype=np.float64)
    n_dims = cov.shape[0]
    eye = np.eye(n_dims, dtype=np.float64)
    jitter = reg_covar

    for _ in range(6):
        cov_jittered = 0.5 * (cov + cov.T) + jitter * eye
        sign, log_det = np.linalg.slogdet(cov_jittered)
        if sign > 0:
            precision = np.linalg.pinv(cov_jittered)
            return precision, log_det
        jitter *= 10

    cov_jittered = 0.5 * (cov + cov.T) + max(jitter, 1e-3) * eye
    precision = np.linalg.pinv(cov_jittered)
    sign, log_det = np.linalg.slogdet(cov_jittered)
    return precision, log_det if sign > 0 else 0.0


def covariance_to_diag_precision_logdet(cov, reg_covar=1e-6):
    var = np.diag(np.asarray(cov, dtype=np.float64))
    var = np.maximum(var, reg_covar)
    precision_diag = 1.0 / var
    log_det = np.sum(np.log(var))
    return precision_diag, log_det


def fit_lw_gaussian_distribution(embedding_vectors, reg_covar=1e-6,
                                 covariance_mode='diag'):
    """Fit a single Ledoit-Wolf shrinkage-regularized Gaussian at each spatial location."""
    B, C, H, W = embedding_vectors.size()
    features = embedding_vectors.view(B, C, H * W).detach().cpu().numpy()
    P = H * W
    covariance_mode = covariance_mode.lower()
    if covariance_mode not in ['diag', 'full']:
        raise ValueError("covariance_mode must be 'diag' or 'full'")

    means_all = np.zeros((P, C), dtype=np.float32)
    if covariance_mode == 'diag':
        precision_diags_all = np.zeros((P, C), dtype=np.float32)
        precisions_all = None
    else:
        precision_diags_all = None
        precisions_all = np.zeros((P, C, C), dtype=np.float32)

    for i in tqdm(range(P), '| Ledoit-Wolf Gaussian modeling |'):
        X = features[:, :, i].astype(np.float64)
        mean, cov = fit_ledoit_wolf_mean_cov(X, reg_covar)
        means_all[i] = mean.astype(np.float32)
        if covariance_mode == 'diag':
            precision_diag, _ = covariance_to_diag_precision_logdet(cov, reg_covar)
            precision_diags_all[i] = precision_diag.astype(np.float32)
        else:
            precision, _ = covariance_to_precision_logdet(cov, reg_covar)
            precisions_all[i] = precision.astype(np.float32)

    distribution = {
        'type': 'lw_gaussian',
        'covariance_mode': covariance_mode,
        'means': means_all,
        'shape': (H, W),
        'channels': C,
        'reg_covar': reg_covar,
    }
    if covariance_mode == 'diag':
        distribution['precision_diags'] = precision_diags_all
    else:
        distribution['precisions'] = precisions_all
    return distribution


def fit_gaussian_distribution(embedding_vectors, reg_covar=1e-6,
                              covariance_mode='diag', cov_method='lw'):
    """
    Fit a single Gaussian at each spatial location. cov_method only switches the
    covariance estimator, used for ablation:
      'lw'        : Ledoit-Wolf shrinkage (current default)
      'empirical' : empirical covariance + reg_covar*I (standard PaDiM; reg ~ 0.01 recommended)
      'identity'  : covariance = identity -> Mahalanobis distance degenerates to Euclidean (lower-bound baseline)
    When covariance_mode='diag', only the diagonal is used; 'full' uses the full covariance.
    The scoring function compute_lw_gaussian_score is decoupled from this function and needs no changes.
    """
    B, C, H, W = embedding_vectors.size()
    features = embedding_vectors.view(B, C, H * W).detach().cpu().numpy()
    P = H * W
    covariance_mode = covariance_mode.lower()

    means_all = np.zeros((P, C), dtype=np.float32)
    if covariance_mode == 'diag':
        precision_diags_all = np.zeros((P, C), dtype=np.float32)
        precisions_all = None
    else:
        precision_diags_all = None
        precisions_all = np.zeros((P, C, C), dtype=np.float32)

    eye = np.eye(C, dtype=np.float64)
    for i in tqdm(range(P), '| Gaussian modeling (%s, %s) |' % (cov_method, covariance_mode)):
        X = features[:, :, i].astype(np.float64)
        means_all[i] = X.mean(axis=0).astype(np.float32)

        if covariance_mode == 'diag':
            # Diagonal mode: compute per-channel variance directly, avoiding a full C x C
            # covariance for large dimensions (critical when d is large).
            if cov_method == 'lw':
                _, cov = fit_ledoit_wolf_mean_cov(X, reg_covar)
                var = np.diag(cov)
            elif cov_method == 'empirical':
                var = X.var(axis=0, ddof=1) if X.shape[0] >= 2 else np.ones(C)
            elif cov_method == 'identity':
                var = np.ones(C)
            else:
                raise ValueError("unknown cov_method: %s" % cov_method)
            var = np.maximum(var, reg_covar)
            precision_diags_all[i] = (1.0 / var).astype(np.float32)
        else:
            if cov_method == 'lw':
                _, cov = fit_ledoit_wolf_mean_cov(X, reg_covar)
            elif cov_method == 'empirical':
                cov = np.cov(X, rowvar=False) if X.shape[0] >= 2 else eye.copy()
                cov = 0.5 * (cov + cov.T) + reg_covar * eye
            elif cov_method == 'identity':
                cov = eye.copy()
            else:
                raise ValueError("unknown cov_method: %s" % cov_method)
            precision, _ = covariance_to_precision_logdet(cov, reg_covar)
            precisions_all[i] = precision.astype(np.float32)

    distribution = {
        'type': 'gaussian', 'covariance_mode': covariance_mode,
        'means': means_all, 'shape': (H, W), 'channels': C,
        'reg_covar': reg_covar, 'cov_method': cov_method,
    }
    if covariance_mode == 'diag':
        distribution['precision_diags'] = precision_diags_all
    else:
        distribution['precisions'] = precisions_all
    return distribution


def compute_lw_gaussian_score(embedding_vectors, distribution, chunk_size=512):
    """Single-Gaussian Mahalanobis-distance scoring (GPU version, numerically identical to the
    original numpy version). On the first call, distribution parameters (means/precisions) are
    moved to device and cached into distribution, and reused for per-image inference."""
    B, C, H, W = embedding_vectors.size()
    P = H * W
    chunk_size = max(1, int(chunk_size))
    diag = distribution.get('covariance_mode', 'diag') == 'diag'

    # features: (B, P, C) on device
    feats = embedding_vectors.view(B, C, P).permute(0, 2, 1).to(device).float()

    # Cache distribution parameters to device (moved only once)
    if distribution.get('_dev_cached', None) != str(device):
        distribution['_means_t'] = torch.as_tensor(distribution['means'], dtype=torch.float32, device=device)
        if diag:
            distribution['_prec_t'] = torch.as_tensor(distribution['precision_diags'], dtype=torch.float32, device=device)
        else:
            distribution['_prec_t'] = torch.as_tensor(distribution['precisions'], dtype=torch.float32, device=device)
        distribution['_dev_cached'] = str(device)
    means_t = distribution['_means_t']        # (P, C)
    prec_t = distribution['_prec_t']          # diag: (P, C) ; full: (P, C, C)

    dist_list = torch.zeros((B, P), dtype=torch.float32, device=device)
    with torch.no_grad():
        for start in range(0, P, chunk_size):
            end = min(start + chunk_size, P)
            diff = feats[:, start:end, :] - means_t[start:end].unsqueeze(0)        # (B, p, C)
            if diag:
                md2 = (diff * diff * prec_t[start:end].unsqueeze(0)).sum(-1)       # (B, p)
            else:
                md2 = torch.einsum('bpc,pcd,bpd->bp', diff, prec_t[start:end], diff)
            md2 = md2.clamp_min(0.0)
            dist_list[:, start:end] = md2.sqrt()

    return dist_list.reshape(B, H, W).detach().cpu().numpy().astype(np.float32)


def parse_args():
    parser = argparse.ArgumentParser('LDG-CSAFNe')  # mvtec_anomaly_detection
    parser.add_argument('--data_path', type=str, default='/home/featurize/work/UFFDM/datasets/VisA')
    parser.add_argument('--save_path', type=str, default='/tmp/mvtec_result')
    parser.add_argument('--arch', type=str, choices=['resnet18', 'wide_resnet50_2', 'resnet50'], default='resnet50')
    parser.add_argument('--feature_dim', type=int, default=64,
                        help='Randomly selected embedding dimension for distribution modeling')
    parser.add_argument('--covariance_mode', type=str, choices=['diag', 'full'], default='diag',
                        help='Use diag for fast inference, full for full-covariance ablation')
    parser.add_argument('--gmm_reg_covar', type=float, default=1e-6,
                        help='Small diagonal jitter added after Ledoit-Wolf covariance estimation')
    parser.add_argument('--score_chunk_size', type=int, default=512,
                        help='Number of patch locations scored at once during vectorized inference')
    parser.add_argument('--cov_method', type=str, choices=['lw', 'empirical', 'identity'], default='lw',
                        help='Covariance estimator for ablation: lw / empirical / identity')
    parser.add_argument('--no_plot', action='store_true',
                        help='Skip per-image visualization to speed up ablation runs')
    parser.add_argument('--px_per_mm', type=float, default=None,
                        help='Pixels per millimeter; when provided, draws a 1mm scale bar in the visualization (fill in for the aluminium data)')
    parser.add_argument('--embed_mode', type=str, choices=['ldcf', 'padim'], default='ldcf',
                        help='ldcf = current fusion (random untrained LDCF); padim = standard PaDiM concatenation of layer1+2+3')
    parser.add_argument('--dump_scores', action='store_true',
                        help='Save per-class raw score maps/masks/labels/images as npz for offline analysis by evaluate_comment5.py')
    parser.add_argument('--data_percent', type=float, default=1,
                        help='Percentage of training data to use (0~1)')
    parser.add_argument('--data_percentest', type=float, default=1,
                        help='Percentage of training data to use (0~1)')
    parser.add_argument('--seed', type=int, default=1024,
                        help='Random seed: controls random dimension selection idx and LDCF initialization (use multiple seeds for ablation)')
    parser.add_argument('--ldcf_variant', type=str, default='full',
                        choices=['full', 'no_csaf', 'no_ldg', 'no_ffm', 'no_channel', 'no_spatial'],
                        help='LDCF ablation variant (only effective when embed_mode=ldcf)')
    parser.add_argument('--time_infer', action='store_true',
                        help='Measure end-to-end per-image inference time (backbone+fusion+scoring+upsampling+smoothing, bypassing cache); report mean ms per image')
    return parser.parse_args()


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class LDCF(nn.Module):
    def __init__(self, channels_high, channels_low, variant='full'):
        super(LDCF, self).__init__()
        self.variant = variant
        self.deConv = nn.ConvTranspose2d(channels_high, channels_low, 2, stride=2)

        self.ChannelGate = ChannelGate(channels_high)
        self.SpatialGate = SpatialGate()
        self.conv1 = conv1x1(channels_high, channels_low)
        self.conv2 = conv3x3(channels_low, channels_low)
        self.relu = nn.ReLU(inplace=True)

        # Layer Difference attention branch
        self.diff_conv = nn.Sequential(
            conv3x3(channels_low, channels_low),
            nn.ReLU(inplace=True),
            conv1x1(channels_low, 1),
            nn.Sigmoid()
        )

    def forward(self, fms_high, fms_low):
        v = self.variant
        # ---- CSAF: channel gating -> upsampling -> spatial gating ----
        # no_csaf: replace attention fusion with element-wise addition (neither gate recalibrates)
        h = fms_high if v in ('no_channel', 'no_csaf') else self.ChannelGate(fms_high)
        x1 = self.deConv(h)
        if v not in ('no_spatial', 'no_csaf'):
            x1 = self.SpatialGate(x1)

        x3 = self.conv2(fms_low)
        x3_resized = F.interpolate(x3, size=x1.shape[2:], mode='bilinear', align_corners=False)

        # ---- LDG: layer-difference attention. With no_ldg / no_csaf it degenerates to plain addition (attn=1) ----
        if v in ('no_ldg', 'no_csaf'):
            out = x1 + x3_resized
        else:
            diff = torch.abs(x1 - x3_resized)
            attn = self.diff_conv(diff)
            out = x1 + x3_resized * attn

        out = self.relu(out)
        return out


def group_reduce(x, out_ch):
    """Deterministic grouped-average channel reduction: split channels into out_ch groups and average.
    Parameter-free and backbone-agnostic. Requires the channel count of x to be divisible by out_ch."""
    B, C, H, W = x.size()
    assert C % out_ch == 0, "channel %d not divisible by %d" % (C, out_ch)
    return x.view(B, out_ch, C // out_ch, H, W).mean(dim=2)


@torch.no_grad()
def build_embedding(fmap1, fmap2, fmap3, fmap4,
                    ldcf1, ldcf2, ldcf3, bn,
                    w_512_256, w_256_128, w_512_512, idx, variant='full'):
    """
    Fuse a batch of intermediate features into an embedding (entirely no_grad, computed on device).
    Channel reduction is done entirely via deterministic grouped averaging (without borrowing
    pretrained weights), so it works for both resnet50 / wide_resnet50_2.
    Inputs (same shapes for both backbones):
      fmap1: stem maxpool  (64,  H/4)
      fmap2: layer1        (256, H/4)  -> reduced to 64 inside the function
      fmap3: layer1        (256, H/4)
      fmap4: layer2        (512, H/8)
    """
    H4 = fmap1.shape[2:]                      # layer1 resolution, e.g. 56x56
    fmap2 = group_reduce(fmap2, 64)           # 256 -> 64, matching the fmap2 channels of the original pipeline

    fmap1_resized = F.interpolate(fmap1, scale_factor=4, mode='bilinear', align_corners=False)        # 64, H
    fmap_cat12 = embedding_concat(fmap1, fmap2)                                   # 128, H/4
    fmap_cat12_resized = F.interpolate(fmap_cat12, scale_factor=2, mode='bilinear', align_corners=False)  # 128, H/2
    # ---- LDCF ----
    f_ldcf3 = ldcf3(fmap4, fmap3)                                                # 256, H/4
    f_ldcf2 = ldcf2(f_ldcf3, fmap_cat12_resized)                                 # 128, H/2
    f_ldcf1 = ldcf1(f_ldcf2, fmap1_resized)                                      # 64,  H
    f_ldcf1 = F.max_pool2d(f_ldcf1, kernel_size=4, stride=4)                     # 64,  H/4
    f_ldcf1 = F.interpolate(f_ldcf1, size=H4, mode='nearest')                    # align to layer1 resolution
    del f_ldcf3, f_ldcf2, fmap1_resized, fmap_cat12_resized

    fmap1 = embedding_add(fmap1, f_ldcf1)
    # fmap1 = f_ldcf1
    del f_ldcf1

    size1 = fmap1.size()[2:]
    fmap_cat12 = embedding_concat(fmap1, fmap2)                                  # 128, H/4 (recomputed with the updated fmap1)

    fmap4_resized = F.interpolate(fmap4, size=size1, mode='nearest')             # 512, H/4
    fmap4_256 = group_reduce(fmap4_resized, 256)                                 # 512 -> 256
    del fmap4_resized
    fmap_add34 = embedding_add(fmap3, fmap4_256)                                 # 256
    fmap_add34_128 = group_reduce(fmap_add34, 128)                               # 256 -> 128
    del fmap_add34
    fmap_cat34 = embedding_concat(fmap3, fmap4_256)                              # 512
    del fmap4_256
    fmap_cat34_256 = group_reduce(fmap_cat34, 256)                               # 512 -> 256
    del fmap_cat34
    fmap_cat12add34 = embedding_concat(fmap_cat12, fmap_add34_128)               # 256
    del fmap_cat12, fmap_add34_128
    fmap_cat1234 = embedding_concat(fmap_cat12add34, fmap_cat34_256)             # 512
    del fmap_cat12add34, fmap_cat34_256

    # ---- FFM: 1x1 conv + BN fusion head. With no_ffm it is removed and the concatenation is used directly ----
    if variant == 'no_ffm':
        embedding = fmap_cat1234
    else:
        embedding = fmap_cat1234 + bn(fmap_cat1234)                              # 512
    del fmap_cat1234

    embedding = torch.index_select(embedding, 1, idx)
    return embedding


@torch.no_grad()
def build_embedding_padim(layer1, layer2, layer3, idx):
    """Standard PaDiM embedding: concatenate layer1+layer2+layer3, then randomly select d dimensions."""
    emb = embedding_concat(layer1, layer2)
    emb = embedding_concat(emb, layer3)
    emb = torch.index_select(emb, 1, idx)
    return emb


def main():
    args = parse_args()

    # load model
    if args.arch == 'resnet18':
        model = resnet18(pretrained=True, progress=True)
        t_d = 448
    elif args.arch == 'wide_resnet50_2':
        model = wide_resnet50_2(pretrained=True, progress=True)
        t_d = 1792
    elif args.arch == 'resnet50':
        model = resnet50(pretrained=True, progress=True)
        t_d = 512
    d = min(args.feature_dim, t_d)
    model.to(device)
    model.eval()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)

    # The embedding channel count depends on embed_mode:
    #   ldcf  : fixed 512 channels after fusion
    #   padim : standard PaDiM, concatenation of layer1+layer2+layer3
    if args.embed_mode == 'padim':
        emb_dim = 448 if args.arch == 'resnet18' else 1792
    else:
        emb_dim = 512
    d = min(args.feature_dim, emb_dim)
    idx = torch.tensor(sample(range(0, emb_dim), d)).to(device)

    # Only the ldcf mode needs these fusion modules (random weights, shared by train/test and set to eval)
    ldcf1 = ldcf2 = ldcf3 = bn = None
    w_512_256 = w_256_128 = w_512_512 = None   # deprecated: reduction switched to group_reduce, no longer borrows pretrained weights
    if args.embed_mode == 'ldcf':
        ldcf3 = LDCF(512, 256, variant=args.ldcf_variant).to(device).eval()
        ldcf2 = LDCF(256, 128, variant=args.ldcf_variant).to(device).eval()
        ldcf1 = LDCF(128, 64, variant=args.ldcf_variant).to(device).eval()
        bn = nn.BatchNorm2d(512).to(device).eval()

    # set model's intermediate outputs
    outputs = []

    def hook(module, input, output):
        outputs.append(output)

    if args.embed_mode == 'ldcf':
        model.maxpool.register_forward_hook(hook)          # fmap1 = stem maxpool, 64,  H/4
        model.layer1[-1].register_forward_hook(hook)       # fmap2 = layer1,       256, H/4 (reduced to 64)
        model.layer1[-1].register_forward_hook(hook)       # fmap3 = layer1,       256, H/4
        model.layer2[-1].register_forward_hook(hook)       # fmap4 = layer2,       512, H/8
    else:  # padim
        model.layer1[-1].register_forward_hook(hook)       # layer1
        model.layer2[-1].register_forward_hook(hook)       # layer2
        model.layer3[-1].register_forward_hook(hook)       # layer3

    os.makedirs(os.path.join(args.save_path, 'temp_%s' % args.arch), exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    fig_img_rocauc = ax[0]
    fig_pixel_rocauc = ax[1]

    total_roc_auc = []
    total_pixel_roc_auc = []

    for class_name in mvtec.CLASS_NAMES:

        train_dataset = mvtec.MVTecDataset(args, class_name=class_name, is_train=True)
        train_dataloader = DataLoader(train_dataset, batch_size=32, pin_memory=True)
        test_dataset = mvtec.MVTecDataset(args, class_name=class_name, is_train=False)
        test_dataloader = DataLoader(test_dataset, batch_size=32, pin_memory=True)

        # Sample the training set by percentage
        full_dataset = train_dataloader.dataset
        data_len = len(full_dataset)
        used_len = int(data_len * args.data_percent)
        indices = random.sample(range(data_len), used_len)
        subset = Subset(full_dataset, indices)
        train_dataloader = torch.utils.data.DataLoader(subset, batch_size=train_dataloader.batch_size, shuffle=False)

        emb_train_path = os.path.join(
            args.save_path, 'temp_%s' % args.arch,
            'emb_train_%s_%s_%s_s%d_D%d.pt' % (class_name, args.embed_mode, args.ldcf_variant, args.seed, d))
        dist_path = os.path.join(
            args.save_path, 'temp_%s' % args.arch,
            'dist_%s_%s_%s_%s_%s_s%d_reg%.0e_D%d.pkl' % (
                class_name, args.embed_mode, args.ldcf_variant, args.covariance_mode,
                args.cov_method, args.seed, args.gmm_reg_covar, d))

        # ---- 1) Load/build training embedding: once cached, different cov_methods reuse the same copy to ensure a fair comparison ----
        if os.path.exists(emb_train_path):
            embedding_vectors = torch.load(emb_train_path, weights_only=False)
        else:
            embedding_list = []
            for (x, _, _) in tqdm(train_dataloader, '| feature extraction | train | %s |' % class_name):
                with torch.no_grad():
                    _ = model(x.to(device))
                if args.embed_mode == 'padim':
                    emb = build_embedding_padim(outputs[0], outputs[1], outputs[2], idx)
                else:
                    emb = build_embedding(outputs[0], outputs[1], outputs[2], outputs[3],
                                          ldcf1, ldcf2, ldcf3, bn,
                                          w_512_256, w_256_128, w_512_512, idx,
                                          variant=args.ldcf_variant)
                embedding_list.append(emb.cpu())
                outputs = []
            embedding_vectors = torch.cat(embedding_list, 0)
            del embedding_list
            torch.save(embedding_vectors, emb_train_path)

        # ---- 2) Fit a single Gaussian using the chosen covariance estimator ----
        if os.path.exists(dist_path):
            print('load distribution from: %s' % dist_path)
            with open(dist_path, 'rb') as f:
                train_outputs = pickle.load(f)
        else:
            train_outputs = fit_gaussian_distribution(
                embedding_vectors,
                reg_covar=args.gmm_reg_covar,
                covariance_mode=args.covariance_mode,
                cov_method=args.cov_method,
            )
            with open(dist_path, 'wb') as f:
                pickle.dump(train_outputs, f)

        # ---- Training-set normal-image scores (for the label-free deployment threshold: 99th percentile of training-normal) ----
        train_amaps = None
        if args.dump_scores:
            tr_dist = compute_lw_gaussian_score(embedding_vectors, train_outputs,
                                                chunk_size=args.score_chunk_size)
            train_amaps = torch.tensor(tr_dist)
            train_amaps = F.interpolate(train_amaps.unsqueeze(1), size=(224, 224),
                                        mode='bilinear', align_corners=False).squeeze(1).numpy()
            for ti in range(train_amaps.shape[0]):
                train_amaps[ti] = gaussian_filter(train_amaps[ti], sigma=2)
            train_amaps = train_amaps.astype(np.float32)
        del embedding_vectors

        gt_list = []
        gt_mask_list = []
        test_imgs = []

        # Sample the test set by percentage
        full_dataset = test_dataloader.dataset
        data_len = len(full_dataset)
        used_len = int(data_len * args.data_percentest)
        indices = random.sample(range(data_len), used_len)
        subset = Subset(full_dataset, indices)
        test_dataloader = torch.utils.data.DataLoader(subset, batch_size=test_dataloader.batch_size, shuffle=False)

        # ===== Load/build test embedding (cached together with labels/masks/images for instant ablation loading) =====
        emb_test_path = os.path.join(
            args.save_path, 'temp_%s' % args.arch,
            'emb_test_%s_%s_%s_s%d_D%d.pt' % (class_name, args.embed_mode, args.ldcf_variant, args.seed, d))
        if os.path.exists(emb_test_path):
            bundle = torch.load(emb_test_path, weights_only=False)
            embedding_vectors = bundle['emb']
            test_imgs = bundle['imgs']
            gt_list = bundle['gt']
            gt_mask_list = bundle['mask']
            img_size = bundle['size']
        else:
            test_embedding_list = []
            img_size = None
            for (x, y, mask) in tqdm(test_dataloader, '| feature extraction | test | %s |' % class_name):
                test_imgs.extend(x.cpu().detach().numpy())
                gt_list.extend(y.cpu().detach().numpy())
                gt_mask_list.extend(mask.cpu().detach().numpy())
                img_size = x.size()[2:]
                with torch.no_grad():
                    _ = model(x.to(device))
                if args.embed_mode == 'padim':
                    emb = build_embedding_padim(outputs[0], outputs[1], outputs[2], idx)
                else:
                    emb = build_embedding(outputs[0], outputs[1], outputs[2], outputs[3],
                                          ldcf1, ldcf2, ldcf3, bn,
                                          w_512_256, w_256_128, w_512_512, idx,
                                          variant=args.ldcf_variant)
                test_embedding_list.append(emb.cpu())
                outputs = []
            embedding_vectors = torch.cat(test_embedding_list, 0)
            del test_embedding_list
            torch.save({'emb': embedding_vectors, 'imgs': test_imgs, 'gt': gt_list,
                        'mask': gt_mask_list, 'size': img_size}, emb_test_path)

        # calculate distance matrix
        inference_start = time.time()
        dist_list = compute_lw_gaussian_score(
            embedding_vectors,
            train_outputs,
            chunk_size=args.score_chunk_size
        )
        del embedding_vectors

        # upsample
        dist_list = torch.tensor(dist_list)
        score_map = F.interpolate(dist_list.unsqueeze(1), size=img_size, mode='bilinear',
                                  align_corners=False).squeeze(1).numpy()

        # apply gaussian smoothing on the score map
        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=2)
        inference_time = time.time() - inference_start
        n_test = score_map.shape[0]
        per_image_ms = inference_time / max(n_test, 1) * 1000.0
        print('{} inference time: total {:.3f}s | per-image {:.2f} ms (scoring stage, {} imgs)'.format(
            class_name, inference_time, per_image_ms, n_test))

        # ---- End-to-end per-image inference timing (backbone + fusion + scoring + upsampling + smoothing), bypassing cache ----
        e2e_ms = float('nan')
        if args.time_infer:
            per_times = []
            warmup = 2
            for ti, im in enumerate(test_imgs):
                xt = torch.from_numpy(np.asarray(im))[None].to(device)
                if use_cuda:
                    torch.cuda.synchronize()
                t0 = time.time()
                with torch.no_grad():
                    _ = model(xt)
                    if args.embed_mode == 'padim':
                        emb1 = build_embedding_padim(outputs[0], outputs[1], outputs[2], idx)
                    else:
                        emb1 = build_embedding(outputs[0], outputs[1], outputs[2], outputs[3],
                                               ldcf1, ldcf2, ldcf3, bn,
                                               w_512_256, w_256_128, w_512_512, idx,
                                               variant=args.ldcf_variant)
                d1 = compute_lw_gaussian_score(emb1, train_outputs, chunk_size=args.score_chunk_size)
                d1 = torch.tensor(d1)
                sm = F.interpolate(d1.unsqueeze(1), size=img_size, mode='bilinear',
                                   align_corners=False).squeeze(1).numpy()
                sm[0] = gaussian_filter(sm[0], sigma=2)
                if use_cuda:
                    torch.cuda.synchronize()
                outputs = []
                if ti >= warmup:                       # skip the first few warm-up images
                    per_times.append(time.time() - t0)
            if per_times:
                e2e_ms = float(np.mean(per_times) * 1000.0)
                print('{} end-to-end per-image inference: {:.2f} ± {:.2f} ms (n={})'.format(
                    class_name, e2e_ms, float(np.std(per_times) * 1000.0), len(per_times)))

        # Normalization
        max_score = score_map.max()
        min_score = score_map.min()
        scores = (score_map - min_score) / (max_score - min_score + 1e-12)

        # calculate image-level ROC AUC score
        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
        gt_list = np.asarray(gt_list)
        fpr, tpr, _ = roc_curve(gt_list, img_scores)
        img_roc_auc = roc_auc_score(gt_list, img_scores)
        total_roc_auc.append(img_roc_auc)
        print('image ROCAUC: %.3f' % (img_roc_auc))
        fig_img_rocauc.plot(fpr, tpr, label='%s img_ROCAUC: %.3f' % (class_name, img_roc_auc), linewidth=4)

        # get optimal threshold
        gt_mask = np.asarray(gt_mask_list)
        precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        threshold = thresholds[np.argmax(f1)]

        # calculate per-pixel level ROCAUC
        fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
        per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
        total_pixel_roc_auc.append(per_pixel_rocauc)
        print('pixel ROCAUC: %.3f' % (per_pixel_rocauc))

        predictions = (scores > threshold).astype(int)
        f1 = calculate_f1_score(gt_mask.flatten(), predictions.flatten())
        print('F1 Score:', f1)

        gt_mask_np = np.asarray(gt_mask_list).squeeze(1).astype(np.uint8)
        aupro = compute_pro(gt_mask_np, scores)
        print('PRO AUC: {:.3f}'.format(aupro))

        # Machine-parseable result row for aggregation by run_ablation.py (last two columns: scoring-stage ms / end-to-end ms)
        print('RESULT,%s,%s,%s,%d,%.6f,%.6f,%.6f,%.6f,%.3f,%.3f' % (
            args.embed_mode, args.ldcf_variant, class_name, args.seed,
            img_roc_auc, per_pixel_rocauc, f1, aupro, per_image_ms, e2e_ms))

        if args.dump_scores:
            # Save the "smoothed, not min-max normalized" raw score maps, convenient for the label-free deployment threshold
            dump_dir = os.path.join(args.save_path, 'scores_dump')
            os.makedirs(dump_dir, exist_ok=True)
            amaps_raw = score_map.astype(np.float32)                       # (N, H, W)
            masks_bin = gt_mask_np                                         # (N, H, W) in {0,1}
            labels_arr = np.asarray(gt_list).astype(np.uint8)             # (N,)
            imgs_u8 = np.stack([denormalization(im) for im in test_imgs]).astype(np.uint8)  # (N,H,W,3)
            out_npz = os.path.join(dump_dir, '%s_%s_%s_%s_D%d.npz' % (
                class_name, args.arch, args.embed_mode, args.cov_method, d))
            np.savez_compressed(out_npz, amaps=amaps_raw, masks=masks_bin,
                                labels=labels_arr, imgs=imgs_u8,
                                train_amaps=(train_amaps if train_amaps is not None
                                             else np.zeros((0, 1, 1), np.float32)))
            print('dumped scores to: %s' % out_npz)

        fig_pixel_rocauc.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (class_name, per_pixel_rocauc), linewidth=4)
        if not args.no_plot:
            save_dir = args.save_path + '/' + f'pictures_{args.arch}'
            os.makedirs(save_dir, exist_ok=True)
            # scores are already min-max normalized to [0,1] for this class; the fixed scale keeps colors consistent across images/classes
            plot_fig(test_imgs, scores, gt_mask_list, threshold, save_dir, class_name,
                     vmin=0.0, vmax=1.0, px_per_mm=args.px_per_mm)

    print('Average ROCAUC: %.3f' % np.mean(total_roc_auc))
    fig_img_rocauc.set_title('Average image ROCAUC: %.3f' % np.mean(total_roc_auc), fontsize=28)
    fig_img_rocauc.legend(loc="lower right")

    print('Average pixel ROCUAC: %.3f' % np.mean(total_pixel_roc_auc))
    fig_pixel_rocauc.set_title('Average pixel ROCAUC: %.3f' % np.mean(total_pixel_roc_auc), fontsize=28)
    fig_pixel_rocauc.legend(loc="lower right", fontsize=28)

    fig_img_rocauc.legend(loc="lower right", fontsize=28)
    fig_pixel_rocauc.legend(loc="lower right", fontsize=28)

    fig_img_rocauc.set_yticks([0, 0.5, 1])
    fig_pixel_rocauc.set_yticks([0, 0.5, 1])

    fig_img_rocauc.set_xlabel('False Positive Rate', fontsize=28)
    fig_img_rocauc.set_ylabel('True Positive Rate', fontsize=28)
    fig_pixel_rocauc.set_xlabel('False Positive Rate', fontsize=28)
    fig_pixel_rocauc.set_ylabel('True Positive Rate', fontsize=28)

    fig_img_rocauc.tick_params(axis='both', which='major', labelsize=28)
    fig_img_rocauc.tick_params(axis='both', which='minor', labelsize=28)
    fig_pixel_rocauc.tick_params(axis='both', which='major', labelsize=28)
    fig_pixel_rocauc.tick_params(axis='both', which='minor', labelsize=28)

    fig.tight_layout()
    fig.savefig(os.path.join(args.save_path, 'roc_curve.png'), dpi=100)


def plot_fig(test_img, scores, gts, threshold, save_dir, class_name,
             vmin=None, vmax=None, px_per_mm=None, inset=True, dpi=300):
    """
    Publication-quality visualization (addressing review Comment 8):
      - All figures share the same fixed color scale vmin/vmax (comparable across samples), with a shared colorbar;
      - Automatically locates the defect region and draws a zoomed inset to reveal weak/small defects;
      - Optional scale bar; requires px_per_mm;
      - Larger font sizes, standardized titles, and does not modify the passed-in scores (avoids contaminating metrics).
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    num = len(scores)
    # Fixed, cross-sample-consistent color scale
    if vmin is None:
        vmin = float(np.min(scores))
    if vmax is None:
        vmax = float(np.max(scores))
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap('jet')
    titles = ['Input', 'Ground truth', 'Anomaly heat map', 'Predicted mask', 'Overlay']

    for i in range(num):
        img = denormalization(test_img[i])
        gt = gts[i].transpose(1, 2, 0).squeeze()
        amap = scores[i]                                   # not modified in place
        H, W = amap.shape

        mask = (amap > threshold).astype(np.uint8)
        mask = morphology.opening(mask, morphology.disk(4))
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')

        fig_img, ax_img = plt.subplots(1, 5, figsize=(16, 3.4))
        fig_img.subplots_adjust(right=0.9, wspace=0.06)
        for ax_i in ax_img:
            ax_i.set_xticks([]); ax_i.set_yticks([])

        ax_img[0].imshow(img)
        ax_img[1].imshow(gt, cmap='gray', vmin=0, vmax=1)
        ax_img[2].imshow(img)
        im = ax_img[2].imshow(amap, cmap=cmap, norm=norm, alpha=0.55, interpolation='nearest')
        ax_img[3].imshow(mask, cmap='gray', vmin=0, vmax=1)
        ax_img[4].imshow(vis_img)
        for c in range(5):
            ax_img[c].set_title(titles[c], fontsize=13)

        # ---- Zoomed inset: crop and magnify a block centered on the defect centroid (or the heatmap peak when no GT) ----
        if inset:
            if gt.sum() > 0:
                ys, xs = np.where(gt > 0)
                cy, cx = int(ys.mean()), int(xs.mean())
            else:
                cy, cx = np.unravel_index(int(np.argmax(amap)), amap.shape)
            half = max(H, W) // 6
            y0, y1 = max(0, cy - half), min(H, cy + half)
            x0, x1 = max(0, cx - half), min(W, cx + half)
            # Place a zoomed inset in both the Input and Heat map columns
            for col, base in [(0, img), (2, None)]:
                axins = inset_axes(ax_img[col], width='45%', height='45%', loc='lower right')
                if base is not None:
                    axins.imshow(img)
                else:
                    axins.imshow(img)
                    axins.imshow(amap, cmap=cmap, norm=norm, alpha=0.55, interpolation='nearest')
                axins.set_xlim(x0, x1); axins.set_ylim(y1, y0)
                axins.set_xticks([]); axins.set_yticks([])
                for spine in axins.spines.values():
                    spine.set_edgecolor('yellow'); spine.set_linewidth(1.5)
                mark_inset(ax_img[col], axins, loc1=2, loc2=4, fc='none', ec='yellow', lw=1.0)

        # ---- Scale bar (requires px_per_mm) ----
        if px_per_mm is not None:
            bar_mm = 1.0
            bar_px = bar_mm * px_per_mm
            x_start = W * 0.06
            y_pos = H * 0.92
            ax_img[0].plot([x_start, x_start + bar_px], [y_pos, y_pos], color='white', lw=3)
            ax_img[0].text(x_start + bar_px / 2, y_pos - H * 0.03, '%g mm' % bar_mm,
                           color='white', ha='center', va='bottom', fontsize=10)

        # ---- Shared colorbar (unified scale) ----
        cbar_ax = fig_img.add_axes([0.915, 0.15, 0.012, 0.7])
        cb = fig_img.colorbar(im, cax=cbar_ax)
        cb.set_label('Anomaly score', fontsize=11)
        cb.ax.tick_params(labelsize=9)

        fig_img.savefig(os.path.join(save_dir, '%s_%d.png' % (class_name, i)),
                        dpi=dpi, bbox_inches='tight')
        plt.close(fig_img)


def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x


def embedding_concat(x, y):
    """Vectorized version: removes the Python for loop and explicit zeros allocation, while keeping
    numerics identical to the original implementation."""
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = H1 // H2
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)        # (B, C1*s*s, H2*W2)
    x = x.view(B, C1, s * s, H2, W2)
    y = y.unsqueeze(2).expand(-1, -1, s * s, -1, -1)            # (B, C2, s*s, H2, W2)
    z = torch.cat([x, y], dim=1)                               # (B, C1+C2, s*s, H2, W2)
    z = z.view(B, (C1 + C2) * s * s, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)
    return z


def embedding_add(x, y):
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()

    assert C1 == C2, "Channel numbers of x and y must be the same for addition."

    if H1 == H2 and W1 == W2:
        return x + y

    if H1 > H2 or W1 > W2:
        y_resized = F.interpolate(y, size=(H1, W1), mode='nearest')
        return x + y_resized

    x_resized = F.interpolate(x, size=(H2, W2), mode='nearest')
    return x_resized + y


def calculate_f1_score(predictions, ground_truth):
    f1 = f1_score(predictions, ground_truth)
    return f1


def calc_pro(binary_maps: ndarray, gt_masks: ndarray):
    pros = []
    for binary_map, gt_mask in zip(binary_maps, gt_masks):
        for region in measure.regionprops(measure.label(gt_mask)):
            axes0_ids = region.coords[:, 0]
            axes1_ids = region.coords[:, 1]
            tp_pixels = binary_map[axes0_ids, axes1_ids].sum()
            pros.append(tp_pixels / region.area)
    return np.array(pros).mean()


def conv_custom(input_tensor, weights, bias=None):
    """Use the functional F.conv2d instead of creating a new nn.Conv2d each time
    (faster, no extra graph construction, no_grad friendly)."""
    kh, kw = weights.shape[2], weights.shape[3]
    if kh == 1 and kw == 1:
        padding = 0
    elif kh == 3 and kw == 3:
        padding = 1
    else:
        raise ValueError("Unsupported kernel size. Only 1x1 and 3x3 are supported.")
    return F.conv2d(input_tensor, weights, bias=bias, stride=1, padding=padding)


def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> float:
    """
    Compute the area under the curve of per-region overlapping (PRO) with FPR from 0 to 0.3.
    Optimization: each mask's connected components are labeled only once (no longer re-labeled per
    threshold); results are collected into a list and turned into a DataFrame in one pass.
    """
    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(np.unique(masks)).issubset({0, 1}), "masks must be binary {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    # Pre-compute the connected components of each mask (unchanged inside the threshold loop)
    region_list = [measure.regionprops(measure.label(m)) for m in masks]
    inverse_masks = 1 - masks
    inv_sum = inverse_masks.sum()

    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / num_th

    rows = []
    for th in np.arange(min_th, max_th, delta):
        binary_amaps = amaps > th

        pros = []
        for binary_amap, regions in zip(binary_amaps, region_list):
            for region in regions:
                coords = region.coords
                tp_pixels = binary_amap[coords[:, 0], coords[:, 1]].sum()
                pros.append(tp_pixels / region.area)

        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inv_sum
        rows.append({"pro": np.mean(pros) if pros else 0.0, "fpr": fpr, "threshold": th})

    df = pd.DataFrame(rows)
    df = df[df["fpr"] <= 0.3].copy()
    df["fpr"] = df["fpr"] / df["fpr"].max()
    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc


if __name__ == '__main__':
    main()
