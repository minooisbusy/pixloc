"""
train_tartanair.py — PixLoc (TwoViewRefiner) training on TartanAir.

Replaces the MegaDepth dataloader with TartanAir while keeping the PixLoc
model, loss functions, and optimizer completely unchanged.  Adds:

  1. Gradient accumulation (--accum_steps, default 2; effective batch = accum_steps × physical)
  2. Adaptive curriculum scheduling over frame stride

Base configuration is loaded from:
    /pixloc/pixloc/pixlib/configs/train_pixloc_cmu.yaml

TartanAir dataloader is loaded from:
    /fdo/dataloaders/tartanair_dataset.py

Usage (4 × GPU via torchrun — torchrun sets LOCAL_RANK / RANK / WORLD_SIZE):
    # With PixLoc's spawn-based DDP:
    python pixloc/train_tartanair.py pixloc_tartanair_baseline \\
        --tartanair_root /datasets/tartanair_zips \\
        --val_scenes abandonedfactory,amusement \\
        --distributed

    # Single-GPU debug:
    python pixloc/train_tartanair.py pixloc_tartanair_baseline \\
        --tartanair_root /datasets/tartanair_zips \\
        --val_scenes abandonedfactory,amusement

    # OmegaConf dot-list overrides:
    python pixloc/train_tartanair.py ...  train.epochs=300 train.lr=3e-6

Gradient accumulation details
------------------------------
  Physical batch (across 4 GPUs): 32 samples / step
    → 8 per GPU per forward pass
  --accum_steps 2 (default)
  Effective batch   : 64 samples per optimizer step
  Loss is divided by accum_steps before backward(); optimizer.step()
  is called every accum_steps steps or at epoch end, whichever comes first.
  Setup A (3× RTX PRO 5000, physical BS=64): run with --accum_steps 1.
  Setup B (4× RTX 3090,     physical BS=32): run with --accum_steps 2.

Coordinate conventions
-----------------------
  World frame = reference (A) camera frame.
    T_w2cam_ref   = I          (ref IS the world frame)
    T_w2cam_query = T_{A→B}    (maps world=refCam to query cam B)
  3D points (points3D) are unprojected from depth_a into ref-cam space.
  This is consistent with how PixLoc consumes SfM point clouds from MegaDepth.
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import re
import shutil
import signal
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# ── PixLoc internals ──────────────────────────────────────────────────────────
_PIXLOC_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PIXLOC_ROOT.parent))

from pixloc.pixlib.models import get_model
from pixloc.pixlib.geometry.wrappers import Camera as PixCamera, Pose as PixPose
from pixloc.pixlib.datasets.sampling import sample_pose_interval
from pixloc.pixlib.utils.tools import AverageMetric, MedianMetric, set_seed, fork_rng
from pixloc.pixlib.utils.tensor import batch_to_device
from pixloc.pixlib.utils.stdout_capturing import capture_outputs
from pixloc.pixlib.utils.experiments import (
    delete_old_checkpoints, get_last_checkpoint, get_best_checkpoint)
from pixloc.pixlib.train import (
    filter_parameters, pack_lr_parameters, default_train_conf)
from pixloc.settings import TRAINING_PATH
from pixloc import logger

# ── TartanAir dataloader ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path('/fdo/dataloaders')))
from tartanair_dataset import build_tartanair_dataset  # noqa: E402

# ── Config / constants ────────────────────────────────────────────────────────
_BASE_CONFIG = Path('/pixloc/pixloc/pixlib/configs/train_pixloc_cmu.yaml')

# 3D points sampled from depth_a per training pair.
# Matches CMU config: max_num_points3D=512 / force_num_points3D=true.
MAX_NUM_POINTS3D = 512

# Noisy initial pose: scale GT pose by U[0.0, 1.0].
# Matches FDO's criterion.init_pose_interval = (0.0, 1.0).
# Starting curriculum at max_stride=3 produces non-trivial baselines that
# already span large pose offsets, so the full [0, 1] range is appropriate.
INIT_POSE_INTERVAL = (0.0, 1.0)

# Native TartanAir resolution — matches FDO (--img_w 640 --img_h 480).
# No cropping, preserves the full field of view.
TARGET_SIZE = (640, 480)   # (W, H)

# Curriculum ceiling: dataset is built with this max stride so that
# DistributedSampler.total_size (= len(dataset) = N - _MAX_STRIDE) never
# changes when the curriculum advances, avoiding IndexError at epoch boundaries.
_MAX_CURRICULUM_STRIDE = 10


# ─────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

def _set_all_max_stride(dataset, max_stride: int) -> None:
    """Recursively call set_max_stride() on every SLAMDatasetBase leaf."""
    from torch.utils.data import ConcatDataset
    if isinstance(dataset, ConcatDataset):
        for ds in dataset.datasets:
            _set_all_max_stride(ds, max_stride)
    elif hasattr(dataset, 'set_max_stride'):
        dataset.set_max_stride(max_stride)


# ─────────────────────────────────────────────────────────────────────────────
# TartanAir → PixLoc format adapter
# ─────────────────────────────────────────────────────────────────────────────

def _build_pixloc_camera(K: torch.Tensor, W: int, H: int) -> PixCamera:
    """
    Build a batched PixLoc Camera from an (B, 3, 3) intrinsic matrix.

    PixLoc Camera data layout: [width, height, fx, fy, cx−0.5, cy−0.5].
    The −0.5 shift converts from the centre-of-pixel origin used in
    SLAMDatasetBase to the top-left corner origin used by PixLoc.
    """
    fx = K[..., 0, 0]
    fy = K[..., 1, 1]
    cx = K[..., 0, 2]
    cy = K[..., 1, 2]
    w_t = torch.full_like(fx, W)
    h_t = torch.full_like(fy, H)
    data = torch.stack([w_t, h_t, fx, fy, cx - 0.5, cy - 0.5], dim=-1)
    return PixCamera(data.float())


def _unproject_depth(
        depth: torch.Tensor,   # (B, 1, H, W) float32, metres; 0 = invalid
        K:     torch.Tensor,   # (B, 3, 3) float32
        max_points: int,
) -> torch.Tensor:
    """
    Back-project valid depth pixels into 3-D points in the camera frame.
    Returns (B, max_points, 3) float32.

    When a frame has fewer than max_points valid pixels the point set is
    tiled (with repetition) to fill the fixed-size output tensor.  This
    mirrors PixLoc's force_num_points3D=True behaviour for MegaDepth.
    """
    B, _, H, W = depth.shape
    device = depth.device

    ys, xs = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing='ij',
    )
    xs_f = xs.reshape(-1)   # (H*W,)
    ys_f = ys.reshape(-1)

    fx = K[:, 0, 0]   # (B,)
    fy = K[:, 1, 1]
    cx = K[:, 0, 2]
    cy = K[:, 1, 2]

    d     = depth.squeeze(1).reshape(B, -1)          # (B, H*W)
    valid = (d > 0.0) & torch.isfinite(d)            # (B, H*W)

    X = (xs_f[None] - cx[:, None]) * d / fx[:, None]
    Y = (ys_f[None] - cy[:, None]) * d / fy[:, None]
    Z = d
    pts_all = torch.stack([X, Y, Z], dim=-1)         # (B, H*W, 3)

    out = torch.zeros(B, max_points, 3, device=device, dtype=torch.float32)
    for b in range(B):
        pts_b = pts_all[b][valid[b]]                 # (N_valid, 3)
        n = pts_b.shape[0]
        if n == 0:
            continue
        if n >= max_points:
            idx = torch.randperm(n, device=device)[:max_points]
            out[b] = pts_b[idx]
        else:
            repeats = (max_points + n - 1) // n
            out[b] = pts_b.repeat(repeats, 1)[:max_points]
    return out


def tartanair_to_pixloc(
        batch: dict,
        device,
        seed:  int = 0,
        step:  int = 0,
) -> dict:
    """
    Convert a collated TartanAir batch (from SLAMDatasetBase) into the dict
    format consumed by PixLoc's TwoViewRefiner.

    TartanAir input keys (after batch_to_device):
        img_a, img_b      (B, C, H, W)  ImageNet-normalised float32
        depth_a           (B, 1, H, W)  metric depth of view A (metres)
        intrinsics        (B, 3, 3)     adjusted K for view A
                                        (= view B: same fixed TartanAir camera)
        pose_a_to_b       (B, 4, 4)    T_{A→B} relative cam-to-cam transform
        true_shape        (B, 1, 2)    [H, W] (unused here; H/W taken from img)

    PixLoc output keys:
        ref / image, camera, T_w2cam, points3D, index
        query / image, camera, T_w2cam, index
        T_r2q_gt    ground-truth relative pose  (PixPose, (B,))
        T_r2q_init  noisy initial pose           (PixPose, (B,))
        scene       list of dummy strings (required by TwoViewRefiner.loss)
    """
    img_a       = batch['img_a']          # (B, C, H, W)
    img_b       = batch['img_b']
    depth_a     = batch['depth_a']        # (B, 1, H, W)
    K           = batch['intrinsics']     # (B, 3, 3)
    pose_a_to_b = batch['pose_a_to_b']   # (B, 4, 4)

    B          = img_a.shape[0]
    _, _, H, W = img_a.shape

    # ── Camera objects ────────────────────────────────────────────────────────
    # TartanAir's fixed camera is identical for both views of every pair.
    cam = _build_pixloc_camera(K, W, H)   # PixCamera shape (B,)

    # ── World-frame poses ─────────────────────────────────────────────────────
    # We canonicalise: world frame = reference (A) camera frame.
    #   T_w2cam_ref   = I        — ref IS the world frame
    #   T_w2cam_query = T_{A→B} — maps world (=refCam) to query cam B
    # Therefore:
    #   T_r2q_gt = T_w2cam_query @ T_w2cam_ref⁻¹ = T_{A→B} @ I = T_{A→B}
    eye = (torch.eye(4, dtype=torch.float32, device=device)
           .unsqueeze(0).expand(B, -1, -1).contiguous())
    T_w2cam_ref   = PixPose.from_4x4mat(eye)
    T_w2cam_query = PixPose.from_4x4mat(pose_a_to_b)
    T_r2q_gt      = T_w2cam_query   # shape (B,)

    # ── Noisy initial pose ────────────────────────────────────────────────────
    # Mirrors the MegaDepth strategy (init_pose: [0.75, 1.0]):
    # scale GT rotation and translation by a random factor in that interval,
    # giving a close but imperfect starting point for the LM optimizer.
    T_init_list: list[PixPose] = []
    for b in range(B):
        T_gt_b   = PixPose.from_4x4mat(pose_a_to_b[b].cpu())
        T_init_b = sample_pose_interval(
            T_gt_b, INIT_POSE_INTERVAL, seed=seed + step * B + b)
        T_init_list.append(T_init_b)
    T_r2q_init = PixPose.stack(T_init_list, dim=0).to(device)  # shape (B,)

    # ── 3-D points in ref (A) camera frame ───────────────────────────────────
    points3D = _unproject_depth(depth_a, K, MAX_NUM_POINTS3D)  # (B, N, 3)

    return {
        'ref': {
            'image':    img_a,
            'camera':   cam,
            'T_w2cam':  T_w2cam_ref,
            'points3D': points3D,
            # 'index' and 'scene' are only used in TwoViewRefiner.loss warning
            'index':    torch.zeros(B, dtype=torch.long, device=device),
        },
        'query': {
            'image':    img_b,
            'camera':   cam,
            'T_w2cam':  T_w2cam_query,
            'index':    torch.zeros(B, dtype=torch.long, device=device),
        },
        'T_r2q_gt':   T_r2q_gt,
        'T_r2q_init': T_r2q_init,
        'scene':      ['tartanair'] * B,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helper
# ─────────────────────────────────────────────────────────────────────────────

def do_evaluation(model, loader, device, loss_fn, metrics_fn, conf,
                  pbar=True, seed=0, step_offset=0):
    """Run a full pass over loader, return averaged metrics dict."""
    model.eval()
    results: dict[str, AverageMetric | MedianMetric] = {}
    for i, raw_batch in enumerate(
            tqdm(loader, desc='Evaluation', ascii=True, disable=not pbar)):
        raw_batch = batch_to_device(raw_batch, device, non_blocking=True)
        data = tartanair_to_pixloc(
            raw_batch, device, seed=seed, step=step_offset + i)
        with torch.no_grad():
            pred    = model(data)
            losses  = loss_fn(pred, data)
            metrics = metrics_fn(pred, data)
            del pred, data
        numbers = {**metrics, **{'loss/' + k: v for k, v in losses.items()}}
        for k, v in numbers.items():
            if k not in results:
                results[k] = AverageMetric()
                if k in conf.median_metrics:
                    results[k + '_median'] = MedianMetric()
            results[k].update(v)
            if k in conf.median_metrics:
                results[k + '_median'].update(v)
    return {k: results[k].compute() for k in results}


# ─────────────────────────────────────────────────────────────────────────────
# Core training function
# ─────────────────────────────────────────────────────────────────────────────

def training(rank: int, conf, output_dir: Path, args: argparse.Namespace):

    # ── Numerical precision — pin before any CUDA op ──────────────────────────
    # TF32 for convolutions (Ampere+ hardware) — same across all setups.
    torch.backends.cudnn.allow_tf32 = True
    # Full FP32 for matmul (LM Hessian inversion) — 'highest' disables TF32/BF16
    # for matmul, ensuring the LM solve is numerically identical across GPU gens.
    torch.set_float32_matmul_precision('highest')

    # ── Restore or fresh start ────────────────────────────────────────────────
    if args.restore:
        logger.info(f'Restoring from previous training of {args.experiment}')
        init_cp = get_last_checkpoint(args.experiment, allow_interrupted=False)
        logger.info(f'Restoring checkpoint {init_cp.name}')
        init_cp = torch.load(str(init_cp), map_location='cpu')
        conf    = OmegaConf.merge(OmegaConf.create(init_cp['conf']), conf)
        epoch   = init_cp['epoch'] + 1

        best_cp   = get_best_checkpoint(args.experiment)
        best_cp   = torch.load(str(best_cp), map_location='cpu')
        best_eval = best_cp['eval'][conf.train.best_key]
        del best_cp

        # Restore curriculum state saved in the checkpoint
        curriculum_state   = init_cp.get('curriculum', {})
        current_max_stride = curriculum_state.get(
            'current_max_stride', args.init_stride)
        epochs_at_stride   = curriculum_state.get('epochs_at_stride', 0)
    else:
        conf.train = OmegaConf.merge(default_train_conf, conf.train)
        # TartanAir has no epoch-level resampling callback
        OmegaConf.update(conf, 'train.dataset_callback_fn', None)
        # Change coarsest pyramid level from stride 16 → stride 8 to match FDO.
        # output_scales [0,2,3] → self.scales = [2^0, 2^2, 2^3] = [1, 4, 8].
        # Must be set before OmegaConf.set_struct locks the config.
        OmegaConf.update(conf, 'model.extractor.output_scales', [0, 2, 3])
        # Freeze BN: eliminates sensitivity to per-step sample count so that
        # accum_steps=1 (Setup A, BS=64) and accum_steps=2 (Setup B, BS=32)
        # produce identical BN statistics.  Running stats remain ImageNet-pretrained.
        OmegaConf.update(conf, 'model.extractor.freeze_batch_normalization', True)
        epoch              = 0
        best_eval          = float('inf')
        current_max_stride = args.init_stride   # curriculum begins here
        epochs_at_stride   = 0

        if conf.train.load_experiment:
            logger.info(f'Fine-tuning from {conf.train.load_experiment}')
            init_cp = get_last_checkpoint(conf.train.load_experiment)
            init_cp = torch.load(str(init_cp), map_location='cpu')
        else:
            init_cp = None

    OmegaConf.set_struct(conf, True)   # guard against typos in dotlist
    set_seed(conf.train.seed)

    main = (rank == 0)
    if main:
        writer = SummaryWriter(log_dir=str(output_dir))

    # ── Distributed setup ─────────────────────────────────────────────────────
    if args.distributed:
        logger.info(f'Training in distributed mode with {args.n_gpus} GPUs')
        assert torch.cuda.is_available()
        device = rank
        torch.distributed.init_process_group(
            backend='nccl', world_size=args.n_gpus, rank=device,
            init_method='env://')
        torch.cuda.set_device(device)

        # Divide batch size and workers evenly across GPUs
        batch_size_per_gpu  = max(1, args.batch_size // args.n_gpus)
        num_workers_per_gpu = max(1,
            (args.num_workers + args.n_gpus - 1) // args.n_gpus)
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        batch_size_per_gpu  = args.batch_size
        num_workers_per_gpu = args.num_workers

    logger.info(
        f'device={device}  batch_size_per_gpu={batch_size_per_gpu}  '
        f'accumulation_steps={args.accum_steps}  '
        f'effective_batch={batch_size_per_gpu * args.n_gpus * args.accum_steps}')

    # ── Build TartanAir datasets ──────────────────────────────────────────────
    val_scenes   = [s.strip() for s in args.val_scenes.split(',')]
    difficulties = (
        [s.strip() for s in args.difficulties.split(',')]
        if args.difficulties else None
    )

    # Train dataset is built with max stride = _MAX_CURRICULUM_STRIDE so that
    # valid_length (= N − max_stride) is fixed for the lifetime of the
    # DistributedSampler.  The curriculum dynamically restricts what stride
    # is *used* in __getitem__ without changing the index universe.
    train_ds = build_tartanair_dataset(
        root           = args.tartanair_root,
        exclude_scenes = val_scenes,
        difficulties   = difficulties,
        min_stride     = args.min_stride,
        max_stride     = _MAX_CURRICULUM_STRIDE,
        target_size    = TARGET_SIZE,
        bidirectional  = True,
        norm           = 'imagenet',
    )
    val_ds = build_tartanair_dataset(
        root          = args.tartanair_root,
        scenes        = val_scenes,
        difficulties  = difficulties,
        min_stride    = 1,
        max_stride    = 1,   # deterministic, stride-1 validation pairs
        target_size   = TARGET_SIZE,
        bidirectional = False,
        norm          = 'imagenet',
    )

    # Apply curriculum-restored stride (fresh runs start at min_stride)
    _set_all_max_stride(train_ds, current_max_stride)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    if args.distributed:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=args.n_gpus, rank=rank, shuffle=True)
        train_loader  = DataLoader(
            train_ds,
            batch_size=batch_size_per_gpu,
            sampler=train_sampler,
            num_workers=num_workers_per_gpu,
            pin_memory=True,
            drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size_per_gpu,
            shuffle=True,
            num_workers=num_workers_per_gpu,
            pin_memory=True,
            drop_last=True,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size_per_gpu,
        shuffle=False,
        num_workers=num_workers_per_gpu,
        pin_memory=True,
        drop_last=False,
    )

    if main:
        logger.info(
            f'Train: {len(train_loader)} batches  |  '
            f'Val: {len(val_loader)} batches')

    # ── SIGINT handler ────────────────────────────────────────────────────────
    stop = False

    def sigint_handler(sig, frame):
        logger.info('Caught SIGINT — will terminate after current iteration.')
        nonlocal stop
        if stop:
            raise KeyboardInterrupt
        stop = True

    signal.signal(signal.SIGINT, sigint_handler)

    # ── Model ─────────────────────────────────────────────────────────────────
    model      = get_model(conf.model.name)(conf.model).to(device)
    loss_fn    = model.loss
    metrics_fn = model.metrics

    if init_cp is not None:
        model.load_state_dict(init_cp['model'])
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device])
    if main:
        logger.info(f'Model:\n{model}')
    torch.backends.cudnn.benchmark = True

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer_fn = {
        'sgd':     torch.optim.SGD,
        'adam':    torch.optim.Adam,
        'rmsprop': torch.optim.RMSprop,
    }[conf.train.optimizer]

    params     = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if conf.train.opt_regexp:
        params = filter_parameters(params, conf.train.opt_regexp)
    all_params = [p for _, p in params]
    lr_params  = pack_lr_parameters(params, conf.train.lr, conf.train.lr_scaling)
    optimizer  = optimizer_fn(
        lr_params, lr=conf.train.lr, **conf.train.optimizer_options)

    def lr_fn(it):
        if conf.train.lr_schedule.type is None:
            return 1
        if conf.train.lr_schedule.type == 'exp':
            gam = 10 ** (-1 / conf.train.lr_schedule.exp_div_10)
            return 1 if it < conf.train.lr_schedule.start else gam
        raise ValueError(conf.train.lr_schedule.type)

    lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_fn)

    if args.restore:
        optimizer.load_state_dict(init_cp['optimizer'])
        if 'lr_scheduler' in init_cp:
            lr_scheduler.load_state_dict(init_cp['lr_scheduler'])

    if main:
        logger.info('Configuration:\n%s', OmegaConf.to_yaml(conf))

    # ── Training loop ─────────────────────────────────────────────────────────
    results: dict = {}   # latest validation results for checkpointing

    while epoch < conf.train.epochs and not stop:
        if main:
            logger.info(f'Starting epoch {epoch}')
        set_seed(conf.train.seed + epoch)
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        # Per-epoch curriculum accumulators (reset each epoch)
        epoch_p0_sum = 0.0
        epoch_p0_n   = 0

        # Zero gradients once at epoch start; zero again after each optimizer step
        optimizer.zero_grad()

        for it, raw_batch in enumerate(train_loader):
            tot_it     = len(train_loader) * epoch + it
            is_last_it = (it == len(train_loader) - 1)

            model.train()

            # ── 1. Move raw TartanAir tensors to device ───────────────────────
            raw_batch = batch_to_device(raw_batch, device, non_blocking=True)

            # ── 2. Convert to PixLoc format (Camera / Pose built on device) ───
            data = tartanair_to_pixloc(
                raw_batch, device,
                seed=conf.train.seed,
                step=tot_it,
            )

            # ── 3. Forward pass through TwoViewRefiner ────────────────────────
            pred   = model(data)
            losses = loss_fn(pred, data)

            # ── 4. Accumulate curriculum statistics ───────────────────────────
            # Matches FDO's reproj_px/p0: mean reprojection error in RAW PIXELS
            # at the finest pyramid level AFTER LM optimization (T_opt).
            #
            # pred['T_r2q_opt'] is ordered coarse→fine (reversed loop in
            # TwoViewRefiner._forward), so [-1] is the finest-level T_opt.
            #
            # world2image returns (p2D, visible) where visible encodes both
            # depth > 0 AND within-image-bounds — using this directly is more
            # correct than a separate in_image() call that would miss the depth
            # check.  Both GT and optimised projections must be visible.
            with torch.no_grad():
                T_final = pred['T_r2q_opt'][-1]                    # finest T_opt
                cam_q   = data['query']['camera']
                p3D     = data['ref']['points3D']                   # (B, N, 3)

                p2D_gt,  vis_gt  = cam_q.world2image(data['T_r2q_gt'] * p3D)
                p2D_opt, vis_opt = cam_q.world2image(T_final        * p3D)

                px_err = torch.sqrt(
                    torch.sum((p2D_gt - p2D_opt) ** 2, dim=-1))    # (B, N) px
                valid  = vis_gt & vis_opt & px_err.isfinite()       # (B, N) bool

                if valid.any():
                    epoch_p0_sum += px_err[valid].sum().item()
                    epoch_p0_n   += int(valid.sum().item())

            # ── 5. Scale loss for gradient accumulation ───────────────────────
            # The total gradient over args.accum_steps backward() calls must
            # equal what a single step with the full effective batch would give.
            loss = torch.mean(losses['total']) / args.accum_steps

            # ── 6. DDP-aware backward guard ───────────────────────────────────
            do_backward = loss.requires_grad
            if args.distributed:
                dbt = torch.tensor(float(do_backward)).to(device)
                torch.distributed.all_reduce(
                    dbt, torch.distributed.ReduceOp.PRODUCT)
                do_backward = dbt.item() > 0

            if do_backward:
                loss.backward()
            elif main:
                logger.warning(f'Skip iteration {it} due to detach.')

            # ── 7. Optimizer step every args.accum_steps or at epoch end ────
            should_step = (
                ((it + 1) % args.accum_steps == 0) or is_last_it
            )
            if do_backward and should_step:
                if conf.train.get('clip_grad', None):
                    # Log the fraction of parameters that exceed the clip value
                    if it % conf.train.log_every_iter == 0:
                        grads = [
                            p.grad.data.abs().reshape(-1)
                            for p in all_params if p.grad is not None
                        ]
                        if grads:
                            ratio = (
                                (torch.cat(grads, 0) > conf.train.clip_grad)
                                .float().mean().item() * 100
                            )
                            if ratio > 25:
                                logger.warning(
                                    f'More than {ratio:.1f}% of the parameters'
                                    ' are larger than the clip value.')
                            del grads, ratio
                    torch.nn.utils.clip_grad_value_(
                        all_params, conf.train.clip_grad)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # ── 8. Periodic loss logging ──────────────────────────────────────
            if it % conf.train.log_every_iter == 0:
                for k in sorted(losses.keys()):
                    if args.distributed:
                        losses[k] = losses[k].sum()
                        torch.distributed.reduce(losses[k], dst=0)
                        losses[k] /= (train_loader.batch_size * args.n_gpus)
                    losses[k] = torch.mean(losses[k]).item()
                if main:
                    str_losses = [f'{k} {v:.3E}' for k, v in losses.items()]
                    logger.info('[E {} | it {}] loss {{{}}}'.format(
                        epoch, it, ', '.join(str_losses)))
                    for k, v in losses.items():
                        writer.add_scalar('training/' + k, v, tot_it)
                    writer.add_scalar(
                        'training/lr',
                        optimizer.param_groups[0]['lr'],
                        tot_it)

            del pred, data, loss, losses

            # ── 9. Periodic validation ────────────────────────────────────────
            if (it % conf.train.eval_every_iter == 0) or stop or is_last_it:
                with fork_rng(seed=conf.train.seed):
                    results = do_evaluation(
                        model, val_loader, device,
                        loss_fn, metrics_fn, conf.train,
                        pbar=main,
                        seed=conf.train.seed,
                        step_offset=tot_it,
                    )
                if main:
                    str_r = [f'{k} {v:.3E}' for k, v in results.items()]
                    logger.info(f'[Validation] {{{", ".join(str_r)}}}')
                    for k, v in results.items():
                        writer.add_scalar('val/' + k, v, tot_it)
                torch.cuda.empty_cache()

            if stop:
                break

        # ── Checkpoint (rank 0 only) ──────────────────────────────────────────
        if main:
            state = (
                model.module if args.distributed else model
            ).state_dict()
            checkpoint = {
                'model':        state,
                'optimizer':    optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'conf':         OmegaConf.to_container(conf, resolve=True),
                'epoch':        epoch,
                'losses':       None,   # kept for checkpoint format compatibility
                'eval':         results,
                # Curriculum state — restored on --restore
                'curriculum': {
                    'current_max_stride': current_max_stride,
                    'epochs_at_stride':   epochs_at_stride,
                },
            }
            cp_name = (
                f'checkpoint_{epoch}' + ('_interrupted' if stop else ''))
            logger.info(f'Saving checkpoint {cp_name}')
            cp_path = str(output_dir / (cp_name + '.tar'))
            torch.save(checkpoint, cp_path)
            if results and results.get(conf.train.best_key, float('inf')) < best_eval:
                best_eval = results[conf.train.best_key]
                logger.info(
                    f'New best: {conf.train.best_key}={best_eval:.4f}')
                shutil.copy(cp_path, str(output_dir / 'checkpoint_best.tar'))
            delete_old_checkpoints(output_dir, conf.train.keep_last_checkpoints)
            del checkpoint

        # ── Adaptive curriculum advancement ───────────────────────────────────
        if main and epoch_p0_n > 0:
            epoch_mean_p0 = epoch_p0_sum / epoch_p0_n
            thresh = current_max_stride * 0.1
            epochs_at_stride += 1
            converged = epoch_mean_p0 < thresh
            forced    = epochs_at_stride >= 3
            if current_max_stride < 10 and (converged or forced):
                reason = 'converged' if converged else 'forced'
                current_max_stride += 1
                epochs_at_stride    = 0
                logging.info(
                    'Curriculum advanced -> max_stride=%d  (reason=%s, epoch_mean_p0=%.4f px, thresh=%.4f px)',
                    current_max_stride, reason, epoch_mean_p0, thresh)
            else:
                logging.info(
                    'Curriculum hold    -> max_stride=%d  epoch_mean_p0=%.4f px  thresh=%.4f px  epochs_at_stride=%d',
                    current_max_stride, epoch_mean_p0, thresh, epochs_at_stride)

        # Broadcast updated curriculum state from rank 0 to all DDP ranks so
        # every worker applies the same stride on the next epoch.
        if args.distributed:
            stride_t = torch.tensor(
                [current_max_stride, epochs_at_stride],
                dtype=torch.int64, device=device)
            torch.distributed.broadcast(stride_t, src=0)
            current_max_stride = int(stride_t[0].item())
            epochs_at_stride   = int(stride_t[1].item())

        # Apply stride update to dataset leaves
        _set_all_max_stride(train_ds, current_max_stride)

        epoch += 1

    logger.info(f'Finished training on process {rank}.')
    if main:
        writer.close()


# ─────────────────────────────────────────────────────────────────────────────
# Process entry points
# ─────────────────────────────────────────────────────────────────────────────

def main_worker(rank: int, conf, output_dir: Path, args: argparse.Namespace):
    if rank == 0:
        with capture_outputs(output_dir / 'log.txt'):
            training(rank, conf, output_dir, args)
    else:
        training(rank, conf, output_dir, args)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train PixLoc on TartanAir with gradient accumulation '
                    'and adaptive curriculum scheduling.')

    parser.add_argument(
        'experiment', type=str,
        help='Experiment name (sub-directory created under TRAINING_PATH)')
    parser.add_argument(
        '--tartanair_root', type=str, required=True,
        help='Root directory of the extracted TartanAir dataset')
    parser.add_argument(
        '--val_scenes', type=str, default='abandonedfactory,amusement',
        help='Comma-separated TartanAir scene names held out for validation')
    parser.add_argument(
        '--difficulties', type=str, default=None,
        help='Comma-separated difficulty levels to include (e.g. Easy,Hard)')
    parser.add_argument(
        '--min_stride', type=int, default=1,
        help='Minimum frame stride within each pair (dataset lower bound, default 1)')
    parser.add_argument(
        '--init_stride', type=int, default=3,
        help='Max stride at which the curriculum begins (default 3, matches FDO start)')
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='Total physical batch size across all GPUs per forward step '
             '(default 32 → 8 per GPU with 4 GPUs)')
    parser.add_argument(
        '--accum_steps', type=int, default=2,
        help='Gradient accumulation steps (default 2). '
             'Setup A (BS=64, 3× PRO 5000): use 1. '
             'Setup B (BS=32, 4× 3090): use 2.')
    parser.add_argument(
        '--num_workers', type=int, default=8,
        help='Total dataloader worker count (divided evenly across GPUs)')
    parser.add_argument(
        '--conf', type=str, default=None,
        help='Optional extra YAML config merged on top of the base config')
    parser.add_argument(
        '--restore', action='store_true',
        help='Resume from the latest checkpoint of this experiment')
    parser.add_argument(
        '--distributed', action='store_true',
        help='Enable DistributedDataParallel (multi-GPU) training')
    parser.add_argument(
        'dotlist', nargs='*',
        help='OmegaConf dot-list overrides, e.g. train.epochs=300 train.lr=3e-6')

    args = parser.parse_args()

    logger.info(f'Starting experiment {args.experiment}')
    output_dir = Path(TRAINING_PATH, args.experiment)
    output_dir.mkdir(exist_ok=True, parents=True)

    # ── Build merged configuration ────────────────────────────────────────────
    conf = OmegaConf.load(str(_BASE_CONFIG))
    if args.conf:
        conf = OmegaConf.merge(conf, OmegaConf.load(args.conf))
    if args.dotlist:
        conf = OmegaConf.merge(conf, OmegaConf.from_cli(args.dotlist))

    if not args.restore:
        seed = conf.train.get('seed')
        if seed is None or str(seed) == '???':
            OmegaConf.update(
                conf, 'train.seed', torch.initial_seed() & (2 ** 32 - 1))
        OmegaConf.save(conf, str(output_dir / 'config.yaml'))

    # ── Launch ────────────────────────────────────────────────────────────────
    if args.distributed:
        args.n_gpus = torch.cuda.device_count()
        torch.multiprocessing.spawn(
            main_worker,
            nprocs=args.n_gpus,
            args=(conf, output_dir, args))
    else:
        args.n_gpus = 1
        main_worker(0, conf, output_dir, args)
