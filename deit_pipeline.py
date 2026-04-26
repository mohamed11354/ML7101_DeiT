from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Sequence

import torch
import torch.distributed as dist
from timm.layers import resample_abs_pos_embed
from torch import nn
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe
from torch.utils.data import DataLoader

from deit_common import (
    EpochMetrics,
    build_datasets,
    build_model,
    build_transforms,
    compute_topk,
    create_optimizer,
    create_scheduler,
    create_train_criterion,
    current_lr,
    format_gpu_utilization,
    format_run_summary,
    move_optimizer_state_to_device,
    resolve_model_output,
    save_checkpoint,
    save_json,
    set_seed,
    setup_cudnn,
    write_metrics_row,
)

import threading

try:
    import pynvml
except Exception:
    pynvml = None


class GpuUtilizationMonitor:
    """
    Low-overhead GPU utilization monitor.

    - Uses NVML directly (no nvidia-smi subprocesses).
    - Samples in a background thread.
    - Starts lazily on the first batch, so DataLoader workers are already alive.
    - Returns per-device epoch-average utilization as:
        [("cuda:0", 81.2), ("cuda:1", 77.5)]
    """

    def __init__(
        self,
        devices: Sequence[torch.device],
        sample_interval_s: float = 1.0,
    ) -> None:
        self.devices = [torch.device(d) for d in devices if torch.device(d).type == "cuda"]
        self.sample_interval_s = max(0.25, float(sample_interval_s))

        self._enabled = bool(self.devices) and pynvml is not None
        self._started = False
        self._closed = False

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

        self._handles = []
        self._names: list[str] = []
        self._sum: dict[str, float] = {}
        self._count: dict[str, int] = {}

        # Handle CUDA_VISIBLE_DEVICES masks/reordering when possible.
        self._visible = [
            token.strip()
            for token in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if token.strip()
        ]

    def _handle_for(self, device: torch.device):
        idx = 0 if device.index is None else int(device.index)

        if idx < len(self._visible):
            token = self._visible[idx]

            if token.isdigit():
                return pynvml.nvmlDeviceGetHandleByIndex(int(token))

            for candidate in (token, token.encode("utf-8")):
                try:
                    return pynvml.nvmlDeviceGetHandleByUUID(candidate)
                except Exception:
                    pass

        return pynvml.nvmlDeviceGetHandleByIndex(idx)

    def _sample_once(self) -> None:
        if not self._started:
            return

        with self._lock:
            for name, handle in zip(self._names, self._handles):
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    self._sum[name] += float(util.gpu)
                    self._count[name] += 1
                except Exception:
                    pass

    def _worker(self) -> None:
        next_t = time.perf_counter() + self.sample_interval_s
        while not self._stop.wait(max(0.0, next_t - time.perf_counter())):
            self._sample_once()
            next_t += self.sample_interval_s

    def _start(self) -> None:
        if not self._enabled or self._started or self._closed:
            return

        try:
            pynvml.nvmlInit()
            self._handles = [self._handle_for(device) for device in self.devices]
            self._names = [str(device) for device in self.devices]
            self._sum = {name: 0.0 for name in self._names}
            self._count = {name: 0 for name in self._names}

            self._started = True
            self._sample_once()  # immediate first sample
            self._thread = threading.Thread(
                target=self._worker,
                name="gpu-util-monitor",
                daemon=True,
            )
            self._thread.start()
        except Exception:
            self._enabled = False
            self._started = False
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def maybe_sample(self, force: bool = False) -> None:
        # Keep the hot path free: only react on forced calls.
        if not force or self._closed or not self._enabled:
            return

        if not self._started:
            # Start on the first batch, not before the loader loop.
            self._start()
        else:
            # Optional trailing sample at epoch end.
            self._sample_once()

    def as_device_utilization(self) -> list[tuple[str, float]]:
        if not self._closed:
            self._closed = True
            self._stop.set()

            if self._thread is not None:
                self._thread.join(timeout=self.sample_interval_s + 0.1)
                self._thread = None

            if self._started:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass

        names = self._names or [str(device) for device in self.devices]
        return [
            (
                name,
                (self._sum[name] / self._count[name])
                if self._count.get(name, 0) > 0
                else float("nan"),
            )
            for name in names
        ]


class DeiTStemStage(nn.Module):
    def __init__(self, model: nn.Module, blocks: Sequence[nn.Module]) -> None:
        super().__init__()
        self.patch_embed = model.patch_embed
        self.cls_token = getattr(model, "cls_token", None)
        self.reg_token = getattr(model, "reg_token", None)
        self.pos_embed = getattr(model, "pos_embed", None)
        self.pos_drop = model.pos_drop
        self.dynamic_img_size = bool(getattr(model, "dynamic_img_size", False))
        self.no_embed_class = bool(getattr(model, "no_embed_class", False))
        self.num_prefix_tokens = int(getattr(model, "num_prefix_tokens", 1))
        self.patch_drop = model.patch_drop
        self.norm_pre = model.norm_pre
        self.blocks = nn.Sequential(*blocks)

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

        if self.pos_embed is None:
            return torch.cat(to_cat + [x.view(x.shape[0], -1, x.shape[-1])], dim=1)

        if self.dynamic_img_size:
            batch_size, height, width, channels = x.shape
            pos_embed = resample_abs_pos_embed(
                self.pos_embed,
                new_size=(height, width),
                old_size=self.patch_embed.grid_size,
                num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
            )
            x = x.view(batch_size, -1, channels)
        else:
            pos_embed = self.pos_embed

        if self.no_embed_class:
            x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        else:
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            x = x + pos_embed

        return self.pos_drop(x)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(images)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        return self.blocks(x)


class DeiTBlocksStage(nn.Module):
    def __init__(self, blocks: Sequence[nn.Module]) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class DeiTHeadStage(nn.Module):
    def __init__(self, model: nn.Module, blocks: Sequence[nn.Module]) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*blocks)
        self.norm = model.norm
        self.fc_norm = model.fc_norm
        self.head_drop = model.head_drop
        self.head = model.head
        # Store pooling config rather than the bound method (model.pool)
        # because the method captures a reference to the original model.
        self.global_pool = model.global_pool
        self.num_prefix_tokens = int(getattr(model, "num_prefix_tokens", 1))

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        if self.global_pool == "token":
            return x[:, 0]
        x = x[:, self.num_prefix_tokens :]
        if self.global_pool == "avg":
            return x.mean(dim=1)
        elif self.global_pool == "max":
            return x.amax(dim=1)
        elif self.global_pool == "avgmax":
            return 0.5 * (x.amax(dim=1) + x.mean(dim=1))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        x = self.norm(x)
        x = self._pool(x)
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return self.head(x)


def split_block_ranges(num_blocks: int, num_stages: int) -> list[tuple[int, int]]:
    base = num_blocks // num_stages
    remainder = num_blocks % num_stages
    ranges: list[tuple[int, int]] = []
    start = 0
    for stage_idx in range(num_stages):
        count = base + (1 if stage_idx < remainder else 0)
        end = start + count
        ranges.append((start, end))
        start = end
    return ranges


def build_pipeline_stage_module(model: nn.Module, stage_index: int, num_stages: int) -> nn.Module:
    blocks = list(model.blocks.children())
    block_start, block_end = split_block_ranges(len(blocks), num_stages)[stage_index]
    block_slice = blocks[block_start:block_end]

    if stage_index == 0:
        return DeiTStemStage(model, block_slice)
    if stage_index == num_stages - 1:
        return DeiTHeadStage(model, block_slice)
    return DeiTBlocksStage(block_slice)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pipeline-parallel DeiT training script.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar100",
        choices=["food101", "cifar100", "flowers102", "imagefolder"],
    )
    parser.add_argument("--data-root", type=str, default="/l/users/mohamed.awad/parallel_project/datasets")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--model", type=str, default="deit_small_patch16_224.fb_in1k")

    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128, help="Global pipeline batch size.")
    parser.add_argument("--microbatches", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--opt-betas", type=float, nargs=2, default=None)
    parser.add_argument("--opt-eps", type=float, default=1e-8)
    parser.add_argument("--warmup-epochs", type=int, default=15)
    parser.add_argument("--warmup-lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--drop-rate", type=float, default=0.0)
    parser.add_argument("--drop-path", type=float, default=0.1)

    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--auto-augment", type=str, default="rand-m9-mstd0.5-inc1")
    parser.add_argument("--re-prob", type=float, default=0.25)
    parser.add_argument("--re-mode", type=str, default="pixel")
    parser.add_argument("--re-count", type=int, default=1)
    parser.add_argument("--color-jitter", type=float, default=0.3)

    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--train-interpolation", type=str, default="bicubic")
    parser.add_argument("--mean", type=float, nargs=3, default=[0.5071, 0.4867, 0.4408])
    parser.add_argument("--std", type=float, nargs=3, default=[0.2675, 0.2565, 0.2761])
    parser.add_argument("--crop-scale", type=float, nargs=2, default=[0.8, 1.0])
    parser.add_argument("--crop-ratio", type=float, nargs=2, default=[0.75, 4 / 3])
    parser.add_argument("--crop-pct", type=float, default=1.0)

    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-top1", type=float, default=None)
    parser.add_argument("--resume-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="./outputs_pipe")

    parser.add_argument("--dist-backend", type=str, default="nccl")
    parser.add_argument("--dist-url", type=str, default="env://")
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def setup_pipeline(args: argparse.Namespace) -> tuple[int, int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("Pipeline training requires CUDA devices.")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 2:
        raise RuntimeError(
            "WORLD_SIZE < 2. Launch with: torchrun --standalone --nproc_per_node=2 deit_pipeline.py ..."
        )
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url)
    return rank, world_size, local_rank, torch.device(f"cuda:{local_rank}")


def cleanup_pipeline() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_first_stage(rank: int) -> bool:
    return rank == 0


def is_last_stage(rank: int, world_size: int) -> bool:
    return rank == world_size - 1


def make_loader(
    dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    *,
    shuffle: bool,
    seed: int,
    drop_last: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,
        drop_last=drop_last,
    )


def maybe_download_dataset_once(args: argparse.Namespace, train_transform, eval_transform, rank: int) -> None:
    if not args.download:
        return
    if rank == 0:
        build_datasets(
            dataset_name=args.dataset,
            data_root=args.data_root,
            train_transform=train_transform,
            eval_transform=eval_transform,
            download=True,
        )
    dist.barrier()


def clip_global_pipeline_grad_norm_(
    module: nn.Module,
    max_norm: float,
    device: torch.device,
    eps: float = 1e-6,
) -> torch.Tensor:
    local_sq = torch.zeros((), device=device, dtype=torch.float32)
    for param in module.parameters():
        if param.grad is not None:
            local_sq += param.grad.detach().float().pow(2).sum()

    dist.all_reduce(local_sq, op=dist.ReduceOp.SUM)
    global_norm = local_sq.sqrt()
    clip_coef = torch.clamp(
        torch.tensor(float(max_norm), device=device) / (global_norm + eps),
        max=1.0,
    )

    for param in module.parameters():
        if param.grad is not None:
            param.grad.mul_(clip_coef.to(dtype=param.grad.dtype))

    return global_norm


def validate_loader_lengths(train_loader: DataLoader, val_loader: DataLoader, device: torch.device) -> None:
    local_len = torch.tensor([len(train_loader), len(val_loader)], device=device, dtype=torch.int64)
    min_len = local_len.clone()
    max_len = local_len.clone()
    dist.all_reduce(min_len, op=dist.ReduceOp.MIN)
    dist.all_reduce(max_len, op=dist.ReduceOp.MAX)
    if not torch.equal(min_len, max_len):
        raise RuntimeError(
            f"Loader length mismatch across ranks: local={local_len.tolist()}, "
            f"min={min_len.tolist()}, max={max_len.tolist()}"
        )


def load_checkpoint_if_requested(
    *,
    stage_module: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    resume_dir: str | None,
    rank: int,
    world_size: int,
    device: torch.device,
) -> tuple[int, float, int]:
    if not resume_dir:
        return 1, -1.0, -1

    checkpoint_path = Path(resume_dir) / f"stage{rank:02d}_last.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if int(checkpoint.get("stage_index", rank)) != rank:
        raise RuntimeError(f"Checkpoint {checkpoint_path} is for a different stage.")
    if int(checkpoint.get("world_size", world_size)) != world_size:
        raise RuntimeError(f"Checkpoint {checkpoint_path} was saved with a different world_size.")
    stage_module.load_state_dict(checkpoint["model"])
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        move_optimizer_state_to_device(optimizer, device)
    if "scheduler" in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint["scheduler"])
        except Exception as exc:
            raise RuntimeError(f"Failed to load scheduler state from {checkpoint_path}") from exc
    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_top1 = float(checkpoint.get("best_val_top1", -1.0))
    best_epoch = int(checkpoint.get("best_epoch", start_epoch - 1))
    return start_epoch, best_top1, best_epoch


def run_pipeline_step(
    schedule: ScheduleGPipe,
    images: torch.Tensor,
    target: torch.Tensor,
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    losses: list[torch.Tensor] | None = None,
    eval_mode: bool = False,
):
    step = schedule.eval if eval_mode else schedule.step
    if is_first_stage(rank):
        return step(images.to(device, non_blocking=True), losses=losses)
    if is_last_stage(rank, world_size):
        return step(target=target.to(device, non_blocking=True), losses=losses)
    return step(losses=losses)


def train_one_epoch_pipeline(
    *,
    schedule: ScheduleGPipe,
    stage_module: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_clip: float | None,
) -> tuple[float, int, float, float, list[tuple[str, float]]]:
    stage_module.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    gpu_monitor = GpuUtilizationMonitor([device])
    loss_sum = 0.0
    loss_count = 0
    num_samples = 0
    start_time = time.perf_counter()

    for step, (images, target) in enumerate(loader):
        optimizer.zero_grad(set_to_none=True)
        losses: list[torch.Tensor] | None = [] if is_last_stage(rank, world_size) else None
        run_pipeline_step(
            schedule,
            images,
            target,
            rank=rank,
            world_size=world_size,
            device=device,
            losses=losses,
            eval_mode=False,
        )
        if grad_clip is not None:
            clip_global_pipeline_grad_norm_(stage_module, float(grad_clip), device)
        optimizer.step()

        batch_size = images.size(0)
        num_samples += batch_size
        if losses is not None and losses:
            loss_sum += float(torch.stack([loss.detach() for loss in losses]).mean().item()) * batch_size
            loss_count += batch_size
        gpu_monitor.maybe_sample(force=(step == 0))

    gpu_monitor.maybe_sample(force=True)
    elapsed = time.perf_counter() - start_time
    peak_mem_gb = (
        float(torch.cuda.max_memory_allocated(device) / (1024**3))
        if device.type == "cuda"
        else 0.0
    )
    train_loss = loss_sum / max(loss_count, 1)
    return train_loss, num_samples, elapsed, peak_mem_gb, gpu_monitor.as_device_utilization()


@torch.no_grad()
def evaluate_pipeline(
    *,
    schedule: ScheduleGPipe,
    stage_module: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    expected_batch_size: int,
) -> tuple[float, float, float]:
    stage_module.eval()
    loss_sum = 0.0
    top1_sum = 0.0
    top5_sum = 0.0
    sample_count = 0

    for images, target in loader:
        valid_n = target.size(0)
        if valid_n < expected_batch_size:
            pad_n = expected_batch_size - valid_n
            repeats = (pad_n + valid_n - 1) // valid_n
            pad_images = images.repeat(repeats, *([1] * (images.ndim - 1)))[:pad_n]
            pad_target = target.repeat(repeats)[:pad_n]
            images = torch.cat([images, pad_images], dim=0)
            target = torch.cat([target, pad_target], dim=0)

        output = run_pipeline_step(
            schedule,
            images,
            target,
            rank=rank,
            world_size=world_size,
            device=device,
            losses=None,
            eval_mode=True,
        )
        if is_last_stage(rank, world_size):
            assert output is not None
            target = target[:valid_n].to(device, non_blocking=True)
            logits = resolve_model_output(output)[:valid_n]
            loss = criterion(logits, target)
            top1, top5 = compute_topk(logits, target, topk=(1, 5))
            batch_size = target.size(0)
            loss_sum += float(loss.item()) * batch_size
            top1_sum += top1 * batch_size
            top5_sum += top5 * batch_size
            sample_count += batch_size

    if is_last_stage(rank, world_size):
        metrics = torch.tensor(
            [
                loss_sum / max(sample_count, 1),
                top1_sum / max(sample_count, 1),
                top5_sum / max(sample_count, 1),
            ],
            device=device,
            dtype=torch.float64,
        )
    else:
        metrics = torch.zeros(3, device=device, dtype=torch.float64)
    dist.broadcast(metrics, src=world_size - 1)
    return float(metrics[0].item()), float(metrics[1].item()), float(metrics[2].item())


def main() -> None:
    args = parse_args()
    rank, world_size, _, device = setup_pipeline(args)

    try:
        if args.batch_size % args.microbatches != 0:
            raise ValueError("--batch-size must be divisible by --microbatches for stable pipeline shapes.")
        if args.microbatches < world_size:
            raise ValueError("--microbatches must be >= the number of pipeline stages.")

        set_seed(args.seed)
        setup_cudnn()
        betas = (0.9, 0.999) if args.opt_betas is None else (float(args.opt_betas[0]), float(args.opt_betas[1]))

        temp_model = build_model(
            args.model,
            num_classes=1000,
            pretrained=not args.no_pretrained,
            drop_rate=args.drop_rate,
            drop_path_rate=args.drop_path,
            img_size=args.input_size,
        )
        train_transform, eval_transform, _ = build_transforms(
            temp_model,
            auto_augment=args.auto_augment,
            re_prob=args.re_prob,
            color_jitter=args.color_jitter,
            input_size=args.input_size,
            mean=args.mean,
            std=args.std,
            train_interpolation=args.train_interpolation,
            crop_scale=args.crop_scale,
            crop_ratio=args.crop_ratio,
            crop_pct=args.crop_pct,
            re_mode=args.re_mode,
            re_count=args.re_count,
        )
        del temp_model
        maybe_download_dataset_once(args, train_transform, eval_transform, rank)
        train_ds, val_ds, num_classes = build_datasets(
            dataset_name=args.dataset,
            data_root=args.data_root,
            train_transform=train_transform,
            eval_transform=eval_transform,
            download=False,
        )

        full_model = build_model(
            args.model,
            num_classes=num_classes,
            pretrained=not args.no_pretrained,
            drop_rate=args.drop_rate,
            drop_path_rate=args.drop_path,
            img_size=args.input_size,
        )
        stage_module = build_pipeline_stage_module(full_model, rank, world_size).to(device)
        set_seed(args.seed + rank)

        train_criterion = create_train_criterion(use_mixup=False, label_smoothing=args.label_smoothing)
        eval_criterion = nn.CrossEntropyLoss()
        optimizer = create_optimizer(
            stage_module,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=betas,
            eps=args.opt_eps,
        )
        scheduler = create_scheduler(
            optimizer,
            epochs=args.epochs,
            base_lr=args.lr,
            warmup_epochs=args.warmup_epochs,
            warmup_lr=args.warmup_lr,
            min_lr=args.min_lr,
        )

        start_epoch, best_top1, best_epoch = load_checkpoint_if_requested(
            stage_module=stage_module,
            optimizer=optimizer,
            scheduler=scheduler,
            resume_dir=args.resume_dir,
            rank=rank,
            world_size=world_size,
            device=device,
        )
        resume_meta = torch.tensor(
            [start_epoch, best_top1, best_epoch],
            device=device,
            dtype=torch.float64,
        )
        dist.broadcast(resume_meta, src=0)
        start_epoch = int(resume_meta[0].item())
        best_top1 = float(resume_meta[1].item())
        best_epoch = int(resume_meta[2].item())

        stage = PipelineStage(stage_module, rank, world_size, device)
        schedule = ScheduleGPipe(
            stage,
            n_microbatches=args.microbatches,
            loss_fn=train_criterion,
        )

        output_dir = Path(args.output_dir)
        metrics_csv = output_dir / "metrics.csv"
        summary_json = output_dir / "summary.json"
        last_ckpt = output_dir / f"stage{rank:02d}_last.pt"
        best_ckpt = output_dir / f"stage{rank:02d}_best.pt"

        time_to_target_s = None
        run_start = time.perf_counter()

        for epoch in range(start_epoch, args.epochs + 1):
            scheduler.step(epoch - 1)
            train_loader = make_loader(
                train_ds,
                args.batch_size,
                args.num_workers,
                device,
                shuffle=True,
                seed=args.seed + epoch,
                drop_last=True,
            )
            val_loader = make_loader(
                val_ds,
                args.batch_size,
                args.num_workers,
                device,
                shuffle=False,
                seed=args.seed,
                drop_last=False,
            )
            validate_loader_lengths(train_loader, val_loader, device)

            local_loss, num_samples, local_epoch_time_s, local_peak_mem_gb, local_gpu_utilization = train_one_epoch_pipeline(
                schedule=schedule,
                stage_module=stage_module,
                loader=train_loader,
                optimizer=optimizer,
                rank=rank,
                world_size=world_size,
                device=device,
                grad_clip=args.grad_clip,
            )

            epoch_time_tensor = torch.tensor([local_epoch_time_s], device=device, dtype=torch.float64)
            mem_tensor = torch.tensor([local_peak_mem_gb], device=device, dtype=torch.float64)
            dist.all_reduce(epoch_time_tensor, op=dist.ReduceOp.MAX)
            dist.all_reduce(mem_tensor, op=dist.ReduceOp.MAX)
            epoch_time_s = float(epoch_time_tensor.item())
            peak_mem_gb = float(mem_tensor.item())
            train_throughput = num_samples / max(epoch_time_s, 1e-8)

            train_loss_tensor = torch.tensor([local_loss], device=device, dtype=torch.float64)
            dist.broadcast(train_loss_tensor, src=world_size - 1)
            train_loss = float(train_loss_tensor.item())

            gathered_gpu_utilization = [None for _ in range(world_size)]
            dist.all_gather_object(gathered_gpu_utilization, local_gpu_utilization)
            gpu_utilization = format_gpu_utilization(
                [
                    (f"rank{rank_idx}/{device_name}", value)
                    for rank_idx, per_rank_utilization in enumerate(gathered_gpu_utilization)
                    for device_name, value in per_rank_utilization
                ]
            )

            val_loss, val_top1, val_top5 = evaluate_pipeline(
                schedule=schedule,
                stage_module=stage_module,
                loader=val_loader,
                criterion=eval_criterion,
                rank=rank,
                world_size=world_size,
                device=device,
                expected_batch_size=args.batch_size,
            )

            is_best = val_top1 > best_top1
            if is_best:
                best_top1 = val_top1
                best_epoch = epoch

            checkpoint = {
                "epoch": epoch,
                "stage_index": rank,
                "world_size": world_size,
                "model": stage_module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_top1": best_top1,
                "best_epoch": best_epoch,
                "args": vars(args),
            }
            save_checkpoint(last_ckpt, checkpoint)
            if is_best:
                save_checkpoint(best_ckpt, checkpoint)
            dist.barrier()

            elapsed_s = time.perf_counter() - run_start
            if args.target_top1 is not None and time_to_target_s is None and val_top1 >= args.target_top1:
                time_to_target_s = elapsed_s

            if rank == 0:
                metrics = EpochMetrics(
                    epoch=epoch,
                    train_loss=train_loss,
                    train_top1=float("nan"),
                    val_loss=val_loss,
                    val_top1=val_top1,
                    val_top5=val_top5,
                    train_throughput_img_s=train_throughput,
                    epoch_time_s=epoch_time_s,
                    peak_mem_gb=peak_mem_gb,
                    lr=current_lr(optimizer),
                    gpu_utilization=gpu_utilization,
                )
                write_metrics_row(metrics_csv, metrics)
                print(
                    f"Epoch {epoch:03d} | "
                    f"lr={current_lr(optimizer):.6g} | "
                    f"train_loss={train_loss:.4f} train_top1=nan | "
                    f"val_loss={val_loss:.4f} val_top1={val_top1:.2f} val_top5={val_top5:.2f} | "
                    f"throughput={train_throughput:.2f} img/s | peak_mem={peak_mem_gb:.2f} GB | "
                    f"gpu_utilization={gpu_utilization or 'unavailable'} | "
                    f"pipeline_stages={world_size}"
                )

        if rank == 0:
            total_time_s = time.perf_counter() - run_start
            summary = format_run_summary(
                args_dict={**vars(args), "world_size": world_size, "parallelism": "pipeline"},
                best_top1=best_top1,
                best_epoch=best_epoch,
                total_time_s=total_time_s,
                time_to_target_s=time_to_target_s,
            )
            save_json(summary_json, summary)
            print("Saved results to:", output_dir.resolve())
            print(summary)
    finally:
        cleanup_pipeline()


if __name__ == "__main__":
    main()
