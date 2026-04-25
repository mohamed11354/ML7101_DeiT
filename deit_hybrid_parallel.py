from __future__ import annotations

import argparse
import contextlib
import math
import os
import time
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from deit_common import (
    AverageMeter,
    EpochMetrics,
    build_datasets,
    build_model,
    build_transforms,
    compute_topk,
    create_mixup_fn,
    create_optimizer,
    create_scheduler,
    create_train_criterion,
    current_lr,
    format_gpu_utilization,
    GpuUtilizationMonitor,
    format_run_summary,
    resolve_model_output,
    save_checkpoint,
    save_json,
    set_seed,
    setup_cudnn,
    write_metrics_row,
)


"""
Hybrid parallel DeiT training.

Launch examples:

1 GPU baseline:
    python deit_hybrid_parallel.py --gpus-per-replica 1

2 GPU DDP baseline:
    torchrun --nproc_per_node 2 deit_hybrid_parallel.py --gpus-per-replica 1

2 GPU model/pipeline parallel:
    python deit_hybrid_parallel.py --gpus-per-replica 2 --pipeline-chunks 4

4 GPU hybrid (2-way model parallel inside each replica + 2-way DDP):
    torchrun --nproc_per_node 2 deit_hybrid_parallel.py --gpus-per-replica 2 --pipeline-chunks 4
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hybrid-parallel DeiT training script.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar100",
        choices=["food101", "cifar100", "flowers102", "imagefolder"],
    )
    parser.add_argument("--data-root", type=str, default="/l/users/mohamed.awad/parallel_project/datasets")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--model", type=str, default="deit_small_patch16_224.fb_in1k")

    # Model / optimization.
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
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

    # Augmentation / regularization.
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--auto-augment", type=str, default="rand-m9-mstd0.5-inc1")
    parser.add_argument("--re-prob", type=float, default=0.25)
    parser.add_argument("--re-mode", type=str, default="pixel")
    parser.add_argument("--re-count", type=int, default=1)
    parser.add_argument("--color-jitter", type=float, default=0.3)
    parser.add_argument("--mixup", type=float, default=0.8)
    parser.add_argument("--cutmix", type=float, default=1.0)
    parser.add_argument("--cutmix-minmax", type=float, nargs="+", default=None)
    parser.add_argument("--mixup-prob", type=float, default=1.0)
    parser.add_argument("--mixup-switch-prob", type=float, default=0.5)
    parser.add_argument("--mixup-mode", type=str, default="batch", choices=["batch", "pair", "elem"])

    # Input pipeline.
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--train-interpolation", type=str, default="bicubic")
    parser.add_argument("--mean", type=float, nargs=3, default=[0.5071, 0.4867, 0.4408])
    parser.add_argument("--std", type=float, nargs=3, default=[0.2675, 0.2565, 0.2761])
    parser.add_argument("--crop-scale", type=float, nargs=2, default=[0.8, 1.0])
    parser.add_argument("--crop-ratio", type=float, nargs=2, default=[0.75, 4 / 3])
    parser.add_argument("--crop-pct", type=float, default=1.0)

    # Runtime / bookkeeping.
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--target-top1", type=float, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="./outputs_hybrid")

    # Hybrid-parallel options.
    parser.add_argument(
        "--gpus-per-replica",
        type=int,
        default=2,
        choices=[1, 2],
        help="1 = single-device replica (plain or DDP), 2 = 2-way model/pipeline parallel replica.",
    )
    parser.add_argument(
        "--split-index",
        type=int,
        default=-1,
        help="Number of transformer blocks on stage 0. -1 uses len(blocks)//2.",
    )
    parser.add_argument(
        "--pipeline-chunks",
        type=int,
        default=4,
        help="Number of micro-batches used inside a 2-stage replica. Ignored when gpus-per-replica=1.",
    )
    parser.add_argument(
        "--ddp-broadcast-buffers",
        action="store_true",
        help="Enable DDP buffer broadcasts. DeiT usually does not need this because it uses LayerNorm, not BatchNorm.",
    )
    return parser


class ModelParallelDeiT(nn.Module):
    """Two-stage DeiT split across two GPUs with simple fill-drain microbatching."""

    def __init__(
        self,
        model: nn.Module,
        stage0_device: torch.device,
        stage1_device: torch.device,
        split_index: int,
        pipeline_chunks: int = 4,
    ) -> None:
        super().__init__()
        if not hasattr(model, "blocks"):
            raise TypeError("Expected a ViT/DeiT-style model with a 'blocks' attribute.")
        num_blocks = len(model.blocks)
        if num_blocks < 2:
            raise ValueError(f"Need at least 2 transformer blocks, got {num_blocks}.")
        if split_index <= 0 or split_index >= num_blocks:
            raise ValueError(f"split_index must be in [1, {num_blocks - 1}], got {split_index}.")

        self.model = model
        self.input_device = torch.device(stage0_device)
        self.output_device = torch.device(stage1_device)
        self.stage_devices = [self.input_device, self.output_device]
        self.split_index = int(split_index)
        self.pipeline_chunks = max(1, int(pipeline_chunks))
        self.num_blocks = num_blocks

        # Stage 0: patch embedding + token/position handling + first half of blocks.
        self.model.patch_embed.to(self.input_device)
        if hasattr(self.model, "pos_drop") and self.model.pos_drop is not None:
            self.model.pos_drop.to(self.input_device)
        if hasattr(self.model, "patch_drop") and self.model.patch_drop is not None:
            self.model.patch_drop.to(self.input_device)
        if hasattr(self.model, "norm_pre") and self.model.norm_pre is not None:
            self.model.norm_pre.to(self.input_device)
        for block_idx in range(self.split_index):
            self.model.blocks[block_idx].to(self.input_device)

        # Stage 1: remaining blocks + normalization + classifier head.
        for block_idx in range(self.split_index, self.num_blocks):
            self.model.blocks[block_idx].to(self.output_device)
        if hasattr(self.model, "norm") and self.model.norm is not None:
            self.model.norm.to(self.output_device)
        if hasattr(self.model, "fc_norm") and self.model.fc_norm is not None:
            self.model.fc_norm.to(self.output_device)
        if hasattr(self.model, "head_drop") and self.model.head_drop is not None:
            self.model.head_drop.to(self.output_device)
        if hasattr(self.model, "attn_pool") and self.model.attn_pool is not None:
            self.model.attn_pool.to(self.output_device)
        if hasattr(self.model, "head") and self.model.head is not None:
            self.model.head.to(self.output_device)

        # Move non-module parameters used before the block stack.
        for name in ("cls_token", "reg_token", "dist_token", "pos_embed"):
            param = getattr(self.model, name, None)
            if param is not None:
                setattr(self.model, name, nn.Parameter(param.detach().clone().to(self.input_device)))

    @torch.jit.ignore
    def no_weight_decay(self):
        base = set()
        if hasattr(self.model, "no_weight_decay") and callable(self.model.no_weight_decay):
            try:
                base = set(self.model.no_weight_decay())
            except Exception:
                base = set()
        return {f"model.{name}" for name in base}

    def _fallback_pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            x = x.flatten(2).transpose(1, 2)

        batch_size = x.shape[0]
        prefix_tokens = []
        for name in ("cls_token", "reg_token", "dist_token"):
            token = getattr(self.model, name, None)
            if token is None:
                continue
            if token.ndim == 2:
                token = token.unsqueeze(0)
            prefix_tokens.append(token.expand(batch_size, -1, -1))

        pos_embed = getattr(self.model, "pos_embed", None)
        no_embed_class = bool(getattr(self.model, "no_embed_class", False))
        if no_embed_class:
            if pos_embed is not None:
                x = x + pos_embed
            if prefix_tokens:
                x = torch.cat([*prefix_tokens, x], dim=1)
        else:
            if prefix_tokens:
                x = torch.cat([*prefix_tokens, x], dim=1)
            if pos_embed is not None:
                x = x + pos_embed

        pos_drop = getattr(self.model, "pos_drop", None)
        if pos_drop is not None:
            x = pos_drop(x)
        return x

    def _stage0_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.patch_embed(x)
        if hasattr(self.model, "_pos_embed") and callable(self.model._pos_embed):
            x = self.model._pos_embed(x)
        else:
            x = self._fallback_pos_embed(x)
        if hasattr(self.model, "patch_drop") and self.model.patch_drop is not None:
            x = self.model.patch_drop(x)
        if hasattr(self.model, "norm_pre") and self.model.norm_pre is not None:
            x = self.model.norm_pre(x)
        for block_idx in range(self.split_index):
            x = self.model.blocks[block_idx](x)
        return x

    def _fallback_forward_head(self, x: torch.Tensor) -> torch.Tensor:
        attn_pool = getattr(self.model, "attn_pool", None)
        if attn_pool is not None:
            x = attn_pool(x)

        pool_type = str(getattr(self.model, "global_pool", "token") or "token")
        num_prefix_tokens = int(getattr(self.model, "num_prefix_tokens", 1))
        include_prefix = bool(getattr(self.model, "pool_include_prefix", False))

        if x.ndim == 3:
            if pool_type == "token":
                x = x[:, 0]
            else:
                token_view = x if include_prefix else x[:, num_prefix_tokens:]
                if pool_type == "avg":
                    x = token_view.mean(dim=1)
                elif pool_type == "avgmax":
                    x = 0.5 * (token_view.amax(dim=1) + token_view.mean(dim=1))
                elif pool_type == "max":
                    x = token_view.amax(dim=1)
                elif not pool_type:
                    x = x[:, 0]
                else:
                    raise ValueError(f"Unsupported global_pool='{pool_type}'.")

        fc_norm = getattr(self.model, "fc_norm", None)
        if fc_norm is not None:
            x = fc_norm(x)
        head_drop = getattr(self.model, "head_drop", None)
        if head_drop is not None:
            x = head_drop(x)
        return self.model.head(x)

    def _stage1_forward(self, x: torch.Tensor) -> torch.Tensor:
        for block_idx in range(self.split_index, self.num_blocks):
            x = self.model.blocks[block_idx](x)
        if hasattr(self.model, "norm") and self.model.norm is not None:
            x = self.model.norm(x)
        if hasattr(self.model, "forward_head") and callable(self.model.forward_head):
            return self.model.forward_head(x)
        return self._fallback_forward_head(x)

    def _forward_single(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self._stage0_forward(x.to(self.input_device, non_blocking=True))
        return self._stage1_forward(hidden.to(self.output_device, non_blocking=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pipeline_chunks <= 1 or x.size(0) <= 1:
            return self._forward_single(x)

        chunk_size = max(1, math.ceil(x.size(0) / self.pipeline_chunks))
        microbatches = list(x.split(chunk_size, dim=0))
        if len(microbatches) <= 1:
            return self._forward_single(x)

        outputs = []
        pending_hidden = None
        for micro_idx in range(len(microbatches) + 1):
            next_hidden = None
            if micro_idx < len(microbatches):
                micro_in = microbatches[micro_idx].to(self.input_device, non_blocking=True)
                next_hidden = self._stage0_forward(micro_in)
            if pending_hidden is not None:
                micro_out = self._stage1_forward(pending_hidden.to(self.output_device, non_blocking=True))
                outputs.append(micro_out)
            pending_hidden = next_hidden
        return torch.cat(outputs, dim=0)


class DistEnv:
    def __init__(self, rank: int, world_size: int, local_rank: int, distributed: bool) -> None:
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.local_rank = int(local_rank)
        self.distributed = bool(distributed)



def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()



def infer_dist_env() -> DistEnv:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return DistEnv(
            rank=int(os.environ["RANK"]),
            world_size=int(os.environ["WORLD_SIZE"]),
            local_rank=int(os.environ.get("LOCAL_RANK", 0)),
            distributed=int(os.environ["WORLD_SIZE"]) > 1,
        )
    return DistEnv(rank=0, world_size=1, local_rank=0, distributed=False)



def resolve_replica_devices(local_rank: int, gpus_per_replica: int) -> list[torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires CUDA because the project requires multi-GPU execution.")

    visible_gpu_count = torch.cuda.device_count()
    start = int(local_rank) * int(gpus_per_replica)
    end = start + int(gpus_per_replica)
    if end > visible_gpu_count:
        raise RuntimeError(
            f"Not enough visible GPUs for local_rank={local_rank}, gpus_per_replica={gpus_per_replica}. "
            f"Visible GPUs={visible_gpu_count}, required up to index {end - 1}."
        )
    return [torch.device(f"cuda:{idx}") for idx in range(start, end)]



def init_process_group_if_needed(dist_env: DistEnv, primary_device: torch.device) -> None:
    torch.cuda.set_device(primary_device)
    if dist_env.distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")



def cleanup_process_group() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()



def is_main_process(dist_env: DistEnv) -> bool:
    return dist_env.rank == 0



def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model



def get_base_model(model: nn.Module) -> nn.Module:
    module = unwrap_model(model)
    if isinstance(module, ModelParallelDeiT):
        return module.model
    return module



def model_input_output_devices(model: nn.Module) -> tuple[torch.device, torch.device]:
    module = unwrap_model(model)
    if isinstance(module, ModelParallelDeiT):
        return module.input_device, module.output_device
    param = next(module.parameters())
    return param.device, param.device



def model_devices(model: nn.Module) -> list[torch.device]:
    module = unwrap_model(model)
    if isinstance(module, ModelParallelDeiT):
        return list(module.stage_devices)
    param = next(module.parameters())
    return [param.device]



def distributed_sum(value: float | int, device: torch.device) -> float:
    tensor = torch.tensor(float(value), device=device, dtype=torch.float64)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())



def distributed_max(value: float | int, device: torch.device) -> float:
    tensor = torch.tensor(float(value), device=device, dtype=torch.float64)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())



def distributed_meter_avg(meter: AverageMeter, device: torch.device) -> float:
    packed = torch.tensor([float(meter.sum), float(meter.count)], device=device, dtype=torch.float64)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    count = max(float(packed[1].item()), 1.0)
    return float(packed[0].item() / count)



def move_optimizer_state_to_param_devices(optimizer: torch.optim.Optimizer) -> None:
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state.get(param, None)
            if not state:
                continue
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(param.device)



def load_checkpoint_if_requested(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.cuda.amp.GradScaler,
    resume_path: str | None,
) -> tuple[int, float, int]:
    if not resume_path:
        return 1, -1.0, -1

    checkpoint = torch.load(resume_path, map_location="cpu")
    get_base_model(model).load_state_dict(checkpoint["model"])
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        move_optimizer_state_to_param_devices(optimizer)
    if "scheduler" in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint["scheduler"])
        except Exception:
            pass
    if "scaler" in checkpoint:
        try:
            scaler.load_state_dict(checkpoint["scaler"])
        except Exception:
            pass

    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_top1 = float(checkpoint.get("best_val_top1", -1.0))
    best_epoch = int(checkpoint.get("best_epoch", start_epoch - 1))
    return start_epoch, best_top1, best_epoch



def build_model_for_training(
    args: argparse.Namespace,
    num_classes: int,
    replica_devices: Sequence[torch.device],
) -> tuple[nn.Module, int]:
    model = build_model(
        args.model,
        num_classes=num_classes,
        pretrained=not args.no_pretrained,
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path,
        img_size=args.input_size,
    )

    if args.gpus_per_replica == 1:
        return model.to(replica_devices[0]), -1

    num_blocks = len(model.blocks)
    split_index = int(args.split_index) if int(args.split_index) > 0 else num_blocks // 2
    wrapped = ModelParallelDeiT(
        model=model,
        stage0_device=replica_devices[0],
        stage1_device=replica_devices[1],
        split_index=split_index,
        pipeline_chunks=args.pipeline_chunks,
    )
    return wrapped, split_index



def make_data_loaders(
    *,
    args: argparse.Namespace,
    train_ds,
    val_ds,
    dist_env: DistEnv,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader, DistributedSampler | None, DistributedSampler | None]:
    train_sampler = None
    val_sampler = None
    if dist_env.distributed:
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=dist_env.world_size,
            rank=dist_env.rank,
            shuffle=True,
            seed=args.seed,
            drop_last=False,
        )
        val_sampler = DistributedSampler(
            val_ds,
            num_replicas=dist_env.world_size,
            rank=dist_env.rank,
            shuffle=False,
            drop_last=False,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, args.batch_size * 2),
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=(args.num_workers > 0),
    )
    return train_loader, val_loader, train_sampler, val_sampler



def build_run_args_dict(
    args: argparse.Namespace,
    dist_env: DistEnv,
    replica_devices: Sequence[torch.device],
    split_index: int,
) -> dict:
    args_dict = vars(args).copy()
    args_dict["world_size"] = dist_env.world_size
    args_dict["rank"] = dist_env.rank
    args_dict["local_rank"] = dist_env.local_rank
    args_dict["replica_devices"] = [str(device) for device in replica_devices]
    args_dict["effective_global_batch"] = args.batch_size * max(1, dist_env.world_size) * args.grad_accum_steps
    args_dict["total_gpus_used"] = len(replica_devices) * max(1, dist_env.world_size)
    args_dict["data_parallel_replicas"] = max(1, dist_env.world_size)
    args_dict["model_parallel_stages"] = len(replica_devices)
    args_dict["actual_split_index"] = split_index
    return args_dict



def train_one_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.cuda.amp.GradScaler,
    use_amp: bool,
    grad_accum_steps: int,
    mixup_fn=None,
    grad_clip: float | None = None,
) -> tuple[float, float, int, float, float, list[tuple[str, float]]]:
    model.train()
    in_device, out_device = model_input_output_devices(model)
    all_devices = model_devices(model)
    if in_device.type == "cuda":
        for device in all_devices:
            torch.cuda.reset_peak_memory_stats(device)

    gpu_monitor = GpuUtilizationMonitor(all_devices)
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    num_samples = 0
    num_top1_samples = 0
    optimizer.zero_grad(set_to_none=True)
    start_time = time.perf_counter()

    for step, (images, target) in enumerate(loader):
        images = images.to(in_device, non_blocking=True)
        target = target.to(in_device, non_blocking=True)
        batch_size = images.size(0)
        hard_target = target

        if mixup_fn is not None:
            images, target = mixup_fn(images, target)

        target_for_loss = target.to(out_device, non_blocking=True)
        hard_target_out = hard_target.to(out_device, non_blocking=True)
        should_step = ((step + 1) % grad_accum_steps == 0) or ((step + 1) == len(loader))
        sync_context = contextlib.nullcontext()
        if isinstance(model, DDP) and not should_step:
            sync_context = model.no_sync()

        with sync_context:
            with torch.amp.autocast(enabled=use_amp):
                logits = resolve_model_output(model(images))
                loss = criterion(logits, target_for_loss)
                loss_for_backward = loss / grad_accum_steps
            scaler.scale(loss_for_backward).backward()

        if should_step:
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), max_norm=float(grad_clip))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if mixup_fn is None:
            top1, _ = compute_topk(logits.detach(), hard_target_out, topk=(1, 5))
            top1_meter.update(top1, batch_size)
            num_top1_samples += batch_size

        loss_meter.update(float(loss.item()), batch_size)
        num_samples += batch_size
        gpu_monitor.maybe_sample(force=(step == 0))

    gpu_monitor.maybe_sample(force=True)
    epoch_time_s = distributed_max(time.perf_counter() - start_time, out_device)
    if in_device.type == "cuda":
        peak_mem_local = max(float(torch.cuda.max_memory_allocated(device) / (1024 ** 3)) for device in all_devices)
        peak_mem_gb = distributed_max(peak_mem_local, out_device)
    else:
        peak_mem_gb = 0.0

    train_loss = distributed_meter_avg(loss_meter, out_device)
    train_top1 = float("nan") if num_top1_samples == 0 else distributed_meter_avg(top1_meter, out_device)
    global_samples = int(round(distributed_sum(num_samples, out_device)))
    return train_loss, train_top1, global_samples, epoch_time_s, peak_mem_gb, gpu_monitor.as_device_utilization()


@torch.no_grad()
def evaluate(
    *,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    use_amp: bool,
) -> tuple[float, float, float]:
    model.eval()
    in_device, out_device = model_input_output_devices(model)

    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    for images, target in loader:
        images = images.to(in_device, non_blocking=True)
        target = target.to(out_device, non_blocking=True)
        with torch.amp.autocast(enabled=use_amp):
            logits = resolve_model_output(model(images))
            loss = criterion(logits, target)
        top1, top5 = compute_topk(logits, target, topk=(1, 5))
        batch_size = images.size(0)
        loss_meter.update(float(loss.item()), batch_size)
        top1_meter.update(top1, batch_size)
        top5_meter.update(top5, batch_size)

    return (
        distributed_meter_avg(loss_meter, out_device),
        distributed_meter_avg(top1_meter, out_device),
        distributed_meter_avg(top5_meter, out_device),
    )



def main() -> None:
    args = parse_args()
    dist_env = infer_dist_env()
    setup_cudnn()
    set_seed(args.seed + dist_env.rank)

    replica_devices = resolve_replica_devices(dist_env.local_rank, args.gpus_per_replica)
    init_process_group_if_needed(dist_env, replica_devices[0])

    use_amp = bool(args.amp)
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
    train_ds, val_ds, num_classes = build_datasets(
        dataset_name=args.dataset,
        data_root=args.data_root,
        train_transform=train_transform,
        eval_transform=eval_transform,
        download=args.download,
    )

    model, split_index = build_model_for_training(args, num_classes=num_classes, replica_devices=replica_devices)

    # Rebuild transforms after constructing the actual training model, matching the sequential script.
    transform_model = get_base_model(model)
    train_transform, eval_transform, _ = build_transforms(
        transform_model,
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
    train_ds, val_ds, num_classes = build_datasets(
        dataset_name=args.dataset,
        data_root=args.data_root,
        train_transform=train_transform,
        eval_transform=eval_transform,
        download=False,
    )

    train_loader, val_loader, train_sampler, _ = make_data_loaders(
        args=args,
        train_ds=train_ds,
        val_ds=val_ds,
        dist_env=dist_env,
        pin_memory=True,
    )

    mixup_fn = create_mixup_fn(
        mixup=args.mixup,
        cutmix=args.cutmix,
        cutmix_minmax=args.cutmix_minmax,
        mixup_prob=args.mixup_prob,
        mixup_switch_prob=args.mixup_switch_prob,
        mixup_mode=args.mixup_mode,
        label_smoothing=args.label_smoothing,
        num_classes=num_classes,
    )
    train_criterion = create_train_criterion(use_mixup=(mixup_fn is not None), label_smoothing=args.label_smoothing)
    eval_criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(
        model,
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
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch, best_top1, best_epoch = load_checkpoint_if_requested(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        resume_path=args.resume,
    )

    if dist_env.distributed:
        model = DDP(
            model,
            device_ids=None,
            output_device=None,
            broadcast_buffers=bool(args.ddp_broadcast_buffers),
            find_unused_parameters=False,
        )

    output_dir = Path(args.output_dir)
    metrics_csv = output_dir / "metrics.csv"
    best_ckpt = output_dir / "best.pt"
    last_ckpt = output_dir / "last.pt"
    summary_json = output_dir / "summary.json"

    time_to_target_s = None
    run_start = time.perf_counter()
    run_args_dict = build_run_args_dict(args, dist_env, replica_devices, split_index)

    for epoch in range(start_epoch, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        scheduler.step(epoch - 1)

        train_loss, train_top1, num_samples, epoch_time_s, peak_mem_gb, local_gpu_utilization = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=train_criterion,
            scaler=scaler,
            use_amp=use_amp,
            grad_accum_steps=args.grad_accum_steps,
            mixup_fn=mixup_fn,
            grad_clip=args.grad_clip,
        )
        if dist_env.distributed:
            gathered_gpu_utilization = [None for _ in range(dist_env.world_size)]
            dist.all_gather_object(gathered_gpu_utilization, local_gpu_utilization)
            gpu_utilization = format_gpu_utilization(
                [
                    (f"rank{rank_idx}/{device_name}", value)
                    for rank_idx, per_rank_utilization in enumerate(gathered_gpu_utilization)
                    for device_name, value in per_rank_utilization
                ]
            )
        else:
            gpu_utilization = format_gpu_utilization(local_gpu_utilization)

        train_throughput = num_samples / max(epoch_time_s, 1e-8)
        val_loss, val_top1, val_top5 = evaluate(
            model=model,
            loader=val_loader,
            criterion=eval_criterion,
            use_amp=use_amp,
        )

        is_best = val_top1 > best_top1
        if is_best:
            best_top1 = val_top1
            best_epoch = epoch

        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            train_top1=train_top1,
            val_loss=val_loss,
            val_top1=val_top1,
            val_top5=val_top5,
            train_throughput_img_s=train_throughput,
            epoch_time_s=epoch_time_s,
            peak_mem_gb=peak_mem_gb,
            lr=current_lr(optimizer),
            gpu_utilization=gpu_utilization,
        )

        if is_main_process(dist_env):
            write_metrics_row(metrics_csv, metrics)
            checkpoint = {
                "epoch": epoch,
                "model": get_base_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_val_top1": best_top1,
                "best_epoch": best_epoch,
                "args": run_args_dict,
            }
            save_checkpoint(last_ckpt, checkpoint)
            if is_best:
                save_checkpoint(best_ckpt, checkpoint)

        elapsed_s = time.perf_counter() - run_start
        if args.target_top1 is not None and time_to_target_s is None and val_top1 >= args.target_top1:
            time_to_target_s = elapsed_s

        if is_main_process(dist_env):
            print(
                f"Epoch {epoch:03d} | "
                f"lr={current_lr(optimizer):.6g} | "
                f"train_loss={train_loss:.4f} train_top1={train_top1:.2f} | "
                f"val_loss={val_loss:.4f} val_top1={val_top1:.2f} val_top5={val_top5:.2f} | "
                f"throughput={train_throughput:.2f} img/s | peak_mem={peak_mem_gb:.2f} GB"
            )

    total_time_s = time.perf_counter() - run_start
    summary = format_run_summary(
        args_dict=run_args_dict,
        best_top1=best_top1,
        best_epoch=best_epoch,
        total_time_s=total_time_s,
        time_to_target_s=time_to_target_s,
    )
    if is_main_process(dist_env):
        save_json(summary_json, summary)
        print("Saved results to:", output_dir.resolve())
        print(summary)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    cleanup_process_group()


if __name__ == "__main__":
    main()
