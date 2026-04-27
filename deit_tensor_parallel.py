from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from deit_common import (
    AverageMeter,
    EpochMetrics,
    GpuUtilizationMonitor,
    build_datasets,
    build_model,
    build_transforms,
    compute_topk,
    create_optimizer,
    create_scheduler,
    create_train_criterion,
    create_mixup_fn,
    mixup_active,
    resolve_model_output,
    save_checkpoint,
    set_seed,
    setup_cudnn,
    write_metrics_row,
)


def log_line(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with path.open("a", encoding="utf-8") as f:
        f.write(f"[{ts}] {message}\n")


def open_progress_stream():
    """
    Best effort:
    - if attached to a real terminal, write tqdm there
    - when stdout/stderr are piped through tee, /dev/tty still gives a live terminal stream
    - if unavailable, disable tqdm cleanly
    """
    try:
        return open("/dev/tty", "w", buffering=1, encoding="utf-8")
    except Exception:
        if sys.stderr.isatty():
            return sys.stderr
        return None


class _AllGatherLastDim(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, tp_group, tp_rank: int, tp_world_size: int) -> torch.Tensor:
        ctx.tp_group = tp_group
        ctx.tp_rank = int(tp_rank)
        ctx.tp_world_size = int(tp_world_size)

        gathered = [torch.empty_like(x) for _ in range(ctx.tp_world_size)]
        dist.all_gather(gathered, x, group=tp_group)
        return torch.cat(gathered, dim=-1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        chunks = grad_output.chunk(ctx.tp_world_size, dim=-1)
        grad_local = chunks[ctx.tp_rank].contiguous()
        dist.all_reduce(grad_local, op=dist.ReduceOp.SUM, group=ctx.tp_group)
        return grad_local, None, None, None


class _AllReduceSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, tp_group) -> torch.Tensor:
        ctx.tp_group = tp_group
        y = x.contiguous()
        dist.all_reduce(y, op=dist.ReduceOp.SUM, group=tp_group)
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_input = grad_output.contiguous()
        dist.all_reduce(grad_input, op=dist.ReduceOp.SUM, group=ctx.tp_group)
        return grad_input, None


class TPLinearColumn(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        tp_rank: int,
        tp_world_size: int,
        tp_group,
        gather_output: bool,
    ) -> None:
        super().__init__()
        self.in_features = int(linear.in_features)
        self.out_features = int(linear.out_features)
        if self.out_features % tp_world_size != 0:
            raise ValueError(
                f"Column shard requires out_features divisible by tp size: {self.out_features} vs {tp_world_size}"
            )
        self.local_out_features = self.out_features // tp_world_size
        self.start = tp_rank * self.local_out_features
        self.end = self.start + self.local_out_features
        self.tp_rank = int(tp_rank)
        self.tp_world_size = int(tp_world_size)
        self.tp_group = tp_group
        self.gather_output = bool(gather_output)

        self.weight = nn.Parameter(linear.weight[self.start:self.end, :].detach().clone())
        if linear.bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(linear.bias[self.start:self.end].detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_local = F.linear(x, self.weight, self.bias)
        if not self.gather_output:
            return y_local
        return _AllGatherLastDim.apply(y_local, self.tp_group, self.tp_rank, self.tp_world_size)


class TPLinearRow(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        tp_rank: int,
        tp_world_size: int,
        tp_group,
        input_is_parallel: bool,
    ) -> None:
        super().__init__()
        self.in_features = int(linear.in_features)
        self.out_features = int(linear.out_features)
        if self.in_features % tp_world_size != 0:
            raise ValueError(
                f"Row shard requires in_features divisible by tp size: {self.in_features} vs {tp_world_size}"
            )
        self.local_in_features = self.in_features // tp_world_size
        self.start = tp_rank * self.local_in_features
        self.end = self.start + self.local_in_features
        self.tp_group = tp_group
        self.input_is_parallel = bool(input_is_parallel)

        self.weight = nn.Parameter(linear.weight[:, self.start:self.end].detach().clone())
        if linear.bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(linear.bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_is_parallel:
            x_local = x
        else:
            x_local = x[..., self.start:self.end]

        y = F.linear(x_local, self.weight, None)
        y = _AllReduceSum.apply(y, self.tp_group)
        if self.bias is not None:
            y = y + self.bias
        return y


def apply_tensor_parallel_to_deit(model: nn.Module, tp_rank: int, tp_world_size: int, tp_group) -> None:
    if not hasattr(model, "blocks"):
        raise TypeError("Expected a DeiT/ViT model with .blocks")

    for block in model.blocks:
        if not hasattr(block, "attn") or not hasattr(block, "mlp"):
            raise TypeError("Unexpected transformer block structure for DeiT")

        attn = block.attn
        mlp = block.mlp

        attn.qkv = TPLinearColumn(
            linear=attn.qkv,
            tp_rank=tp_rank,
            tp_world_size=tp_world_size,
            tp_group=tp_group,
            gather_output=True,
        )
        attn.proj = TPLinearRow(
            linear=attn.proj,
            tp_rank=tp_rank,
            tp_world_size=tp_world_size,
            tp_group=tp_group,
            input_is_parallel=False,
        )

        mlp.fc1 = TPLinearColumn(
            linear=mlp.fc1,
            tp_rank=tp_rank,
            tp_world_size=tp_world_size,
            tp_group=tp_group,
            gather_output=False,
        )
        mlp.fc2 = TPLinearRow(
            linear=mlp.fc2,
            tp_rank=tp_rank,
            tp_world_size=tp_world_size,
            tp_group=tp_group,
            input_is_parallel=True,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tensor-parallel DeiT training with comprehensive metrics.")

    parser.add_argument("--dataset", type=str, default="cifar100", choices=["food101", "cifar100", "flowers102", "imagefolder"])
    parser.add_argument("--data-root", type=str, default="/l/users/mohamed.awad/parallel_project/datasets")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--model", type=str, default="deit_small_patch16_224.fb_in1k")

    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--opt-betas", type=float, nargs=2, default=None)
    parser.add_argument("--opt-eps", type=float, default=1e-8)
    parser.add_argument("--warmup-epochs", type=int, default=15)
    parser.add_argument("--warmup-lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)

    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--drop-rate", type=float, default=0.0)
    parser.add_argument("--drop-path", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--amp", action="store_true")

    parser.add_argument("--mixup", type=float, default=0.8)
    parser.add_argument("--cutmix", type=float, default=1.0)
    parser.add_argument("--cutmix-minmax", type=float, nargs="+", default=None)
    parser.add_argument("--mixup-prob", type=float, default=1.0)
    parser.add_argument("--mixup-switch-prob", type=float, default=0.5)
    parser.add_argument("--mixup-mode", type=str, default="batch", choices=["batch", "pair", "elem"])

    parser.add_argument("--auto-augment", type=str, default="rand-m9-mstd0.5-inc1")
    parser.add_argument("--re-prob", type=float, default=0.25)
    parser.add_argument("--re-mode", type=str, default="pixel")
    parser.add_argument("--re-count", type=int, default=1)
    parser.add_argument("--color-jitter", type=float, default=0.3)
    parser.add_argument("--train-interpolation", type=str, default="bicubic")
    parser.add_argument("--mean", type=float, nargs=3, default=[0.5071, 0.4867, 0.4408])
    parser.add_argument("--std", type=float, nargs=3, default=[0.2675, 0.2565, 0.2761])
    parser.add_argument("--crop-scale", type=float, nargs=2, default=[0.8, 1.0])
    parser.add_argument("--crop-ratio", type=float, nargs=2, default=[0.75, 4 / 3])
    parser.add_argument("--crop-pct", type=float, default=1.0)

    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=1)

    parser.add_argument("--world-size", type=int, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--master-addr", type=str, default=None)
    parser.add_argument("--master-port", type=int, default=None)
    parser.add_argument("--dist-backend", type=str, default="nccl")

    return parser


def setup_dist(args: argparse.Namespace) -> tuple[int, int, int, torch.device, object]:
    world_size = int(args.world_size) if args.world_size is not None else int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(args.rank) if args.rank is not None else int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    master_addr = args.master_addr if args.master_addr is not None else os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = int(args.master_port) if args.master_port is not None else int(os.environ.get("MASTER_PORT", "29500"))

    if world_size != 2:
        raise RuntimeError("This tensor-parallel script currently supports world_size=2 only.")
    if rank not in (0, 1):
        raise RuntimeError("Rank must be 0 or 1 for this tensor-parallel script.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["MASTER_ADDR"] = str(master_addr)
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ.setdefault("NCCL_DEBUG", "WARN")

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    dist.init_process_group(
        backend=args.dist_backend,
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=world_size,
        rank=rank,
    )
    return rank, world_size, local_rank, device, dist.group.WORLD


def cleanup_dist() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def infer_num_classes(dataset: str, data_root: str) -> int:
    if dataset == "food101":
        return 101
    if dataset == "cifar100":
        return 100
    if dataset == "flowers102":
        return 102
    if dataset == "imagefolder":
        train_dir = Path(data_root) / "train"
        if not train_dir.exists():
            raise FileNotFoundError(f"Missing train dir for imagefolder dataset: {train_dir}")
        from torchvision import datasets as tv_datasets
        return len(tv_datasets.ImageFolder(train_dir).classes)
    raise ValueError(f"Unsupported dataset: {dataset}")


def broadcast_batch(images: torch.Tensor | None, target: torch.Tensor | None, device: torch.device, src: int = 0):
    rank = dist.get_rank()
    if rank == src:
        assert images is not None and target is not None
        shape = torch.tensor(list(images.shape), dtype=torch.int64, device=device)
        if target.ndim == 1:
            target_meta = torch.tensor([0, int(target.shape[0]), 0], dtype=torch.int64, device=device)
            target = target.to(device, dtype=torch.int64, non_blocking=True)
        elif target.ndim == 2:
            target_meta = torch.tensor([1, int(target.shape[0]), int(target.shape[1])], dtype=torch.int64, device=device)
            target = target.to(device, dtype=torch.float32, non_blocking=True)
        else:
            raise ValueError(f"Unsupported target shape for broadcast: {tuple(target.shape)}")
        images = images.to(device, non_blocking=True)
    else:
        shape = torch.empty(4, dtype=torch.int64, device=device)
        target_meta = torch.empty(3, dtype=torch.int64, device=device)

    dist.broadcast(shape, src=src)
    bsz, c, h, w = [int(x) for x in shape.tolist()]
    dist.broadcast(target_meta, src=src)
    target_kind, target_dim0, target_dim1 = [int(x) for x in target_meta.tolist()]

    if rank != src:
        images = torch.empty((bsz, c, h, w), dtype=torch.float32, device=device)
        if target_kind == 0:
            target = torch.empty((target_dim0,), dtype=torch.int64, device=device)
        elif target_kind == 1:
            target = torch.empty((target_dim0, target_dim1), dtype=torch.float32, device=device)
        else:
            raise ValueError(f"Unsupported broadcast target kind: {target_kind}")

    dist.broadcast(images, src=src)
    dist.broadcast(target, src=src)
    return images, target


@torch.no_grad()
def evaluate_tensor_parallel(
    model: nn.Module,
    loader: DataLoader | None,
    device: torch.device,
    criterion: nn.Module,
    rank: int,
    use_amp: bool,
) -> tuple[float, float, float]:
    model.eval()

    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    if rank == 0:
        assert loader is not None
        data_iter = iter(loader)
        num_steps = len(loader)
    else:
        data_iter = None
        num_steps = 0

    num_steps_tensor = torch.tensor([num_steps], dtype=torch.int64, device=device)
    dist.broadcast(num_steps_tensor, src=0)
    num_steps = int(num_steps_tensor.item())

    for _ in range(num_steps):
        if rank == 0:
            images, target = next(data_iter)
        else:
            images, target = None, None

        images, target = broadcast_batch(images, target, device=device, src=0)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = resolve_model_output(model(images))
            loss = criterion(logits, target)

        top1, top5 = compute_topk(logits, target, topk=(1, 5))

        if rank == 0:
            batch_size = images.size(0)
            loss_meter.update(float(loss.item()), batch_size)
            top1_meter.update(float(top1), batch_size)
            top5_meter.update(float(top5), batch_size)

    if rank == 0:
        metrics = torch.tensor(
            [loss_meter.avg, top1_meter.avg, top5_meter.avg],
            dtype=torch.float32,
            device=device,
        )
    else:
        metrics = torch.zeros(3, dtype=torch.float32, device=device)

    dist.broadcast(metrics, src=0)
    return float(metrics[0].item()), float(metrics[1].item()), float(metrics[2].item())


def gather_gpu_utilization_across_ranks(
    local_device_utilization: list[tuple[str, float]],
    rank: int,
    world_size: int,
) -> str:
    local_dict = {str(device_name): round(float(util), 2) for device_name, util in local_device_utilization}

    gathered: list[dict[str, float] | None] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_dict)

    if rank != 0:
        return ""

    merged: dict[str, float] = {}
    for item in gathered:
        if not item:
            continue
        for key, value in item.items():
            merged[str(key)] = round(float(value), 2)

    ordered_items = sorted(
        merged.items(),
        key=lambda kv: (
            0 if kv[0].startswith("cuda:") else 1,
            int(kv[0].split(":")[1]) if kv[0].startswith("cuda:") and kv[0].split(":")[1].isdigit() else kv[0],
        ),
    )
    return json.dumps({k: v for k, v in ordered_items})


def reduce_peak_memory_max(peak_mem_gb: float, device: torch.device) -> float:
    value = torch.tensor([float(peak_mem_gb)], dtype=torch.float32, device=device)
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return float(value.item())


def train_one_epoch_tensor_parallel(
    model: nn.Module,
    loader: DataLoader | None,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    mixup_fn,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    grad_accum_steps: int,
    rank: int,
    world_size: int,
    epoch: int,
    total_epochs: int,
    log_interval: int,
    grad_clip: float | None,
) -> tuple[float, float, int, float, float, list[tuple[str, float]]]:
    model.train()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    gpu_monitor = GpuUtilizationMonitor([device])

    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    sample_count = 0
    start_time = time.perf_counter()

    if rank == 0:
        assert loader is not None
        data_iter = iter(loader)
        num_steps = len(loader)
    else:
        data_iter = None
        num_steps = 0

    num_steps_tensor = torch.tensor([num_steps], dtype=torch.int64, device=device)
    dist.broadcast(num_steps_tensor, src=0)
    num_steps = int(num_steps_tensor.item())

    progress_stream = open_progress_stream() if rank == 0 else None
    pbar = None
    if rank == 0 and progress_stream is not None:
        pbar = tqdm(
            total=num_steps,
            desc=f"Epoch {epoch:03d}/{total_epochs:03d}",
            dynamic_ncols=True,
            leave=False,
            mininterval=1.0,
            file=progress_stream,
        )

    optimizer.zero_grad(set_to_none=True)
    for step in range(1, num_steps + 1):
        if rank == 0:
            images, target = next(data_iter)
            if mixup_fn is not None:
                images, target = mixup_fn(images, target)
        else:
            images, target = None, None

        images, target = broadcast_batch(images, target, device=device, src=0)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = resolve_model_output(model(images))
            loss = criterion(logits, target)
            loss_for_backward = loss / grad_accum_steps

        scaler.scale(loss_for_backward).backward()

        if (step % grad_accum_steps == 0) or (step == num_steps):
            if grad_clip is not None and grad_clip > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        loss_scalar = torch.tensor([float(loss.item())], dtype=torch.float32, device=device)
        dist.all_reduce(loss_scalar, op=dist.ReduceOp.SUM)
        mean_loss = float((loss_scalar / world_size).item())

        batch_size = images.shape[0]
        loss_meter.update(mean_loss, batch_size)
        sample_count += batch_size

        if target.ndim == 1:
            top1, _ = compute_topk(logits.detach(), target.long(), topk=(1, 5))
            top1_scalar = torch.tensor([float(top1)], dtype=torch.float32, device=device)
            dist.all_reduce(top1_scalar, op=dist.ReduceOp.SUM)
            mean_top1 = float((top1_scalar / world_size).item())
            top1_meter.update(mean_top1, batch_size)

        gpu_monitor.maybe_sample(force=(step == 1))

        if pbar is not None:
            pbar.update(1)
            if step % max(1, int(log_interval)) == 0 or step == num_steps:
                elapsed = max(time.perf_counter() - start_time, 1e-8)
                throughput = sample_count / elapsed
                top1_display = "nan" if top1_meter.count == 0 else f"{top1_meter.avg:.2f}"
                pbar.set_postfix(
                    loss=f"{loss_meter.avg:.4f}",
                    top1=top1_display,
                    lr=f"{optimizer.param_groups[0]['lr']:.3e}",
                    img_s=f"{throughput:.1f}",
                )

    if pbar is not None:
        pbar.close()
    if progress_stream is not None and progress_stream not in (sys.stderr, sys.stdout):
        progress_stream.close()

    gpu_monitor.maybe_sample(force=True)

    epoch_time_s = time.perf_counter() - start_time
    peak_mem_gb = (
        float(torch.cuda.max_memory_allocated(device) / (1024 ** 3))
        if device.type == "cuda"
        else 0.0
    )

    train_top1_value = float("nan") if top1_meter.count == 0 else top1_meter.avg

    return (
        loss_meter.avg,
        train_top1_value,
        sample_count,
        epoch_time_s,
        peak_mem_gb,
        gpu_monitor.as_device_utilization(),
    )


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(output_dir)

    rank, world_size, local_rank, device, tp_group = setup_dist(args)
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    try:
        set_seed(args.seed)
        setup_cudnn()

        num_classes = infer_num_classes(args.dataset, args.data_root)
        betas = (0.9, 0.999) if args.opt_betas is None else (float(args.opt_betas[0]), float(args.opt_betas[1]))

        model = build_model(
            args.model,
            num_classes=num_classes,
            pretrained=not args.no_pretrained,
            drop_rate=args.drop_rate,
            drop_path_rate=args.drop_path,
            img_size=args.input_size,
        ).to(device)

        apply_tensor_parallel_to_deit(model, tp_rank=rank, tp_world_size=world_size, tp_group=tp_group)

        if rank == 0:
            log_line(output_dir / "train.log", "Initialized tensor-parallel run")
            log_line(output_dir / "train.log", "Sharded linear layers: attn.qkv, attn.proj, mlp.fc1, mlp.fc2")

        if rank == 0:
            train_transform, eval_transform, _ = build_transforms(
                model,
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
            train_ds, val_ds, _ = build_datasets(
                dataset_name=args.dataset,
                data_root=args.data_root,
                train_transform=train_transform,
                eval_transform=eval_transform,
                download=args.download,
            )
            train_loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=(args.num_workers > 0),
                drop_last=False,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=args.batch_size * 2,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=(args.num_workers > 0),
                drop_last=False,
            )
        else:
            train_loader = None
            val_loader = None

        optimizer = create_optimizer(model, lr=args.lr, weight_decay=args.weight_decay, betas=betas, eps=args.opt_eps)
        scheduler = create_scheduler(
            optimizer,
            epochs=args.epochs,
            base_lr=args.lr,
            warmup_epochs=args.warmup_epochs,
            warmup_lr=args.warmup_lr,
            min_lr=args.min_lr,
        )
        use_mixup = mixup_active(args.mixup, args.cutmix, args.cutmix_minmax)
        mixup_fn = None
        if rank == 0 and use_mixup:
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

        criterion = create_train_criterion(use_mixup=use_mixup, label_smoothing=args.label_smoothing).to(device)

        metrics_csv = output_dir / "metrics.csv"
        train_log = output_dir / "train.log"

        best_val_top1 = -1.0
        best_epoch = -1

        for epoch in range(1, args.epochs + 1):
            scheduler.step(epoch - 1)

            train_loss, train_top1, num_samples, epoch_time_s, peak_mem_gb, local_gpu_utilization = (
                train_one_epoch_tensor_parallel(
                    model=model,
                    loader=train_loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    mixup_fn=mixup_fn,
                    scaler=scaler,
                    device=device,
                    use_amp=use_amp,
                    grad_accum_steps=args.grad_accum_steps,
                    rank=rank,
                    world_size=world_size,
                    epoch=epoch,
                    total_epochs=args.epochs,
                    log_interval=args.log_interval,
                    grad_clip=args.grad_clip,
                )
            )

            val_loss, val_top1, val_top5 = evaluate_tensor_parallel(
                model=model,
                loader=val_loader,
                device=device,
                criterion=nn.CrossEntropyLoss().to(device),
                rank=rank,
                use_amp=use_amp,
            )

            gpu_util_str = gather_gpu_utilization_across_ranks(
                local_device_utilization=local_gpu_utilization,
                rank=rank,
                world_size=world_size,
            )

            peak_mem_gb = reduce_peak_memory_max(peak_mem_gb, device=device)

            is_best = val_top1 > best_val_top1
            if is_best:
                best_val_top1 = val_top1
                best_epoch = epoch

            throughput = num_samples / max(epoch_time_s, 1e-8)

            metrics = EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                train_top1=train_top1,
                val_loss=val_loss,
                val_top1=val_top1,
                val_top5=val_top5,
                train_throughput_img_s=throughput,
                epoch_time_s=epoch_time_s,
                peak_mem_gb=peak_mem_gb,
                lr=float(optimizer.param_groups[0]["lr"]),
                gpu_utilization=gpu_util_str,
            )

            if rank == 0:
                write_metrics_row(metrics_csv, metrics)

                summary = (
                    f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | train_top1={train_top1:.2f}% | "
                    f"val_loss={val_loss:.4f} | val_top1={val_top1:.2f}% | val_top5={val_top5:.2f}% | "
                    f"throughput={throughput:.2f} img/s | lr={optimizer.param_groups[0]['lr']:.6g} | "
                    f"peak_mem={peak_mem_gb:.2f}GB | gpu_util={gpu_util_str}"
                )
                print(summary)
                log_line(train_log, summary)

            checkpoint = {
                "epoch": epoch,
                "rank": rank,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "args": vars(args),
                "best_val_top1": best_val_top1,
                "best_epoch": best_epoch,
            }
            save_checkpoint(output_dir / f"tp_rank{rank}_last.pt", checkpoint)

            if rank == 0 and is_best:
                save_checkpoint(output_dir / f"tp_rank{rank}_best.pt", checkpoint)

            if int(args.save_every) > 0 and epoch % int(args.save_every) == 0:
                save_checkpoint(output_dir / f"tp_rank{rank}_epoch{epoch:03d}.pt", checkpoint)

        dist.barrier()

        if rank == 0:
            print(f"\nTraining complete! Best val_top1: {best_val_top1:.2f}% at epoch {best_epoch}")

    finally:
        cleanup_dist()


if __name__ == "__main__":
    main()