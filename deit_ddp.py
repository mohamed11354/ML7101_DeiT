from __future__ import annotations

import argparse
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

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
    evaluate,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Naive PyTorch DDP DeiT training script.")
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
    parser.add_argument("--batch-size", type=int, default=128, help="Per-GPU batch size.")
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
    parser.add_argument("--output-dir", type=str, default="./outputs_ddp")

    # DDP-specific.
    parser.add_argument("--dist-backend", type=str, default="nccl")
    parser.add_argument("--dist-url", type=str, default="env://")
    parser.add_argument("--find-unused-parameters", action="store_true")
    parser.add_argument("--sync-bn", action="store_true")
    parser.add_argument("--broadcast-buffers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bucket-cap-mb", type=int, default=25)
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def setup_ddp(args: argparse.Namespace) -> tuple[int, int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("DDP training requires CUDA devices.")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 2:
        raise RuntimeError(
            "WORLD_SIZE < 2. Launch with: torchrun --standalone --nproc_per_node=2 deit_kaggle_ddp.py ..."
        )
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url)
    return rank, world_size, local_rank, torch.device(f"cuda:{local_rank}")


def cleanup_ddp() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def load_checkpoint_if_requested(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.cuda.amp.GradScaler,
    resume_path: str | None,
    device: torch.device,
) -> tuple[int, float, int]:
    if not resume_path:
        return 1, -1.0, -1

    checkpoint = torch.load(resume_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        move_optimizer_state_to_device(optimizer, device)
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


def maybe_download_dataset_once(
    *,
    args: argparse.Namespace,
    train_transform,
    eval_transform,
    rank: int,
) -> None:
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


def reduce_average(local_avg: float, local_count: int, device: torch.device) -> float:
    if math.isnan(local_avg):
        total = torch.tensor([0.0, 0.0], device=device, dtype=torch.float64)
    else:
        total = torch.tensor(
            [float(local_avg) * float(local_count), float(local_count)],
            device=device,
            dtype=torch.float64,
        )
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    if total[1].item() <= 0.0:
        return float("nan")
    return float((total[0] / total[1]).item())


def reduce_epoch_stats(num_samples: int, epoch_time_s: float, peak_mem_gb: float, device: torch.device):
    samples_tensor = torch.tensor([float(num_samples)], device=device, dtype=torch.float64)
    time_tensor = torch.tensor([float(epoch_time_s)], device=device, dtype=torch.float64)
    mem_tensor = torch.tensor([float(peak_mem_gb)], device=device, dtype=torch.float64)
    dist.all_reduce(samples_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(time_tensor, op=dist.ReduceOp.MAX)
    dist.all_reduce(mem_tensor, op=dist.ReduceOp.MAX)
    return int(samples_tensor.item()), float(time_tensor.item()), float(mem_tensor.item())


def train_one_epoch_ddp(
    model: DDP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    grad_accum_steps: int,
    mixup_fn=None,
    grad_clip: float | None = None,
) -> tuple[float, float, int, float, float, list[tuple[str, float]]]:
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    gpu_monitor = GpuUtilizationMonitor([device])
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    num_samples = 0
    num_top1_samples = 0
    optimizer.zero_grad(set_to_none=True)
    start_time = time.perf_counter()

    for step, (images, target) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        batch_size = images.size(0)
        hard_target = target
        if mixup_fn is not None:
            images, target = mixup_fn(images, target)

        should_step = ((step + 1) % grad_accum_steps == 0) or ((step + 1) == len(loader))
        sync_context = nullcontext() if should_step else model.no_sync()

        with sync_context:
            with torch.amp.autocast(enabled=use_amp):
                logits = resolve_model_output(model(images))
                loss = criterion(logits, target)
                loss_for_backward = loss / grad_accum_steps
            scaler.scale(loss_for_backward).backward()

        if should_step:
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if mixup_fn is None:
            top1, _ = compute_topk(logits.detach(), hard_target, topk=(1, 5))
            top1_meter.update(top1, batch_size)
            num_top1_samples += batch_size

        loss_meter.update(float(loss.item()), batch_size)
        num_samples += batch_size
        gpu_monitor.maybe_sample(force=(step == 0))

    gpu_monitor.maybe_sample(force=True)
    epoch_time_s = time.perf_counter() - start_time
    peak_mem_gb = (
        float(torch.cuda.max_memory_allocated(device) / (1024 ** 3))
        if device.type == "cuda"
        else 0.0
    )
    top1_value = float("nan") if num_top1_samples == 0 else top1_meter.avg
    return loss_meter.avg, top1_value, num_samples, epoch_time_s, peak_mem_gb, gpu_monitor.as_device_utilization()


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank, device = setup_ddp(args)

    try:
        set_seed(args.seed + rank)
        setup_cudnn()
        use_amp = bool(args.amp and device.type == "cuda")
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

        maybe_download_dataset_once(
            args=args,
            train_transform=train_transform,
            eval_transform=eval_transform,
            rank=rank,
        )
        train_ds, val_ds, num_classes = build_datasets(
            dataset_name=args.dataset,
            data_root=args.data_root,
            train_transform=train_transform,
            eval_transform=eval_transform,
            download=False,
        )

        model = build_model(
            args.model,
            num_classes=num_classes,
            pretrained=not args.no_pretrained,
            drop_rate=args.drop_rate,
            drop_path_rate=args.drop_path,
            img_size=args.input_size,
        ).to(device)
        if args.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

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
        train_ds, val_ds, num_classes = build_datasets(
            dataset_name=args.dataset,
            data_root=args.data_root,
            train_transform=train_transform,
            eval_transform=eval_transform,
            download=False,
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
            device=device,
        )

        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=args.find_unused_parameters,
            broadcast_buffers=args.broadcast_buffers,
            bucket_cap_mb=args.bucket_cap_mb,
        )

        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=(args.num_workers > 0),
        )
        if is_main_process():
            val_loader = DataLoader(
                val_ds,
                batch_size=max(1, args.batch_size * 2),
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=(args.num_workers > 0),
            )
        else:
            val_loader = None

        output_dir = Path(args.output_dir)
        metrics_csv = output_dir / "metrics.csv"
        best_ckpt = output_dir / "best.pt"
        last_ckpt = output_dir / "last.pt"
        summary_json = output_dir / "summary.json"

        time_to_target_s = None
        run_start = time.perf_counter()

        for epoch in range(start_epoch, args.epochs + 1):
            train_sampler.set_epoch(epoch)
            scheduler.step(epoch - 1)

            local_train_loss, local_train_top1, local_num_samples, local_epoch_time_s, local_peak_mem_gb, local_gpu_utilization = train_one_epoch_ddp(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=train_criterion,
                scaler=scaler,
                device=device,
                use_amp=use_amp,
                grad_accum_steps=args.grad_accum_steps,
                mixup_fn=mixup_fn,
                grad_clip=args.grad_clip,
            )

            train_loss = reduce_average(local_train_loss, local_num_samples, device)
            train_top1 = reduce_average(
                local_train_top1,
                0 if math.isnan(local_train_top1) else local_num_samples,
                device,
            )
            total_num_samples, epoch_time_s, peak_mem_gb = reduce_epoch_stats(
                num_samples=local_num_samples,
                epoch_time_s=local_epoch_time_s,
                peak_mem_gb=local_peak_mem_gb,
                device=device,
            )
            gathered_gpu_utilization = [None for _ in range(world_size)]
            dist.all_gather_object(gathered_gpu_utilization, local_gpu_utilization)
            gpu_utilization = format_gpu_utilization(
                [
                    (f"rank{rank_idx}/{device_name}", value)
                    for rank_idx, per_rank_utilization in enumerate(gathered_gpu_utilization)
                    for device_name, value in per_rank_utilization
                ]
            )
            train_throughput = total_num_samples / max(epoch_time_s, 1e-8)

            dist.barrier()
            if is_main_process():
                assert val_loader is not None
                val_loss, val_top1, val_top5 = evaluate(
                    model=model.module,
                    loader=val_loader,
                    device=device,
                    criterion=eval_criterion,
                    use_amp=use_amp,
                )
            else:
                val_loss, val_top1, val_top5 = 0.0, 0.0, 0.0
            dist.barrier()

            if is_main_process():
                if val_top1 > best_top1:
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
                write_metrics_row(metrics_csv, metrics)

                checkpoint = {
                    "epoch": epoch,
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_val_top1": best_top1,
                    "best_epoch": best_epoch,
                    "world_size": world_size,
                    "args": vars(args),
                }
                save_checkpoint(last_ckpt, checkpoint)
                if val_top1 >= best_top1:
                    save_checkpoint(best_ckpt, checkpoint)

                elapsed_s = time.perf_counter() - run_start
                if args.target_top1 is not None and time_to_target_s is None and val_top1 >= args.target_top1:
                    time_to_target_s = elapsed_s

                print(
                    f"Epoch {epoch:03d} | "
                    f"lr={current_lr(optimizer):.6g} | "
                    f"train_loss={train_loss:.4f} train_top1={train_top1:.2f} | "
                    f"val_loss={val_loss:.4f} val_top1={val_top1:.2f} val_top5={val_top5:.2f} | "
                    f"throughput={train_throughput:.2f} img/s | peak_mem={peak_mem_gb:.2f} GB | world_size={world_size}"
                )

            dist.barrier()

        if is_main_process():
            total_time_s = time.perf_counter() - run_start
            summary = format_run_summary(
                args_dict={**vars(args), "world_size": world_size},
                best_top1=best_top1,
                best_epoch=best_epoch,
                total_time_s=total_time_s,
                time_to_target_s=time_to_target_s,
            )
            save_json(summary_json, summary)
            print("Saved results to:", output_dir.resolve())
            print(summary)
    finally:
        cleanup_ddp()


if __name__ == "__main__":
    main()
