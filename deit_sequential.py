from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from deit_common import (
    EpochMetrics,
    build_datasets,
    build_model,
    build_transforms,
    create_mixup_fn,
    create_optimizer,
    create_scheduler,
    create_train_criterion,
    current_lr,
    evaluate,
    format_gpu_utilization,
    format_run_summary,
    save_checkpoint,
    save_json,
    set_seed,
    setup_cudnn,
    train_one_epoch,
    write_metrics_row,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sequential DeiT training script.")
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
    parser.add_argument("--output-dir", type=str, default="./outputs_seq")
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


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
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
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


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    setup_cudnn()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    train_ds, val_ds, num_classes = build_datasets(
        dataset_name=args.dataset,
        data_root=args.data_root,
        train_transform=train_transform,
        eval_transform=eval_transform,
        download=args.download,
    )

    model = build_model(
        args.model,
        num_classes=num_classes,
        pretrained=not args.no_pretrained,
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path,
        img_size=args.input_size,
    ).to(device)
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

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, args.batch_size * 2),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
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

    output_dir = Path(args.output_dir)
    metrics_csv = output_dir / "metrics.csv"
    best_ckpt = output_dir / "best.pt"
    last_ckpt = output_dir / "last.pt"
    summary_json = output_dir / "summary.json"

    time_to_target_s = None
    run_start = time.perf_counter()

    for epoch in range(start_epoch, args.epochs + 1):
        scheduler.step(epoch - 1)

        train_loss, train_top1, num_samples, epoch_time_s, peak_mem_gb, local_gpu_utilization = train_one_epoch(
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
        train_throughput = num_samples / max(epoch_time_s, 1e-8)
        val_loss, val_top1, val_top5 = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            criterion=eval_criterion,
            use_amp=use_amp,
        )

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
            gpu_utilization=format_gpu_utilization(local_gpu_utilization),
        )
        write_metrics_row(metrics_csv, metrics)

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_val_top1": best_top1,
            "best_epoch": best_epoch,
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
            f"throughput={train_throughput:.2f} img/s | peak_mem={peak_mem_gb:.2f} GB"
        )

    total_time_s = time.perf_counter() - run_start
    summary = format_run_summary(
        args_dict=vars(args),
        best_top1=best_top1,
        best_epoch=best_epoch,
        total_time_s=total_time_s,
        time_to_target_s=time_to_target_s,
    )
    save_json(summary_json, summary)
    print("Saved results to:", output_dir.resolve())
    print(summary)


if __name__ == "__main__":
    main()
