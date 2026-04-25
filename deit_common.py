from __future__ import annotations

import csv
import json
import math
import os
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import timm
from timm.data import Mixup, create_transform, resolve_model_data_config
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_top1: float
    val_loss: float
    val_top1: float
    val_top5: float
    train_throughput_img_s: float
    epoch_time_s: float
    peak_mem_gb: float
    lr: float = 0.0
    gpu_utilization: str = ""

    def as_dict(self) -> Dict[str, int | float | str]:
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "train_top1": self.train_top1,
            "val_loss": self.val_loss,
            "val_top1": self.val_top1,
            "val_top5": self.val_top5,
            "train_throughput_img_s": self.train_throughput_img_s,
            "epoch_time_s": self.epoch_time_s,
            "peak_mem_gb": self.peak_mem_gb,
            "lr": self.lr,
            "gpu_utilization": self.gpu_utilization,
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_cudnn() -> None:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def build_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
    img_size: int | None = None,
) -> nn.Module:
    model_kwargs = {
        "pretrained": pretrained,
        "num_classes": num_classes,
        "drop_rate": drop_rate,
        "drop_path_rate": drop_path_rate,
    }
    if img_size is not None:
        model_kwargs["img_size"] = img_size
    return timm.create_model(model_name, **model_kwargs)


def build_transforms(
    model: nn.Module,
    auto_augment: str | None,
    re_prob: float,
    color_jitter: float | None,
    input_size: int | None = None,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
    train_interpolation: str | None = None,
    crop_scale: Sequence[float] | None = None,
    crop_ratio: Sequence[float] | None = None,
    crop_pct: float | None = None,
    re_mode: str = "pixel",
    re_count: int = 1,
):
    data_config = dict(resolve_model_data_config(model))
    if input_size is not None:
        data_config["input_size"] = (3, int(input_size), int(input_size))
    if mean is not None:
        data_config["mean"] = tuple(float(x) for x in mean)
    if std is not None:
        data_config["std"] = tuple(float(x) for x in std)
    if train_interpolation is not None:
        data_config["interpolation"] = train_interpolation
    if crop_pct is not None:
        data_config["crop_pct"] = float(crop_pct)

    train_kwargs = dict(data_config)
    train_kwargs.update(
        {
            "is_training": True,
            "auto_augment": auto_augment,
            "re_prob": re_prob,
            "re_mode": re_mode,
            "re_count": int(re_count),
            "color_jitter": color_jitter,
        }
    )
    if crop_scale is not None:
        train_kwargs["scale"] = tuple(float(x) for x in crop_scale)
    if crop_ratio is not None:
        train_kwargs["ratio"] = tuple(float(x) for x in crop_ratio)

    eval_kwargs = dict(data_config)
    eval_kwargs["is_training"] = False

    train_transform = create_transform(**train_kwargs)
    eval_transform = create_transform(**eval_kwargs)
    return train_transform, eval_transform, data_config


class TransformView(torch.utils.data.Dataset):
    def __init__(self, base_dataset: torch.utils.data.Dataset, transform) -> None:
        self.base_dataset = base_dataset
        self.transform = transform
        self.classes = getattr(base_dataset, "classes", None)
        self.class_to_idx = getattr(base_dataset, "class_to_idx", None)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        image, target = self.base_dataset[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def build_datasets(
    dataset_name: str,
    data_root: str,
    train_transform,
    eval_transform,
    download: bool,
):
    root = Path(data_root)

    if dataset_name == "food101":
        train_ds = datasets.Food101(root=root, split="train", transform=train_transform, download=download)
        val_ds = datasets.Food101(root=root, split="test", transform=eval_transform, download=download)
        num_classes = 101
    elif dataset_name == "cifar100":
        train_ds = datasets.CIFAR100(root=root, train=True, transform=train_transform, download=download)
        val_ds = datasets.CIFAR100(root=root, train=False, transform=eval_transform, download=download)
        num_classes = 100
    elif dataset_name == "flowers102":
        # Torchvision provides official train / val / test splits. We combine train+val for training
        # only when users explicitly point to a local copy or use download=True.
        train_part = datasets.Flowers102(root=root, split="train", transform=None, download=download)
        val_part = datasets.Flowers102(root=root, split="val", transform=None, download=download)
        test_part = datasets.Flowers102(root=root, split="test", transform=None, download=download)
        concat_train = torch.utils.data.ConcatDataset([
            TransformView(train_part, train_transform),
            TransformView(val_part, train_transform),
        ])
        train_ds = concat_train
        val_ds = TransformView(test_part, eval_transform)
        num_classes = 102
    elif dataset_name == "imagefolder":
        train_dir = root / "train"
        val_dir = root / "val"
        if not train_dir.exists() or not val_dir.exists():
            raise FileNotFoundError(
                f"Expected ImageFolder layout with '{train_dir}' and '{val_dir}'."
            )
        train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
        val_ds = datasets.ImageFolder(val_dir, transform=eval_transform)
        num_classes = len(train_ds.classes)
    else:
        raise ValueError(
            f"Unsupported dataset='{dataset_name}'. Choose from food101, cifar100, flowers102, imagefolder."
        )

    return train_ds, val_ds, num_classes


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    @property
    def avg(self) -> float:
        return 0.0 if self.count == 0 else self.sum / self.count

    def update(self, value: float, n: int = 1) -> None:
        self.sum += value * n
        self.count += n


def resolve_model_output(output: torch.Tensor | Sequence[torch.Tensor] | Dict[str, torch.Tensor]) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, dict):
        if "logits" in output and torch.is_tensor(output["logits"]):
            return output["logits"]
        tensor_values = [value for value in output.values() if torch.is_tensor(value)]
        if not tensor_values:
            raise TypeError("Model output dict does not contain tensor logits.")
        if len(tensor_values) == 1:
            return tensor_values[0]
        return sum(tensor_values) / len(tensor_values)
    if isinstance(output, (tuple, list)):
        tensor_values = [value for value in output if torch.is_tensor(value)]
        if not tensor_values:
            raise TypeError("Model output tuple/list does not contain tensor logits.")
        if len(tensor_values) == 1:
            return tensor_values[0]
        if len(tensor_values) == 2 and tensor_values[0].shape == tensor_values[1].shape:
            return (tensor_values[0] + tensor_values[1]) / 2.0
        return tensor_values[0]
    raise TypeError(f"Unsupported model output type: {type(output)!r}")


def compute_topk(logits: torch.Tensor, target: torch.Tensor, topk: Sequence[int] = (1, 5)) -> List[float]:
    if logits.ndim != 2:
        raise ValueError(f"Expected 2D logits, got shape={tuple(logits.shape)}")
    max_k = min(max(topk), logits.shape[1])
    _, pred = logits.topk(max_k, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    out: List[float] = []
    for k in topk:
        k_eff = min(k, logits.shape[1])
        correct_k = correct[:k_eff].reshape(-1).float().sum(0)
        out.append(float(correct_k.item() * 100.0 / target.size(0)))
    return out


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    use_amp: bool,
) -> Tuple[float, float, float]:
    model.eval()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    for images, target in loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.amp.autocast(device_type='cuda', enabled=use_amp):
            logits = resolve_model_output(model(images))
            loss = criterion(logits, target)
        top1, top5 = compute_topk(logits, target, topk=(1, 5))
        batch_size = images.size(0)
        loss_meter.update(float(loss.item()), batch_size)
        top1_meter.update(top1, batch_size)
        top5_meter.update(top5, batch_size)

    return loss_meter.avg, top1_meter.avg, top5_meter.avg


def get_weight_decay_param_groups(model: nn.Module, weight_decay: float):
    no_decay = set()
    if hasattr(model, "no_weight_decay") and callable(model.no_weight_decay):
        try:
            no_decay = set(model.no_weight_decay())
        except Exception:
            no_decay = set()

    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias") or name in no_decay:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": no_decay_params, "weight_decay": 0.0},
        {"params": decay_params, "weight_decay": float(weight_decay)},
    ]


def create_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> torch.optim.Optimizer:
    param_groups = get_weight_decay_param_groups(model, weight_decay=weight_decay)
    return torch.optim.AdamW(param_groups, lr=lr, betas=betas, eps=eps)


class WarmupCosineScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        base_lr: float | None = None,
        warmup_epochs: int = 0,
        warmup_lr: float = 1e-6,
        min_lr: float = 0.0,
    ) -> None:
        self.optimizer = optimizer
        self.epochs = int(epochs)
        self.warmup_epochs = int(warmup_epochs)
        self.warmup_lr = float(warmup_lr)
        self.min_lr = float(min_lr)
        self.base_lr = float(base_lr if base_lr is not None else optimizer.param_groups[0]["lr"])
        self.last_epoch = -1

    def _lr_at_epoch(self, epoch: int) -> float:
        epoch = int(epoch)
        if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
            if self.warmup_epochs == 1:
                return self.base_lr
            alpha = epoch / max(self.warmup_epochs - 1, 1)
            return self.warmup_lr + alpha * (self.base_lr - self.warmup_lr)

        if self.epochs <= self.warmup_epochs:
            return self.min_lr

        progress = (epoch - self.warmup_epochs) / max(self.epochs - self.warmup_epochs - 1, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + cosine * (self.base_lr - self.min_lr)

    def step(self, epoch: int | None = None) -> float:
        if epoch is None:
            epoch = self.last_epoch + 1
        lr = self._lr_at_epoch(epoch)
        for param_group in self.optimizer.param_groups:
            lr_scale = float(param_group.get("lr_scale", 1.0))
            param_group["lr"] = lr * lr_scale
        self.last_epoch = int(epoch)
        return lr

    def get_last_lr(self) -> List[float]:
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def state_dict(self) -> Dict[str, float | int]:
        return {
            "epochs": self.epochs,
            "base_lr": self.base_lr,
            "warmup_epochs": self.warmup_epochs,
            "warmup_lr": self.warmup_lr,
            "min_lr": self.min_lr,
            "last_epoch": self.last_epoch,
        }

    def load_state_dict(self, state_dict: Dict[str, float | int]) -> None:
        self.epochs = int(state_dict.get("epochs", self.epochs))
        self.base_lr = float(state_dict.get("base_lr", self.base_lr))
        self.warmup_epochs = int(state_dict.get("warmup_epochs", self.warmup_epochs))
        self.warmup_lr = float(state_dict.get("warmup_lr", self.warmup_lr))
        self.min_lr = float(state_dict.get("min_lr", self.min_lr))
        self.last_epoch = int(state_dict.get("last_epoch", -1))


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    base_lr: float | None = None,
    warmup_epochs: int = 0,
    warmup_lr: float = 1e-6,
    min_lr: float = 0.0,
):
    return WarmupCosineScheduler(
        optimizer=optimizer,
        epochs=epochs,
        base_lr=base_lr,
        warmup_epochs=warmup_epochs,
        warmup_lr=warmup_lr,
        min_lr=min_lr,
    )


def mixup_active(mixup: float, cutmix: float, cutmix_minmax: Sequence[float] | None = None) -> bool:
    return bool(mixup > 0.0 or cutmix > 0.0 or cutmix_minmax is not None)


def create_mixup_fn(
    mixup: float,
    cutmix: float,
    cutmix_minmax: Sequence[float] | None,
    mixup_prob: float,
    mixup_switch_prob: float,
    mixup_mode: str,
    label_smoothing: float,
    num_classes: int,
):
    if not mixup_active(mixup=mixup, cutmix=cutmix, cutmix_minmax=cutmix_minmax):
        return None
    return Mixup(
        mixup_alpha=float(mixup),
        cutmix_alpha=float(cutmix),
        cutmix_minmax=None if cutmix_minmax is None else tuple(float(x) for x in cutmix_minmax),
        prob=float(mixup_prob),
        switch_prob=float(mixup_switch_prob),
        mode=str(mixup_mode),
        label_smoothing=float(label_smoothing),
        num_classes=int(num_classes),
    )


def create_train_criterion(use_mixup: bool, label_smoothing: float) -> nn.Module:
    if use_mixup:
        return SoftTargetCrossEntropy()
    if label_smoothing > 0.0:
        return LabelSmoothingCrossEntropy(smoothing=float(label_smoothing))
    return nn.CrossEntropyLoss()


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def _visible_cuda_device_tokens() -> List[str]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        return []
    return [token.strip() for token in raw.split(",") if token.strip()]


def _cuda_index_to_monitor_selector(cuda_index: int) -> str:
    visible_tokens = _visible_cuda_device_tokens()
    if 0 <= int(cuda_index) < len(visible_tokens):
        return visible_tokens[int(cuda_index)]
    return str(int(cuda_index))


def _query_gpu_utilization_with_nvml(cuda_indices: Sequence[int]) -> Dict[int, float]:
    import pynvml

    if not getattr(_query_gpu_utilization_with_nvml, "_initialized", False):
        pynvml.nvmlInit()
        setattr(_query_gpu_utilization_with_nvml, "_initialized", True)

    handles = getattr(_query_gpu_utilization_with_nvml, "_handles", {})
    utilization: Dict[int, float] = {}
    for cuda_index in cuda_indices:
        if cuda_index not in handles:
            selector = _cuda_index_to_monitor_selector(cuda_index)
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(int(selector))
            except ValueError:
                handle = pynvml.nvmlDeviceGetHandleByUUID(selector.encode("utf-8"))
            handles[cuda_index] = handle
        utilization[cuda_index] = float(pynvml.nvmlDeviceGetUtilizationRates(handles[cuda_index]).gpu)

    setattr(_query_gpu_utilization_with_nvml, "_handles", handles)
    return utilization


def _query_gpu_utilization_with_nvidia_smi(cuda_indices: Sequence[int]) -> Dict[int, float]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    values_by_selector: Dict[str, float] = {}
    for line in output.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        gpu_index, gpu_uuid, gpu_util = parts[:3]
        try:
            utilization = float(gpu_util)
        except ValueError:
            continue
        values_by_selector[gpu_index] = utilization
        values_by_selector[gpu_uuid] = utilization

    utilization: Dict[int, float] = {}
    for cuda_index in cuda_indices:
        selector = _cuda_index_to_monitor_selector(cuda_index)
        if selector in values_by_selector:
            utilization[cuda_index] = values_by_selector[selector]
        elif str(cuda_index) in values_by_selector:
            utilization[cuda_index] = values_by_selector[str(cuda_index)]
    return utilization


def query_gpu_utilization(cuda_indices: Sequence[int]) -> Dict[int, float]:
    if not torch.cuda.is_available():
        return {}

    normalized_indices = sorted({int(index) for index in cuda_indices})
    if not normalized_indices:
        return {}

    backend = getattr(query_gpu_utilization, "_backend", None)
    if backend != "unavailable":
        if backend in (None, "nvml"):
            try:
                utilization = _query_gpu_utilization_with_nvml(normalized_indices)
                setattr(query_gpu_utilization, "_backend", "nvml")
                return utilization
            except Exception:
                setattr(query_gpu_utilization, "_backend", "nvidia-smi")
        if getattr(query_gpu_utilization, "_backend", None) == "nvidia-smi":
            try:
                utilization = _query_gpu_utilization_with_nvidia_smi(normalized_indices)
                setattr(query_gpu_utilization, "_backend", "nvidia-smi")
                return utilization
            except Exception:
                setattr(query_gpu_utilization, "_backend", "unavailable")
    return {}


def format_gpu_utilization(device_utilization: Sequence[Tuple[str, float]]) -> str:
    if not device_utilization:
        return ""
    return json.dumps({str(device): round(float(utilization), 2) for device, utilization in device_utilization})


class GpuUtilizationMonitor:
    def __init__(self, devices: Sequence[torch.device], sample_interval_s: float = 0.5) -> None:
        self.cuda_indices: List[int] = []
        self.device_names: Dict[int, str] = {}
        for device in devices:
            if device.type != "cuda":
                continue
            index = torch.cuda.current_device() if device.index is None else int(device.index)
            if index not in self.device_names:
                self.cuda_indices.append(index)
                self.device_names[index] = str(torch.device("cuda", index))

        self.sample_interval_s = max(0.0, float(sample_interval_s))
        self.meters = {index: AverageMeter() for index in self.cuda_indices}
        self.last_sample_time = 0.0

    def maybe_sample(self, force: bool = False) -> None:
        if not self.cuda_indices:
            return

        now = time.perf_counter()
        if not force and self.last_sample_time > 0.0 and (now - self.last_sample_time) < self.sample_interval_s:
            return

        utilization = query_gpu_utilization(self.cuda_indices)
        if not utilization:
            return

        self.last_sample_time = now
        for cuda_index, value in utilization.items():
            self.meters[cuda_index].update(float(value), 1)

    def as_device_utilization(self) -> List[Tuple[str, float]]:
        return [
            (self.device_names[cuda_index], self.meters[cuda_index].avg)
            for cuda_index in self.cuda_indices
            if self.meters[cuda_index].count > 0
        ]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    grad_accum_steps: int,
    mixup_fn=None,
    grad_clip: float | None = None,
) -> Tuple[float, float, int, float, float, List[Tuple[str, float]]]:
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    gpu_monitor = GpuUtilizationMonitor([device])
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    num_samples = 0
    num_top1_batches = 0
    optimizer.zero_grad(set_to_none=True)
    start_time = time.perf_counter()

    for step, (images, target) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        batch_size = images.size(0)
        hard_target = target
        if mixup_fn is not None:
            images, target = mixup_fn(images, target)

        with torch.amp.autocast(device_type='cuda', enabled=use_amp):
            logits = resolve_model_output(model(images))
            loss = criterion(logits, target)
            loss_for_backward = loss / grad_accum_steps

        scaler.scale(loss_for_backward).backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if mixup_fn is None:
            top1, _ = compute_topk(logits.detach(), hard_target, topk=(1, 5))
            top1_meter.update(top1, batch_size)
            num_top1_batches += batch_size

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
    top1_value = float("nan") if num_top1_batches == 0 else top1_meter.avg
    return loss_meter.avg, top1_value, num_samples, epoch_time_s, peak_mem_gb, gpu_monitor.as_device_utilization()


def write_metrics_row(csv_path: Path, metrics: EpochMetrics) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    row = metrics.as_dict()
    fieldnames = list(row.keys())

    rewrite_existing_rows: List[Dict[str, str]] | None = None
    if csv_path.exists():
        try:
            with csv_path.open(newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames != fieldnames:
                    rewrite_existing_rows = list(reader)
        except Exception:
            rewrite_existing_rows = []

    write_header = (not csv_path.exists()) or (rewrite_existing_rows is not None)
    mode = "w" if write_header else "a"
    with csv_path.open(mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
            if rewrite_existing_rows:
                for existing_row in rewrite_existing_rows:
                    writer.writerow({name: existing_row.get(name, "") for name in fieldnames})
        writer.writerow(row)


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def save_checkpoint(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def format_run_summary(
    args_dict: Dict,
    best_top1: float,
    best_epoch: int,
    total_time_s: float,
    time_to_target_s: float | None,
) -> Dict:
    return {
        "args": args_dict,
        "best_val_top1": best_top1,
        "best_epoch": best_epoch,
        "total_time_s": total_time_s,
        "total_time_min": total_time_s / 60.0,
        "time_to_target_s": time_to_target_s,
        "time_to_target_min": None if time_to_target_s is None else time_to_target_s / 60.0,
    }


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