import json
import os
import tempfile

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from data.assistments_dataset import AssistmentsDataset
from data.splits import IndexedSubset, split_temporal_by_student
from eval.evaluate_mirt import evaluate_mirt
from models.mirt import MIRTItemModule, MIRTModel


def _student_partition_subset(train_set, rank, world_size):
    base = train_set.dataset if hasattr(train_set, "dataset") else train_set
    idxs = np.asarray(train_set.idxs, dtype=np.int64)
    df = base.data.iloc[idxs]
    mask = (df["student_idx"].to_numpy() % world_size) == rank
    local_idxs = idxs[mask]
    return IndexedSubset(base, local_idxs)


def _setup_process(rank, world_size, backend, init_method):
    # On local single-machine runs (especially macOS), Gloo may need explicit loopback binding.
    if backend == "gloo" and "GLOO_SOCKET_IFNAME" not in os.environ:
        os.environ["GLOO_SOCKET_IFNAME"] = "lo0"
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )


def _cleanup_process():
    if dist.is_initialized():
        dist.destroy_process_group()


def _worker(rank, world_size, cfg):
    backend = cfg["backend"]
    _setup_process(rank, world_size, backend, cfg["init_method"])

    use_cuda = cfg["use_cuda"]
    if use_cuda:
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")

    dataset = AssistmentsDataset(cfg["data_path"], cache_path=cfg.get("cache_path"))

    if cfg["max_students"] > 0:
        rng = np.random.default_rng(cfg["seed"])
        all_students = np.array(dataset.data["student_idx"].unique())
        n_take = min(cfg["max_students"], len(all_students))
        keep_students = set(rng.choice(all_students, size=n_take, replace=False).tolist())
        keep_rows = dataset.data.index[dataset.data["student_idx"].isin(keep_students)].to_numpy()
        dataset = IndexedSubset(dataset, keep_rows)

    train_set, test_set = split_temporal_by_student(dataset, train_ratio=cfg["train_ratio"])
    local_train_set = _student_partition_subset(train_set, rank, world_size)
    base = train_set.dataset if hasattr(train_set, "dataset") else train_set

    num_students = len(base.students)
    num_items = len(base.items)

    student_emb = torch.nn.Embedding(num_students, cfg["emb_dim"]).to(device)
    item_module = MIRTItemModule(num_items, cfg["emb_dim"]).to(device)
    ddp_item = DDP(item_module, device_ids=[rank] if use_cuda else None)

    opt_student = torch.optim.Adam(student_emb.parameters(), lr=cfg["lr"])
    opt_item = torch.optim.Adam(ddp_item.parameters(), lr=cfg["lr"])
    criterion = torch.nn.BCELoss()

    pin_memory = use_cuda
    loader = DataLoader(
        local_train_set,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=pin_memory,
    )

    for epoch in range(cfg["epochs"]):
        losses = []
        for batch in loader:
            student_idx = batch["student_idx"].long().to(device, non_blocking=pin_memory)
            item_idx = batch["item_idx"].long().to(device, non_blocking=pin_memory)
            correct = batch["correct"].float().to(device, non_blocking=pin_memory)

            theta = student_emb(student_idx)
            pred = ddp_item(theta, item_idx)
            loss = criterion(pred, correct)
            if cfg["student_l2"] > 0.0:
                loss = loss + cfg["student_l2"] * theta.pow(2).sum(dim=1).mean()

            opt_student.zero_grad()
            opt_item.zero_grad()
            loss.backward()
            opt_student.step()
            opt_item.step()
            losses.append(loss.item())

        local_mean = (
            torch.tensor(float(np.mean(losses)), device=device) if losses else torch.tensor(0.0, device=device)
        )
        dist.all_reduce(local_mean, op=dist.ReduceOp.SUM)
        if rank == 0:
            global_mean = local_mean.item() / world_size
            print(f"[DDP] Epoch {epoch+1}: mean loss = {global_mean:.4f}")

    # Reconstruct full student embedding on rank0 (students partitioned by modulo rank).
    local_weight = student_emb.weight.detach().cpu()
    gathered = [torch.zeros_like(local_weight) for _ in range(world_size)]
    dist.all_gather(gathered, local_weight)

    if rank == 0:
        merged_student = local_weight.clone()
        student_ids = torch.arange(num_students)
        for r in range(world_size):
            mask = (student_ids % world_size) == r
            merged_student[mask] = gathered[r][mask]

        model = MIRTModel(num_students, num_items, emb_dim=cfg["emb_dim"]).to(device)
        with torch.no_grad():
            model.student_emb.weight.copy_(merged_student.to(device))
            model.item_emb.weight.copy_(ddp_item.module.item_emb.weight.detach())
            model.item_bias.weight.copy_(ddp_item.module.item_bias.weight.detach())

        train_metrics = evaluate_mirt(
            model,
            train_set,
            batch_size=cfg["eval_batch_size"],
            device=device,
            num_workers=cfg["num_workers"],
        )
        test_metrics = evaluate_mirt(
            model,
            test_set,
            batch_size=cfg["eval_batch_size"],
            device=device,
            num_workers=cfg["num_workers"],
        )
        with open(cfg["metrics_path"], "w", encoding="utf-8") as f:
            json.dump({"train": train_metrics, "test": test_metrics}, f, indent=2)

    _cleanup_process()


def _single_process_train(cfg):
    dataset = AssistmentsDataset(cfg["data_path"], cache_path=cfg.get("cache_path"))

    if cfg["max_students"] > 0:
        rng = np.random.default_rng(cfg["seed"])
        all_students = np.array(dataset.data["student_idx"].unique())
        n_take = min(cfg["max_students"], len(all_students))
        keep_students = set(rng.choice(all_students, size=n_take, replace=False).tolist())
        keep_rows = dataset.data.index[dataset.data["student_idx"].isin(keep_students)].to_numpy()
        dataset = IndexedSubset(dataset, keep_rows)

    train_set, test_set = split_temporal_by_student(dataset, train_ratio=cfg["train_ratio"])
    base = train_set.dataset if hasattr(train_set, "dataset") else train_set
    num_students = len(base.students)
    num_items = len(base.items)
    device = torch.device("cuda" if cfg["use_cuda"] else "cpu")

    student_emb = torch.nn.Embedding(num_students, cfg["emb_dim"]).to(device)
    item_module = MIRTItemModule(num_items, cfg["emb_dim"]).to(device)
    opt_student = torch.optim.Adam(student_emb.parameters(), lr=cfg["lr"])
    opt_item = torch.optim.Adam(item_module.parameters(), lr=cfg["lr"])
    criterion = torch.nn.BCELoss()

    loader = DataLoader(
        train_set,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=cfg["use_cuda"],
    )

    for epoch in range(cfg["epochs"]):
        losses = []
        for batch in loader:
            student_idx = batch["student_idx"].long().to(device, non_blocking=cfg["use_cuda"])
            item_idx = batch["item_idx"].long().to(device, non_blocking=cfg["use_cuda"])
            correct = batch["correct"].float().to(device, non_blocking=cfg["use_cuda"])

            theta = student_emb(student_idx)
            pred = item_module(theta, item_idx)
            loss = criterion(pred, correct)
            if cfg["student_l2"] > 0.0:
                loss = loss + cfg["student_l2"] * theta.pow(2).sum(dim=1).mean()

            opt_student.zero_grad()
            opt_item.zero_grad()
            loss.backward()
            opt_student.step()
            opt_item.step()
            losses.append(loss.item())

        print(f"[DDP-single] Epoch {epoch+1}: mean loss = {float(np.mean(losses)):.4f}")

    model = MIRTModel(num_students, num_items, emb_dim=cfg["emb_dim"]).to(device)
    with torch.no_grad():
        model.student_emb.weight.copy_(student_emb.weight.detach())
        model.item_emb.weight.copy_(item_module.item_emb.weight.detach())
        model.item_bias.weight.copy_(item_module.item_bias.weight.detach())

    train_metrics = evaluate_mirt(
        model,
        train_set,
        batch_size=cfg["eval_batch_size"],
        device=device,
        num_workers=cfg["num_workers"],
    )
    test_metrics = evaluate_mirt(
        model,
        test_set,
        batch_size=cfg["eval_batch_size"],
        device=device,
        num_workers=cfg["num_workers"],
    )
    with open(cfg["metrics_path"], "w", encoding="utf-8") as f:
        json.dump({"train": train_metrics, "test": test_metrics}, f, indent=2)


def train_mirt_ddp(
    data_path,
    cache_path=None,
    emb_dim=2,
    batch_size=2048,
    lr=1e-2,
    epochs=5,
    train_ratio=0.75,
    student_l2=0.0,
    max_students=0,
    seed=42,
    num_workers=0,
    eval_batch_size=8192,
    world_size=None,
    backend=None,
):
    if world_size is None:
        gpu_count = torch.cuda.device_count()
        world_size = gpu_count if gpu_count > 0 else 1
    if world_size < 1:
        raise ValueError("world_size must be >= 1")

    use_cuda = torch.cuda.is_available() and world_size > 0 and world_size <= torch.cuda.device_count()
    if backend is None:
        backend = "nccl" if use_cuda else "gloo"

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        metrics_path = tmp.name

    cfg = {
        "data_path": data_path,
        "cache_path": cache_path,
        "emb_dim": emb_dim,
        "batch_size": batch_size,
        "lr": lr,
        "epochs": epochs,
        "train_ratio": train_ratio,
        "student_l2": student_l2,
        "max_students": max_students,
        "seed": seed,
        "num_workers": num_workers,
        "eval_batch_size": eval_batch_size,
        "world_size": world_size,
        "use_cuda": use_cuda,
        "backend": backend,
        "metrics_path": metrics_path,
    }
    if world_size == 1:
        _single_process_train(cfg)
    else:
        with tempfile.NamedTemporaryFile(prefix="ddp_init_", delete=False) as init_file:
            init_path = init_file.name
        cfg["init_method"] = f"file://{init_path}"
        mp.spawn(
            _worker,
            args=(world_size, cfg),
            nprocs=world_size,
            join=True,
        )
        if os.path.exists(init_path):
            os.remove(init_path)

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    os.remove(metrics_path)
    return metrics
