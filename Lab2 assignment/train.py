"""
train.py  --  Lab 2: train LeNet-5 / AlexNet / VGG16 on CIFAR-10 and CIFAR-100.

Examples
    python train.py --model lenet --dataset cifar10 --epochs 2     # quick test
    python train.py --model all --dataset all --epochs 30          # all 6 runs

For every (model, dataset) pair it saves, in ./results :
    <model>_<dataset>_history.json   per-epoch loss/accuracy/time
    <model>_<dataset>_curves.png     training vs validation curves
and finally summary.csv with test accuracy, top-5, accuracy drop, params, time.

The SAME settings are used for every run (see the argparse defaults), so that
differences in results come from the architecture only.
"""

import argparse
import copy
import csv
import json
import os
import random
import time

import matplotlib
matplotlib.use("Agg")                      # draw plots without a screen
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

from models import get_model, count_params

NUM_CLASSES = {"cifar10": 10, "cifar100": 100}
# per-channel mean / std of each training set (used to normalise the images)
STATS = {
    "cifar10":  ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
}


# ---------------------------------------------------------------- data ----
def get_loaders(dataset, batch_size, augment, seed, data_dir):
    mean, std = STATS[dataset]
    eval_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    if augment:
        train_tf = T.Compose([T.RandomCrop(32, padding=4),
                              T.RandomHorizontalFlip(),
                              T.ToTensor(), T.Normalize(mean, std)])
    else:
        train_tf = eval_tf

    cls = (torchvision.datasets.CIFAR10 if dataset == "cifar10"
           else torchvision.datasets.CIFAR100)
    # two copies of the training set: one with train transforms, one without,
    # so the validation images are never augmented
    full_train = cls(data_dir, train=True, download=True, transform=train_tf)
    full_val = cls(data_dir, train=True, download=True, transform=eval_tf)
    test_set = cls(data_dir, train=False, download=True, transform=eval_tf)

    # fixed 45,000 / 5,000 train / validation split (same for every model)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(full_train), generator=g).tolist()
    train_ds = Subset(full_train, perm[:45000])
    val_ds = Subset(full_val, perm[45000:])

    pin = torch.cuda.is_available()
    kw = dict(batch_size=batch_size, num_workers=2, pin_memory=pin)
    return (DataLoader(train_ds, shuffle=True, **kw),
            DataLoader(val_ds, shuffle=False, **kw),
            DataLoader(test_set, shuffle=False, **kw))


# ------------------------------------------------------------- helpers ----
def topk_correct(outputs, targets, k):
    """How many samples have the true label among the k highest scores."""
    k = min(k, outputs.size(1))
    _, pred = outputs.topk(k, dim=1)
    return pred.eq(targets.view(-1, 1)).any(dim=1).sum().item()


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass over `loader`. Trains if an optimizer is given, else evaluates.
    Returns (avg loss, top-1 %, top-5 %)."""
    training = optimizer is not None
    model.train(training)                    # switches dropout on/off
    total_loss, n, c1, c5 = 0.0, 0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)                  # 1. forward pass
            loss = criterion(outputs, labels)        # 2. how wrong are we?
            if training:
                optimizer.zero_grad()                # 3. clear old gradients
                loss.backward()                      # 4. backpropagation
                optimizer.step()                     # 5. update the weights
            bs = labels.size(0)
            total_loss += loss.item() * bs
            n += bs
            c1 += topk_correct(outputs, labels, 1)
            c5 += topk_correct(outputs, labels, 5)
    return total_loss / n, 100.0 * c1 / n, 100.0 * c5 / n


def plot_curves(hist, title, path):
    ep = range(1, len(hist["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, hist["train_loss"], label="train")
    ax[0].plot(ep, hist["val_loss"], label="validation")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend()
    ax[1].plot(ep, hist["train_acc"], label="train")
    ax[1].plot(ep, hist["val_acc"], label="validation")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("top-1 accuracy (%)")
    ax[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------- experiment ----
def run_experiment(model_name, dataset, args, device):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    tag = f"{model_name}_{dataset}"
    n_cls = NUM_CLASSES[dataset]
    print(f"\n{'=' * 60}\n{tag}\n{'=' * 60}")

    train_loader, val_loader, test_loader = get_loaders(
        dataset, args.batch_size, args.augment, args.seed, args.data_dir)

    kwargs = {"use_bn": True} if (model_name == "vgg16" and args.vgg_bn) else {}
    model = get_model(model_name, n_cls, **kwargs).to(device)
    n_params = count_params(model)
    print(f"trainable parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    hist = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc",
                            "val_top5", "epoch_time"]}
    best_val, best_state = -1.0, None

    for epoch in range(1, args.epochs + 1):
        sync(device); t0 = time.time()
        tr_loss, tr_acc, _ = run_epoch(model, train_loader, criterion,
                                       device, optimizer)
        sync(device); dt = time.time() - t0          # training time only
        va_loss, va_acc, va_5 = run_epoch(model, val_loader, criterion, device)

        for k, v in zip(hist, [tr_loss, tr_acc, va_loss, va_acc, va_5, dt]):
            hist[k].append(v)
        if va_acc > best_val:                        # remember best weights
            best_val, best_state = va_acc, copy.deepcopy(model.state_dict())

        print(f"epoch {epoch:3d}/{args.epochs} | "
              f"train loss {tr_loss:.3f} acc {tr_acc:5.2f} | "
              f"val loss {va_loss:.3f} acc {va_acc:5.2f} | {dt:5.1f}s")

    chance = 100.0 / n_cls
    if best_val < chance + 2:
        print(f"WARNING: validation accuracy ~ chance ({chance:.0f}%). "
              f"The network did not learn. See the troubleshooting notes.")

    # final test evaluation, using the weights with the best validation acc
    model.load_state_dict(best_state)
    _, test_top1, test_top5 = run_epoch(model, test_loader, criterion, device)
    print(f"TEST  top-1 {test_top1:.2f}%   top-5 {test_top5:.2f}%")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{tag}_history.json"), "w") as f:
        json.dump(hist, f, indent=2)
    plot_curves(hist, tag, os.path.join(args.out, f"{tag}_curves.png"))

    return {
        "model": model_name, "dataset": dataset, "params": n_params,
        "test_top1": round(test_top1, 2), "test_top5": round(test_top5, 2),
        "accuracy_drop": round(100.0 - test_top1, 2),
        "final_train_acc": round(hist["train_acc"][-1], 2),
        "final_val_acc": round(hist["val_acc"][-1], 2),
        "train_val_gap": round(hist["train_acc"][-1] - hist["val_acc"][-1], 2),
        "avg_epoch_time_s": round(sum(hist["epoch_time"]) /
                                  len(hist["epoch_time"]), 2),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="all",
                   choices=["lenet", "alexnet", "vgg16", "all"])
    p.add_argument("--dataset", default="all",
                   choices=["cifar10", "cifar100", "all"])
    # ---- settings kept identical for every model ----
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--augment", action="store_true",
                   help="random crop + horizontal flip (off by default)")
    p.add_argument("--seed", type=int, default=42)
    # ---- other ----
    p.add_argument("--vgg-bn", action="store_true",
                   help="add BatchNorm to VGG16 (only if it fails to learn)")
    p.add_argument("--out", default="results")
    p.add_argument("--data-dir", default="./data")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    models = ["lenet", "alexnet", "vgg16"] if args.model == "all" else [args.model]
    datasets = ["cifar10", "cifar100"] if args.dataset == "all" else [args.dataset]

    rows = [run_experiment(m, d, args, device) for d in datasets for m in models]

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("\nSaved summary to", os.path.join(args.out, "summary.csv"))


if __name__ == "__main__":
    main()
