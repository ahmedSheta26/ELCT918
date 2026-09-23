#!/usr/bin/env python3
"""
ELCT918 - Lab 1
Exhaustive multi-objective DSE of parameterizable MLPs on MNIST.

This is a single-file implementation of Task 1 + Task 2:
  1. Programmatic MNIST MLP generation.
  2. Fixed-budget MNIST training/evaluation.
  3. Analytical usage-cost calculation.
  4. Exhaustive exploration of the complete selected design space.
  5. Pareto-front extraction and cost-vs-accuracy-drop visualization.

No surrogate/response-surface exploration is implemented or required.

Default design space:
  n = 1, 2, 3 hidden layers
  m = 10, 20, 40, 80, 160, 200 nodes
  --space uniform   -> 3 * 6 = 18 configurations
  --space per-layer -> 6^1 + 6^2 + 6^3 = 258 configurations

Default training settings follow the restricted-MNIST setup used by the
assignment/reference setup: SGD, batch size 200, learning rate 0.1,
momentum 0, 10 epochs, ReLU.

Examples
--------
Pipeline-only verification (no PyTorch/MNIST training):
    python run_dse.py --mock --space uniform

Actual exhaustive run over the 18 uniform configurations:
    python run_dse.py --space uniform

Actual exhaustive run over the 258 independent-per-layer configurations:
    python run_dse.py --space per-layer

For a short real functionality check before the full run:
    python run_dse.py --space uniform --n-min 1 --n-max 1 \
        --m-values 10 20 --epochs 1

Every completed configuration is appended to the CSV immediately, so an
interrupted exhaustive run can be resumed with the same command.
"""

import argparse
import csv
import itertools
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


# =============================================================================
# Task 1: Parameterizable network generator
# =============================================================================

MNIST_INPUT_DIM = 28 * 28
MNIST_NUM_CLASSES = 10


def resolve_widths(
    num_layers: int,
    nodes_per_layer: Union[int, Sequence[int]]
) -> List[int]:
    """Turn (n, m) into an explicit list of n hidden-layer widths."""
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1")

    if isinstance(nodes_per_layer, (int, np.integer)):
        widths = [int(nodes_per_layer)] * num_layers
    else:
        widths = [int(w) for w in nodes_per_layer]
        if len(widths) != num_layers:
            raise ValueError(
                f"nodes_per_layer has {len(widths)} entries "
                f"but num_layers={num_layers}"
            )

    if any(w < 1 for w in widths):
        raise ValueError("every hidden layer needs at least 1 node")

    return widths


def build_model(
    num_layers: int,
    nodes_per_layer: Union[int, Sequence[int]],
    input_dim: int = MNIST_INPUT_DIM,
    num_classes: int = MNIST_NUM_CLASSES,
):
    """
    Build an MLP programmatically:
        784 -> hidden layers -> 10 logits

    The topology is generated in a loop. ReLU is used after each hidden
    layer. The final layer returns logits; CrossEntropyLoss applies the
    equivalent softmax internally.
    """
    import torch.nn as nn

    widths = resolve_widths(num_layers, nodes_per_layer)

    layers = [nn.Flatten()]
    in_features = input_dim

    for width in widths:
        layers.append(nn.Linear(in_features, width))
        layers.append(nn.ReLU())
        in_features = width

    layers.append(nn.Linear(in_features, num_classes))
    return nn.Sequential(*layers)


# =============================================================================
# Task 1: MNIST training and evaluation
# =============================================================================

@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 10
    batch_size: int = 200
    lr: float = 0.1
    momentum: float = 0.0
    seed: int = 0


def pick_device(name: str = "auto"):
    import torch

    if name != "auto":
        return torch.device(name)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_mnist(data_dir: str, device):
    """
    Download/load MNIST once, normalize it using standard MNIST statistics,
    and keep the tensors on the selected device.
    """
    from torchvision import datasets

    train_set = datasets.MNIST(data_dir, train=True, download=True)
    test_set = datasets.MNIST(data_dir, train=False, download=True)

    def prepare(ds):
        x = ds.data.float().div_(255.0).sub_(0.1307).div_(0.3081)
        return x.to(device), ds.targets.to(device)

    x_train, y_train = prepare(train_set)
    x_test, y_test = prepare(test_set)

    return x_train, y_train, x_test, y_test


def train_and_evaluate(
    widths: Sequence[int],
    data,
    cfg: TrainConfig,
    device,
) -> dict:
    """
    Generate, train and evaluate one network.

    Training settings are fixed by TrainConfig and are identical for every
    configuration; only the architecture widths vary.
    """
    import torch
    import torch.nn as nn

    x_train, y_train, x_test, y_test = data

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    generator = torch.Generator().manual_seed(cfg.seed)

    model = build_model(len(widths), list(widths)).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
    )
    loss_fn = nn.CrossEntropyLoss()

    start = time.time()

    n_train = x_train.shape[0]
    model.train()

    for _ in range(cfg.epochs):
        permutation = torch.randperm(n_train, generator=generator).to(device)

        for i in range(0, n_train, cfg.batch_size):
            idx = permutation[i:i + cfg.batch_size]

            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x_train[idx]), y_train[idx])
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        test_acc = (
            (model(x_test).argmax(dim=1) == y_test)
            .float()
            .mean()
            .item()
            * 100.0
        )

    return {
        "test_acc": test_acc,
        "train_time_s": time.time() - start,
    }


# =============================================================================
# Task 2, Step 1: Design-space definition
# =============================================================================

class DesignSpace:
    """
    Bounded exhaustive design space.

    uniform:
        one m is shared by all hidden layers.
        Number of configurations = n_count * m_count.

    per-layer:
        each hidden layer independently chooses one m value.
        Number of configurations = sum(m_count ** n) for n in [n_min,n_max].
    """

    def __init__(
        self,
        n_min: int = 1,
        n_max: int = 3,
        m_values: Sequence[int] = (10, 20, 40, 80, 160, 200),
        per_layer: bool = False,
    ):
        if n_min < 1 or n_max < n_min:
            raise ValueError("Require 1 <= n_min <= n_max")

        self.n_min = int(n_min)
        self.n_max = int(n_max)
        self.m_values = tuple(int(x) for x in m_values)
        self.per_layer = bool(per_layer)

        if not self.m_values:
            raise ValueError("m_values cannot be empty")
        if any(m < 1 for m in self.m_values):
            raise ValueError("all m_values must be >= 1")

    def __len__(self):
        if not self.per_layer:
            return (self.n_max - self.n_min + 1) * len(self.m_values)

        return sum(
            len(self.m_values) ** n
            for n in range(self.n_min, self.n_max + 1)
        )

    def all_configs(self) -> List[Tuple[int, ...]]:
        """Return every configuration exactly once."""
        configs = []

        for n in range(self.n_min, self.n_max + 1):
            if self.per_layer:
                for widths in itertools.product(self.m_values, repeat=n):
                    configs.append(tuple(widths))
            else:
                for m in self.m_values:
                    configs.append(tuple([m] * n))

        return configs


# =============================================================================
# Task 2, Step 2: Cost model
# =============================================================================

WEIGHT_UNIT_COST = 139.0
MULT_UNIT_COST = 1.0


def layer_dims(
    widths: Sequence[int],
    input_dim: int = MNIST_INPUT_DIM,
    num_classes: int = MNIST_NUM_CLASSES,
):
    dims = [input_dim, *widths, num_classes]
    return list(zip(dims[:-1], dims[1:]))


def count_weights(widths: Sequence[int], **kwargs) -> int:
    """
    Stored parameters = matrix weights + bias values for every FC layer.
    """
    return sum(
        i * j + j
        for i, j in layer_dims(widths, **kwargs)
    )


def count_mults(widths: Sequence[int], **kwargs) -> int:
    """Number of multiplications/MACs per inference."""
    return sum(
        i * j
        for i, j in layer_dims(widths, **kwargs)
    )


def usage_cost(widths: Sequence[int], **kwargs) -> float:
    """
    Reference-paper-style normalized usage cost:

        cost = (#weights * 139) + (#multiplications * 1)
    """
    return (
        count_weights(widths, **kwargs) * WEIGHT_UNIT_COST
        + count_mults(widths, **kwargs) * MULT_UNIT_COST
    )


# =============================================================================
# Task 2, Step 3: Pareto front and visualization
# =============================================================================

def dominates(a, b) -> bool:
    """True if point a dominates point b for minimization objectives."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return bool(np.all(a <= b) and np.any(a < b))


def pareto_mask(points) -> np.ndarray:
    """
    Boolean mask of non-dominated rows for two minimization objectives.
    """
    pts = np.asarray(points, dtype=float)
    keep = np.ones(len(pts), dtype=bool)

    for i in range(len(pts)):
        no_worse = np.all(pts <= pts[i], axis=1)
        better = np.any(pts < pts[i], axis=1)

        if np.any(no_worse & better):
            keep[i] = False

    return keep


def pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    """Return non-dominated rows sorted by increasing cost."""
    mask = pareto_mask(df[["cost", "acc_drop"]].values)
    return df[mask].sort_values("cost").reset_index(drop=True)


def plot_pareto(
    df: pd.DataFrame,
    out_path: str,
    title: str,
    annotate: bool = True,
    log_x: bool = True,
) -> None:
    """
    Generate the required 2-D cost vs. accuracy-drop plot.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mask = pareto_mask(df[["cost", "acc_drop"]].values)
    dominated = df[~mask]
    front = df[mask].sort_values("cost")

    fig, ax = plt.subplots(figsize=(10, 6.5))

    ax.scatter(
        dominated["cost"],
        dominated["acc_drop"],
        s=32,
        c="#b0b7c3",
        edgecolors="none",
        alpha=0.85,
        label=f"Dominated ({len(dominated)})",
    )

    ax.step(
        front["cost"],
        front["acc_drop"],
        where="post",
        color="#d62728",
        lw=1.3,
        alpha=0.8,
    )

    ax.scatter(
        front["cost"],
        front["acc_drop"],
        s=60,
        c="#d62728",
        edgecolors="black",
        linewidths=0.6,
        zorder=3,
        label=f"Pareto-optimal ({len(front)})",
    )

    if annotate:
        for _, row in front.iterrows():
            ax.annotate(
                row["widths"],
                (row["cost"], row["acc_drop"]),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=7,
            )

    if log_x:
        ax.set_xscale("log")

    ax.set_xlabel(
        "Usage cost  (#weights × 139 + #multiplications × 1, normalised)"
    )
    ax.set_ylabel("Accuracy drop (%)  =  100 - test accuracy")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# =============================================================================
# DSE execution / persistence
# =============================================================================

FIELDS = [
    "n",
    "widths",
    "n_weights",
    "n_mults",
    "cost",
    "test_acc",
    "acc_drop",
    "train_time_s",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    design = parser.add_argument_group("design space")
    design.add_argument(
        "--space",
        choices=["uniform", "per-layer"],
        default="uniform",
        help="uniform: shared m; per-layer: independent m for each layer",
    )
    design.add_argument("--n-min", type=int, default=1)
    design.add_argument("--n-max", type=int, default=3)
    design.add_argument(
        "--m-values",
        type=int,
        nargs="+",
        default=[10, 20, 40, 80, 160, 200],
        help="allowed hidden-layer widths",
    )

    training = parser.add_argument_group("fixed training settings")
    training.add_argument("--epochs", type=int, default=10)
    training.add_argument("--batch-size", type=int, default=200)
    training.add_argument("--lr", type=float, default=0.1)
    training.add_argument("--momentum", type=float, default=0.0)
    training.add_argument("--seed", type=int, default=0)
    training.add_argument(
        "--seeds",
        type=int,
        default=1,
        help="number of independent training seeds averaged per configuration",
    )

    misc = parser.add_argument_group("execution")
    misc.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    misc.add_argument("--data-dir", default="./data")
    misc.add_argument("--out", default="results")
    misc.add_argument(
        "--no-annotate",
        action="store_true",
        help="do not label Pareto points",
    )
    misc.add_argument(
        "--linear-x",
        action="store_true",
        help="use a linear cost axis instead of logarithmic",
    )
    misc.add_argument(
        "--mock",
        action="store_true",
        help="pipeline test only: no PyTorch/MNIST training",
    )

    return parser.parse_args()


def make_train_fn(args):
    """
    Return a function cfg -> {'test_acc', 'train_time_s'}.

    The mock mode exists only to verify the complete DSE pipeline quickly.
    It must not be used for the actual lab results.
    """
    if args.mock:
        def mock_train(cfg):
            rng = np.random.default_rng(
                1000 + sum((i + 1) * w for i, w in enumerate(cfg))
            )
            params = count_weights(cfg)
            drop = (
                1.2
                + 300 / np.sqrt(params)
                + 25 / min(cfg)
                + rng.normal(0, 0.08)
            )
            drop = max(drop, 0.5)

            return {
                "test_acc": 100.0 - drop,
                "train_time_s": 0.0,
            }

        return mock_train

    import torch

    device = pick_device(args.device)
    print(f"Device: {device} | loading MNIST ...")
    data = load_mnist(args.data_dir, device)

    def real_train(cfg):
        accuracies = []
        total_time = 0.0

        for seed_offset in range(args.seeds):
            train_cfg = TrainConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                momentum=args.momentum,
                seed=args.seed + seed_offset,
            )

            result = train_and_evaluate(
                cfg,
                data,
                train_cfg,
                device,
            )

            accuracies.append(result["test_acc"])
            total_time += result["train_time_s"]

        return {
            "test_acc": float(np.mean(accuracies)),
            "train_time_s": total_time,
        }

    return real_train


def load_results(path: str):
    """
    Load already-completed configurations so an interrupted exhaustive run
    can resume without retraining them.
    """
    if not os.path.exists(path):
        return {}

    df = pd.read_csv(path, dtype={"widths": str})

    results = {}
    for row in df.itertuples(index=False):
        cfg = tuple(int(x) for x in row.widths.split("-"))
        results[cfg] = row

    return results


def append_row(path: str, row: dict):
    is_new = not os.path.exists(path)

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)

        if is_new:
            writer.writeheader()

        writer.writerow(row)


def run_exhaustive(args, space: DesignSpace, train_fn, csv_path: str):
    configs = space.all_configs()
    explored = load_results(csv_path)

    if explored:
        print(f"Resuming: {len(explored)} configurations already completed.")

    todo = [cfg for cfg in configs if cfg not in explored]

    if not todo:
        print("All configurations are already present in the results CSV.")
        return

    start_time = time.time()

    for k, cfg in enumerate(todo, start=1):
        result = train_fn(cfg)

        row = {
            "n": len(cfg),
            "widths": "-".join(map(str, cfg)),
            "n_weights": count_weights(cfg),
            "n_mults": count_mults(cfg),
            "cost": usage_cost(cfg),
            "test_acc": round(result["test_acc"], 4),
            "acc_drop": round(100.0 - result["test_acc"], 4),
            "train_time_s": round(result["train_time_s"], 2),
        }

        append_row(csv_path, row)

        elapsed = time.time() - start_time
        eta_min = (
            (elapsed / k) * (len(todo) - k) / 60.0
            if k
            else 0.0
        )

        # Find the global position in the full exhaustive list.
        completed_before_this_run = len(configs) - len(todo)
        global_index = completed_before_this_run + k

        print(
            f"[{global_index:3d}/{len(configs)}] "
            f"{row['widths']:<14} "
            f"cost={row['cost']:>12,.0f} "
            f"acc={row['test_acc']:.2f}% "
            f"drop={row['acc_drop']:.2f}% "
            f"({row['train_time_s']:.1f}s, ETA {eta_min:.1f} min)"
        )


def main():
    args = parse_args()

    if args.seeds < 1:
        raise ValueError("--seeds must be >= 1")

    os.makedirs(args.out, exist_ok=True)

    space = DesignSpace(
        n_min=args.n_min,
        n_max=args.n_max,
        m_values=args.m_values,
        per_layer=(args.space == "per-layer"),
    )

    prefix = "MOCK_" if args.mock else ""
    name = f"{prefix}exhaustive_{args.space}"
    csv_path = os.path.join(args.out, f"{name}.csv")

    print("=" * 72)
    print("ELCT918 - Exhaustive Multi-Objective DSE")
    print("=" * 72)
    print(
        f"Design space : n in [{args.n_min}, {args.n_max}], "
        f"m in {tuple(args.m_values)}"
    )
    print(f"Space type   : {args.space}")
    print(f"Configurations: {len(space)}")
    print(
        f"Training     : SGD lr={args.lr}, momentum={args.momentum}, "
        f"batch={args.batch_size}, epochs={args.epochs}, seeds={args.seeds}"
    )
    print(f"Mock mode    : {args.mock}")
    print(f"Results CSV  : {csv_path}")
    print()

    train_fn = make_train_fn(args)

    start = time.time()

    # Exhaustive exploration ONLY.
    run_exhaustive(args, space, train_fn, csv_path)

    elapsed_min = (time.time() - start) / 60.0
    print(f"\nExploration finished in {elapsed_min:.2f} min.")

    # -------------------------------------------------------------------------
    # Pareto analysis over all explored configurations.
    # -------------------------------------------------------------------------
    df = pd.read_csv(csv_path, dtype={"widths": str})

    expected = len(space)
    if len(df) < expected:
        print(
            f"WARNING: only {len(df)}/{expected} configurations are present. "
            "The Pareto front below is for the completed portion."
        )

    front = pareto_front(df)

    front_csv = os.path.join(
        args.out,
        f"{name}_pareto_front.csv",
    )
    front.to_csv(front_csv, index=False)

    png = os.path.join(
        args.out,
        f"pareto_{name}.png",
    )

    title = (
        f"Exhaustive DSE, {args.space} space - MNIST MLP "
        f"({len(df)} of {len(space)} configurations)"
    )

    plot_pareto(
        df,
        png,
        title,
        annotate=not args.no_annotate,
        log_x=not args.linear_x,
    )

    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)
    print(
        f"Trained {len(df)} / {len(space)} configurations "
        f"({100.0 * len(df) / len(space):.1f}%)"
    )
    print(f"\nPareto-optimal configurations ({len(front)}):")

    print(
        front[
            ["widths", "cost", "test_acc", "acc_drop"]
        ].to_string(index=False)
    )

    print("\nSaved:")
    print(f"  {csv_path}")
    print(f"  {front_csv}")
    print(f"  {png}")


# =============================================================================
# Standalone functionality checks
# =============================================================================

def self_test():
    """
    Fast, dependency-light checks for the merged implementation.

    This checks:
      - exhaustive configuration counts,
      - network generator topology,
      - analytical cost,
      - Pareto dominance,
      - mock end-to-end pipeline.
    """
    print("Running self-test...")

    # Design-space checks from the selected assignment space.
    uniform = DesignSpace(per_layer=False)
    per_layer = DesignSpace(per_layer=True)

    assert len(uniform) == 18, len(uniform)
    assert len(per_layer) == 258, len(per_layer)

    assert len(uniform.all_configs()) == 18
    assert len(per_layer.all_configs()) == 258

    # Cost sanity check for one hidden layer of 10 nodes:
    # 784->10: 7840 weights + 10 biases
    # 10->10: 100 weights + 10 biases
    # multiplications = 7840 + 100
    cfg = (10,)
    assert count_weights(cfg) == 7960
    assert count_mults(cfg) == 7940
    assert usage_cost(cfg) == 7960 * 139 + 7940

    # Pareto sanity check.
    test_df = pd.DataFrame([
        {"widths": "10", "cost": 100.0, "acc_drop": 10.0},
        {"widths": "20", "cost": 200.0, "acc_drop": 10.0},
        {"widths": "30", "cost": 150.0, "acc_drop": 8.0},
    ])

    front = pareto_front(test_df)
    assert set(front["widths"]) == {"10", "30"}

    # Network-generator check requires PyTorch but not MNIST.
    try:
        model = build_model(3, [10, 20, 40])
        linear_layers = [
            layer for layer in model
            if layer.__class__.__name__ == "Linear"
        ]
        assert len(linear_layers) == 4
        assert linear_layers[0].in_features == 784
        assert linear_layers[0].out_features == 10
        assert linear_layers[-1].out_features == 10
    except ImportError:
        print("PyTorch not installed: network topology check skipped.")

    print("Self-test PASSED.")
    print("Expected exhaustive sizes: uniform=18, per-layer=258.")


if __name__ == "__main__":
    # Optional internal self-test without argparse.
    # Use: python run_dse.py --self-test
    import sys

    if "--self-test" in sys.argv:
        self_test()
    else:
        main()
