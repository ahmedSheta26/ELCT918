# Exhaustive Multi-Objective Design Space Exploration (DSE) on MNIST

A script performing exhaustive multi-objective Design Space Exploration (DSE) for parameterizable Multi-Layer Perceptrons (MLPs) trained on the MNIST dataset. It evaluates trade-offs between analytical hardware usage cost ($139 \times \text{weights} + 1 \times \text{MACs}$) and accuracy drop ($100 - \text{test accuracy}$), automatically extracting and plotting the Pareto front.

---

## Frameworks & Dependencies

### Deep Learning & Numerical Frameworks
- **[PyTorch](https://pytorch.org/)** (`torch`, `torchvision`): Core deep learning framework used for programmatic MLP construction, GPU/MPS acceleration, and MNIST dataset handling.
- **[NumPy](https://numpy.org/)**: Numerical operations and Pareto-dominance filtering.
- **[Pandas](https://pandas.pydata.org/)**: Tabular data structuring, CSV logging, and Pareto-front parsing.
- **[Matplotlib](https://matplotlib.org/)**: Generating Pareto-front trade-off curves (non-interactive backend `Agg`).

### Installation
Install the necessary dependencies using `pip`:

```bash
pip install torch torchvision numpy pandas matplotlib
```

---

## How to Run

### 1. Internal Sanity Checks & Self-Test
Verify configuration spaces, cost models, and model topology logic without downloading MNIST:

```bash
python run_dse.py --self-test
```

### 2. Fast Pipeline Verification (Mock Run)
Simulate training without loading PyTorch or the MNIST dataset to test CSV generation and plotting:

```bash
python run_dse.py --mock --space uniform
```

### 3. Quick Functionality Run
Run a lightweight real training run (1 layer, widths 10 and 20, 1 epoch):

```bash
python run_dse.py --space uniform --n-min 1 --n-max 1 --m-values 10 20 --epochs 1
```

### 4. Full Exhaustive Exploration

- **Uniform Space (18 configurations)**:
  All hidden layers share the same width.
  ```bash
  python run_dse.py --space uniform
  ```

- **Per-Layer Independent Space (258 configurations)**:
  Each hidden layer explores candidate widths independently.
  ```bash
  python run_dse.py --space per-layer
  ```

*Note: Results are appended to the CSV file immediately after each configuration completes. Interrupted runs will automatically resume from the last completed configuration.*

---

## Useful Command-Line Options

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--space` | `uniform` | Search space type: `uniform` (18 configs) or `per-layer` (258 configs). |
| `--device` | `auto` | Target hardware: `auto`, `cpu`, `cuda`, or `mps`. |
| `--epochs` | `10` | Number of training epochs per configuration. |
| `--batch-size`| `200` | Mini-batch size for SGD optimization. |
| `--lr` | `0.1` | Learning rate for SGD. |
| `--seeds` | `1` | Number of random seeds to train and average per configuration. |
| `--out` | `results` | Output directory for logs and plots. |
| `--data-dir` | `./data` | Directory where MNIST will be downloaded/cached. |
| `--linear-x` | *Off* | Use a linear scale instead of logarithmic on the usage cost axis. |

---

## Output Files

All outputs are saved to the directory specified by `--out` (default: `./results/`):

- `<name>.csv`: Complete log of all evaluated models (weights, mults, usage cost, test accuracy, train time).
- `<name>_pareto_front.csv`: Non-dominated optimal network architectures.
- `pareto_<name>.png`: 2D scatter and step plot visualizing dominated designs vs. the Pareto frontier.