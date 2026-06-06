"""
eda.py — Exploratory Data Analysis

Produces:
  results/figures/eda_degree_distribution.png
  results/figures/eda_feature_counts.png
  results/figures/eda_class_balance.png
  results/figures/eda_summary.png   (4-panel overview)

Prints graph statistics to stdout.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import networkx as nx

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_pipeline import load_graph

FIGURES_DIR = Path(__file__).parent.parent / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")


# ---------------------------------------------------------------------------
# Graph statistics
# ---------------------------------------------------------------------------

def print_graph_stats(G: nx.Graph, x, y) -> None:
    degrees = [d for _, d in G.degree()]
    features_per_node = x.sum(dim=1).numpy()

    print("=" * 50)
    print("GRAPH STATISTICS")
    print("=" * 50)
    print(f"  Nodes               : {G.number_of_nodes():,}")
    print(f"  Edges (undirected)  : {G.number_of_edges():,}")
    print(f"  Density             : {nx.density(G):.6f}")
    print(f"  Avg degree          : {np.mean(degrees):.2f}")
    print(f"  Median degree       : {np.median(degrees):.0f}")
    print(f"  Max degree          : {np.max(degrees):,}")
    print(f"  Min degree          : {np.min(degrees)}")
    print(f"  Connected components: {nx.number_connected_components(G)}")
    print(f"  Node feature dim    : {x.shape[1]}")
    print(f"  Avg features/node   : {features_per_node.mean():.1f}")
    print(f"  Max features/node   : {int(features_per_node.max())}")
    print(f"  Nodes with 0 feat.  : {(features_per_node == 0).sum():,}")
    n_web = int((y == 0).sum())
    n_ml  = int((y == 1).sum())
    print(f"  Web developers (0)  : {n_web:,} ({100*n_web/len(y):.1f}%)")
    print(f"  ML developers  (1)  : {n_ml:,}  ({100*n_ml/len(y):.1f}%)")
    print("=" * 50)


# ---------------------------------------------------------------------------
# Individual plots
# ---------------------------------------------------------------------------

def plot_degree_distribution(degrees: np.ndarray, ax: plt.Axes) -> None:
    counts = np.bincount(degrees)
    nonzero = counts > 0
    deg_vals = np.where(nonzero)[0]
    deg_counts = counts[nonzero]

    ax.scatter(deg_vals, deg_counts, s=8, alpha=0.6, color="#4C72B0")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Degree (log scale)", fontsize=11)
    ax.set_ylabel("Count (log scale)", fontsize=11)
    ax.set_title("Degree Distribution (log-log)", fontsize=12)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    # Annotate basic stats
    textstr = (
        f"mean = {degrees.mean():.1f}\n"
        f"median = {np.median(degrees):.0f}\n"
        f"max = {degrees.max()}"
    )
    ax.text(0.97, 0.97, textstr, transform=ax.transAxes,
            fontsize=9, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))


def plot_degree_histogram(degrees: np.ndarray, ax: plt.Axes) -> None:
    # Cap at 95th percentile to avoid the long tail squashing the histogram
    cap = int(np.percentile(degrees, 95))
    clipped = np.clip(degrees, 0, cap)

    ax.hist(clipped, bins=50, color="#4C72B0", edgecolor="white", linewidth=0.4)
    ax.set_xlabel(f"Degree (capped at 95th pct = {cap})", fontsize=11)
    ax.set_ylabel("Number of nodes", fontsize=11)
    ax.set_title("Degree Distribution (histogram)", fontsize=12)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))


def plot_feature_counts(features_per_node: np.ndarray, ax: plt.Axes) -> None:
    ax.hist(features_per_node, bins=60, color="#DD8452", edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Number of active features per node", fontsize=11)
    ax.set_ylabel("Number of nodes", fontsize=11)
    ax.set_title("Feature Sparsity Distribution", fontsize=12)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    textstr = (
        f"mean = {features_per_node.mean():.1f}\n"
        f"median = {np.median(features_per_node):.0f}"
    )
    ax.text(0.97, 0.97, textstr, transform=ax.transAxes,
            fontsize=9, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))


def plot_class_balance(y, ax: plt.Axes) -> None:
    labels = ["Web dev (0)", "ML dev (1)"]
    counts = [(y == 0).sum().item(), (y == 1).sum().item()]
    colors = ["#4C72B0", "#55A868"]

    bars = ax.bar(labels, counts, color=colors, edgecolor="white", width=0.5)
    ax.set_ylabel("Number of nodes", fontsize=11)
    ax.set_title("Node Class Distribution", fontsize=12)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.bar_label(bars, labels=[f"{c:,}\n({100*c/sum(counts):.1f}%)" for c in counts],
                 padding=4, fontsize=10)
    ax.set_ylim(0, max(counts) * 1.2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading graph...")
    data = load_graph()

    x = data.x
    y = data.y
    edge_index = data.edge_index

    # Build NetworkX graph for stats
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    G.add_edges_from(edge_index.t().tolist())

    degrees = np.array([d for _, d in G.degree()])
    features_per_node = x.sum(dim=1).numpy()

    print_graph_stats(G, x, y)

    # --- 4-panel summary figure ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("GitHub Developer Network — EDA", fontsize=14, y=1.01)

    plot_degree_distribution(degrees, axes[0, 0])
    plot_degree_histogram(degrees, axes[0, 1])
    plot_feature_counts(features_per_node, axes[1, 0])
    plot_class_balance(y, axes[1, 1])

    fig.tight_layout()
    out = FIGURES_DIR / "eda_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out}")
    plt.close(fig)

    # --- Individual figures ---
    for fname, fn, args in [
        ("eda_degree_distribution.png", plot_degree_distribution, (degrees,)),
        ("eda_feature_counts.png",      plot_feature_counts,      (features_per_node,)),
        ("eda_class_balance.png",       plot_class_balance,       (y,)),
    ]:
        fig, ax = plt.subplots(figsize=(7, 5))
        fn(*args, ax)
        fig.tight_layout()
        out = FIGURES_DIR / fname
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved {out}")
        plt.close(fig)


if __name__ == "__main__":
    main()
