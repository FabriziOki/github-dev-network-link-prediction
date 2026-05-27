"""
analysis.py — UMAP visualization + results comparison table

Produces:
  results/figures/umap_tier3.png
  results/figures/umap_tier4.png
  results/figures/umap_comparison.png   (side-by-side)
  results/figures/results_table.png     (AUC / AP bar chart)
"""
import sys
from pathlib import Path

import numpy as np
import torch
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import umap

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_pipeline import load_graph, load_splits
from graphsage import GraphSAGE
from graphsage_mlp import GraphSAGEMLP

SEED = 42
HIDDEN_DIM = 64
DROPOUT = 0.3

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_embeddings(model, x: torch.Tensor, edge_index: torch.Tensor) -> np.ndarray:
    model.eval()
    z = model.encode(x, edge_index)
    return z.numpy()


def load_tier3(in_channels: int) -> GraphSAGE:
    weights_path = RESULTS_DIR / "graphsage_weights.pt"
    if not weights_path.exists():
        raise FileNotFoundError(
            "graphsage_weights.pt not found — run src/graphsage.py first"
        )
    model = GraphSAGE(in_channels, HIDDEN_DIM, DROPOUT)
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    return model


def load_tier4(in_channels: int) -> GraphSAGEMLP:
    weights_path = RESULTS_DIR / "graphsage_mlp_weights.pt"
    if not weights_path.exists():
        raise FileNotFoundError(
            "graphsage_mlp_weights.pt not found — run src/graphsage_mlp.py first"
        )
    model = GraphSAGEMLP(in_channels, HIDDEN_DIM, DROPOUT)
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    return model


# ---------------------------------------------------------------------------
# Louvain community detection
# ---------------------------------------------------------------------------

def detect_communities(edge_index: torch.Tensor, num_nodes: int) -> np.ndarray:
    print("Running Louvain community detection...")
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edge_index.t().tolist())

    communities = nx.community.louvain_communities(G, seed=SEED)
    node_community = np.zeros(num_nodes, dtype=int)
    for comm_id, members in enumerate(communities):
        for node in members:
            node_community[node] = comm_id

    n_comm = len(communities)
    sizes = sorted([len(c) for c in communities], reverse=True)
    print(f"  Found {n_comm} communities. Top-5 sizes: {sizes[:5]}")
    return node_community


# ---------------------------------------------------------------------------
# UMAP
# ---------------------------------------------------------------------------

def run_umap(z: np.ndarray) -> np.ndarray:
    reducer = umap.UMAP(n_components=2, random_state=SEED, n_jobs=1)
    return reducer.fit_transform(z)


def plot_umap_single(
    embedding_2d: np.ndarray,
    communities: np.ndarray,
    title: str,
    ax: plt.Axes,
    max_communities: int = 10,
) -> None:
    # Only colour the top-N largest communities; lump the rest as 'Other'
    comm_ids, counts = np.unique(communities, return_counts=True)
    top_ids = set(comm_ids[np.argsort(counts)[-max_communities:]])
    labels = np.array(
        [c if c in top_ids else -1 for c in communities], dtype=int
    )

    palette = sns.color_palette("tab10", max_communities)
    color_map = {cid: palette[i] for i, cid in enumerate(sorted(top_ids))}
    color_map[-1] = (0.85, 0.85, 0.85)  # grey for 'Other'

    colors = [color_map[l] for l in labels]
    ax.scatter(
        embedding_2d[:, 0], embedding_2d[:, 1],
        c=colors, s=2, alpha=0.5, linewidths=0,
    )

    legend_handles = [
        mpatches.Patch(color=color_map[cid], label=f"Community {cid}")
        for cid in sorted(top_ids)
    ]
    legend_handles.append(mpatches.Patch(color=color_map[-1], label="Other"))
    ax.legend(handles=legend_handles, markerscale=3, fontsize=7,
              loc="upper right", framealpha=0.7)
    ax.set_title(title, fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])


# ---------------------------------------------------------------------------
# Results comparison table
# ---------------------------------------------------------------------------

def plot_results_table(csv_path: Path) -> None:
    df = pd.read_csv(csv_path)
    test_df = df[df["split"] == "test"].copy()

    method_order = [
        "Common Neighbors", "Jaccard", "Adamic-Adar",
        "Node2Vec + LR", "GraphSAGE", "GraphSAGE + Heur.",
    ]
    test_df["method"] = pd.Categorical(test_df["method"], categories=method_order, ordered=True)
    test_df = test_df.sort_values("method")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    tier_colors = {1: "#4C72B0", 2: "#DD8452", 3: "#55A868", 4: "#C44E52"}
    bar_colors = [tier_colors[t] for t in test_df["tier"]]

    for ax, metric, label in zip(axes, ["auc", "ap"], ["AUC-ROC", "Average Precision"]):
        bars = ax.bar(range(len(test_df)), test_df[metric], color=bar_colors, edgecolor="white")
        ax.set_xticks(range(len(test_df)))
        ax.set_xticklabels(test_df["method"], rotation=25, ha="right", fontsize=9)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f"Test {label} by Method", fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    # Tier legend
    legend_handles = [
        mpatches.Patch(color=c, label=f"Tier {t}")
        for t, c in tier_colors.items()
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    out = FIGURES_DIR / "results_table.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    graph = load_graph()
    train_data, val_data, test_data = load_splits()

    x = graph.x
    in_channels = x.shape[1]
    edge_index = train_data.edge_index  # training graph for embedding extraction

    # --- Community detection (once, shared across both plots) ---
    communities = detect_communities(edge_index, graph.num_nodes)

    # --- Tier 3 embeddings ---
    print("\nExtracting Tier 3 embeddings...")
    t3_model = load_tier3(in_channels)
    z3 = extract_embeddings(t3_model, x, edge_index)

    print("Running UMAP on Tier 3...")
    umap3 = run_umap(z3)

    # --- Tier 4 embeddings ---
    print("\nExtracting Tier 4 embeddings...")
    t4_model = load_tier4(in_channels)
    z4 = extract_embeddings(t4_model, x, edge_index)

    print("Running UMAP on Tier 4...")
    umap4 = run_umap(z4)

    # --- Side-by-side UMAP plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_umap_single(umap3, communities, "Tier 3 — GraphSAGE", axes[0])
    plot_umap_single(umap4, communities, "Tier 4 — GraphSAGE + Heuristics", axes[1])
    fig.suptitle("Node Embedding Space (UMAP), colored by Louvain community", fontsize=13)
    fig.tight_layout()

    out = FIGURES_DIR / "umap_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out}")
    plt.close(fig)

    # --- Individual UMAP plots ---
    for umap_2d, title, fname in [
        (umap3, "Tier 3 — GraphSAGE", "umap_tier3.png"),
        (umap4, "Tier 4 — GraphSAGE + Heuristics", "umap_tier4.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 7))
        plot_umap_single(umap_2d, communities, title, ax)
        fig.tight_layout()
        out = FIGURES_DIR / fname
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved {out}")
        plt.close(fig)

    # --- Results comparison bar chart ---
    csv_path = RESULTS_DIR / "metrics.csv"
    if csv_path.exists():
        print("\nGenerating results comparison chart...")
        plot_results_table(csv_path)
    else:
        print(f"\nSkipping results chart — {csv_path} not found.")


if __name__ == "__main__":
    main()
