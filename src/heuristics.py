import random
import sys
from pathlib import Path

import numpy as np
import torch
import networkx as nx
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).parent))
from data_pipeline import load_graph, load_splits

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

RESULTS_DIR = Path(__file__).parent.parent / "results"


def build_nx_graph(edge_index: torch.Tensor, num_nodes: int) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edges = edge_index.t().tolist()
    G.add_edges_from(edges)
    return G


def score_edges(
    G: nx.Graph, edge_label_index: torch.Tensor, method: str
) -> np.ndarray:
    pairs = [
        (int(u), int(v)) for u, v in edge_label_index.t().tolist()
    ]

    if method == "cn":
        scores = [len(list(nx.common_neighbors(G, u, v))) for u, v in pairs]
    elif method == "jaccard":
        scores = [p for _, _, p in nx.jaccard_coefficient(G, pairs)]
    elif method == "adamic_adar":
        scores = [p for _, _, p in nx.adamic_adar_index(G, pairs)]
    else:
        raise ValueError(f"Unknown method: {method}")

    return np.array(scores, dtype=float)


def evaluate(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    auc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)
    return auc, ap


def run_tier1() -> dict:
    graph = load_graph()
    train_data, val_data, test_data = load_splits()

    # Use the appropriate graph backbone for each split:
    # val_data.edge_index  = training edges only
    # test_data.edge_index = training + val edges
    splits = {"val": val_data, "test": test_data}
    methods = ["cn", "jaccard", "adamic_adar"]
    results = {}

    print(f"{'Method':<18} {'Split':<6} {'AUC':>7}  {'AP':>7}")
    print("-" * 42)

    for split_name, split_data in splits.items():
        G = build_nx_graph(split_data.edge_index, graph.num_nodes)
        labels = split_data.edge_label.numpy()

        for method in methods:
            scores = score_edges(G, split_data.edge_label_index, method)
            auc, ap = evaluate(scores, labels)
            results[(method, split_name)] = {"auc": auc, "ap": ap}
            print(f"{method:<18} {split_name:<6} {auc:>7.4f}  {ap:>7.4f}")

    save_results(results)
    return results


def save_results(results: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "metrics.csv"

    method_labels = {
        "cn": "Common Neighbors",
        "jaccard": "Jaccard",
        "adamic_adar": "Adamic-Adar",
    }

    lines = ["method,tier,split,auc,ap"]
    for (method, split), metrics in results.items():
        name = method_labels[method]
        lines.append(
            f"{name},1,{split},{metrics['auc']:.6f},{metrics['ap']:.6f}"
        )

    csv_path.write_text("\n".join(lines) + "\n")
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    run_tier1()
