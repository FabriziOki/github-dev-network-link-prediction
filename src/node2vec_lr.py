import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.nn import Node2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).parent))
from data_pipeline import load_graph, load_splits

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

EMBEDDING_DIM = 64
WALK_LENGTH = 20
CONTEXT_SIZE = 10
WALKS_PER_NODE = 10
P = 1.0
Q = 1.0
BATCH_SIZE = 256
EPOCHS = 5

RESULTS_DIR = Path(__file__).parent.parent / "results"


def train_node2vec(edge_index: torch.Tensor, num_nodes: int) -> Node2Vec:
    model = Node2Vec(
        edge_index,
        embedding_dim=EMBEDDING_DIM,
        walk_length=WALK_LENGTH,
        context_size=CONTEXT_SIZE,
        walks_per_node=WALKS_PER_NODE,
        p=P,
        q=Q,
        num_negative_samples=1,
        num_nodes=num_nodes,
        sparse=True,
    )

    loader = model.loader(batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw, neg_rw)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch}/{EPOCHS}  loss={total_loss / len(loader):.4f}")

    return model


def get_embeddings(model: Node2Vec, num_nodes: int) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        z = model(torch.arange(num_nodes))
    return z.numpy()


def hadamard(z: np.ndarray, edge_label_index: torch.Tensor) -> np.ndarray:
    src = edge_label_index[0].numpy()
    dst = edge_label_index[1].numpy()
    return z[src] * z[dst]


def run_tier2() -> dict:
    graph = load_graph()
    train_data, val_data, test_data = load_splits()

    print("Training Node2Vec on training graph...")
    torch.manual_seed(SEED)
    model = train_node2vec(train_data.edge_index, graph.num_nodes)

    print("\nExtracting embeddings...")
    z = get_embeddings(model, graph.num_nodes)

    X_train = hadamard(z, train_data.edge_label_index)
    y_train = train_data.edge_label.numpy()

    print("Training Logistic Regression...")
    clf = LogisticRegression(
        random_state=SEED, max_iter=1000, solver="saga", n_jobs=-1
    )
    clf.fit(X_train, y_train)

    results = {}
    print(f"\n{'Method':<18} {'Split':<6} {'AUC':>7}  {'AP':>7}")
    print("-" * 42)

    for split_name, split_data in [("val", val_data), ("test", test_data)]:
        X = hadamard(z, split_data.edge_label_index)
        y = split_data.edge_label.numpy()
        probs = clf.predict_proba(X)[:, 1]

        auc = roc_auc_score(y, probs)
        ap = average_precision_score(y, probs)
        results[(split_name,)] = {"auc": auc, "ap": ap}
        print(f"{'Node2Vec + LR':<18} {split_name:<6} {auc:>7.4f}  {ap:>7.4f}")

    save_results(results)
    return results


def save_results(results: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "metrics.csv"

    # Read existing rows (skip header), filter out any prior Node2Vec rows
    existing_lines = []
    if csv_path.exists():
        all_lines = csv_path.read_text().splitlines()
        existing_lines = [
            l for l in all_lines[1:] if not l.startswith("Node2Vec")
        ]

    new_lines = []
    for (split,), metrics in results.items():
        new_lines.append(
            f"Node2Vec + LR,2,{split},{metrics['auc']:.6f},{metrics['ap']:.6f}"
        )

    header = "method,tier,split,auc,ap"
    csv_path.write_text(
        "\n".join([header] + existing_lines + new_lines) + "\n"
    )
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    run_tier2()
