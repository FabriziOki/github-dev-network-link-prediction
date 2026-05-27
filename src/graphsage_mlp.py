import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx
from torch_geometric.nn import SAGEConv
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).parent))
from data_pipeline import load_graph, load_splits

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

HIDDEN_DIM = 64
DROPOUT = 0.3
LR = 0.01
EPOCHS = 100
PATIENCE = 10

RESULTS_DIR = Path(__file__).parent.parent / "results"


# ---------------------------------------------------------------------------
# Heuristic feature computation
# ---------------------------------------------------------------------------

def build_nx_graph(edge_index: torch.Tensor, num_nodes: int) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edge_index.t().tolist())
    return G


def compute_heuristics(G: nx.Graph, edge_label_index: torch.Tensor) -> torch.Tensor:
    """Returns a (E, 3) tensor with [CN, Jaccard, Adamic-Adar] per edge."""
    pairs = [(int(u), int(v)) for u, v in edge_label_index.t().tolist()]

    cn = np.array(
        [len(list(nx.common_neighbors(G, u, v))) for u, v in pairs], dtype=np.float32
    )
    jac = np.array(
        [p for _, _, p in nx.jaccard_coefficient(G, pairs)], dtype=np.float32
    )
    aa = np.array(
        [p for _, _, p in nx.adamic_adar_index(G, pairs)], dtype=np.float32
    )

    return torch.tensor(np.stack([cn, jac, aa], axis=1))  # (E, 3)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GraphSAGEMLP(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, dropout: float):
        super().__init__()
        # Encoder: 2-layer GraphSAGE
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr="mean")
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr="mean")
        # MLP decoder: Hadamard(64) + heuristics(3) → 64 → 1
        self.lin1 = torch.nn.Linear(hidden_channels + 3, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, 1)
        self.dropout = dropout

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def decode(
        self, z: torch.Tensor, edge_label_index: torch.Tensor, heuristics: torch.Tensor
    ) -> torch.Tensor:
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        edge_feat = torch.cat([src * dst, heuristics], dim=-1)  # (E, 67)
        h = F.relu(self.lin1(edge_feat))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.lin2(h).squeeze(-1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_label_index: torch.Tensor,
        heuristics: torch.Tensor,
    ) -> torch.Tensor:
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index, heuristics)


# ---------------------------------------------------------------------------
# Train / eval helpers
# ---------------------------------------------------------------------------

def train_step(model, optimizer, x, train_data, train_heuristics):
    model.train()
    optimizer.zero_grad()
    logits = model(x, train_data.edge_index, train_data.edge_label_index, train_heuristics)
    loss = F.binary_cross_entropy_with_logits(logits, train_data.edge_label.float())
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, x, split_data, heuristics):
    model.eval()
    logits = model(x, split_data.edge_index, split_data.edge_label_index, heuristics)
    probs = logits.sigmoid().numpy()
    labels = split_data.edge_label.numpy()
    auc = roc_auc_score(labels, probs)
    ap = average_precision_score(labels, probs)
    return auc, ap


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_tier4() -> dict:
    graph = load_graph()
    train_data, val_data, test_data = load_splits()
    x = graph.x

    print("Precomputing heuristic features...")
    train_G = build_nx_graph(train_data.edge_index, graph.num_nodes)
    val_G   = build_nx_graph(val_data.edge_index,   graph.num_nodes)
    test_G  = build_nx_graph(test_data.edge_index,  graph.num_nodes)

    train_heur = compute_heuristics(train_G, train_data.edge_label_index)
    val_heur   = compute_heuristics(val_G,   val_data.edge_label_index)
    test_heur  = compute_heuristics(test_G,  test_data.edge_label_index)
    print("Done.")

    torch.manual_seed(SEED)
    model = GraphSAGEMLP(x.shape[1], HIDDEN_DIM, DROPOUT)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_auc = 0.0
    best_state = None
    patience_count = 0

    print(f"\n{'Epoch':<8} {'Loss':>8}  {'Val AUC':>9}  {'Val AP':>8}")
    print("-" * 40)

    for epoch in range(1, EPOCHS + 1):
        loss = train_step(model, optimizer, x, train_data, train_heur)
        val_auc, val_ap = evaluate(model, x, val_data, val_heur)

        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:<8} {loss:>8.4f}  {val_auc:>9.4f}  {val_ap:>8.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)

    results = {}
    print(f"\n{'Method':<22} {'Split':<6} {'AUC':>7}  {'AP':>7}")
    print("-" * 46)

    for split_name, split_data, heur in [
        ("val",  val_data,  val_heur),
        ("test", test_data, test_heur),
    ]:
        auc, ap = evaluate(model, x, split_data, heur)
        results[split_name] = {"auc": auc, "ap": ap}
        print(f"{'GraphSAGE + Heur.':<22} {split_name:<6} {auc:>7.4f}  {ap:>7.4f}")

    save_results(results)
    torch.save(model.state_dict(), RESULTS_DIR / "graphsage_mlp_weights.pt")
    return results


def save_results(results: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "metrics.csv"

    existing_lines = []
    if csv_path.exists():
        all_lines = csv_path.read_text().splitlines()
        existing_lines = [l for l in all_lines[1:] if not l.startswith("GraphSAGE + Heur.")]

    new_lines = [
        f"GraphSAGE + Heur.,4,{split},{m['auc']:.6f},{m['ap']:.6f}"
        for split, m in results.items()
    ]

    header = "method,tier,split,auc,ap"
    csv_path.write_text("\n".join([header] + existing_lines + new_lines) + "\n")
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    run_tier4()
