import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, dropout: float):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr="mean")
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr="mean")
        self.dropout = dropout

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        return (src * dst).sum(dim=-1)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_label_index: torch.Tensor
    ) -> torch.Tensor:
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)


@torch.no_grad()
def evaluate(
    model: GraphSAGE,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_label_index: torch.Tensor,
    edge_label: torch.Tensor,
) -> tuple[float, float]:
    model.eval()
    logits = model(x, edge_index, edge_label_index)
    probs = logits.sigmoid().numpy()
    labels = edge_label.numpy()
    auc = roc_auc_score(labels, probs)
    ap = average_precision_score(labels, probs)
    return auc, ap


def train(model, optimizer, x, train_data):
    model.train()
    optimizer.zero_grad()
    logits = model(x, train_data.edge_index, train_data.edge_label_index)
    loss = F.binary_cross_entropy_with_logits(logits, train_data.edge_label.float())
    loss.backward()
    optimizer.step()
    return loss.item()


def run_tier3() -> dict:
    graph = load_graph()
    train_data, val_data, test_data = load_splits()

    x = graph.x
    in_channels = x.shape[1]

    torch.manual_seed(SEED)
    model = GraphSAGE(in_channels, HIDDEN_DIM, DROPOUT)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_auc = 0.0
    best_state = None
    patience_count = 0

    print(f"{'Epoch':<8} {'Loss':>8}  {'Val AUC':>9}  {'Val AP':>8}")
    print("-" * 40)

    for epoch in range(1, EPOCHS + 1):
        loss = train(model, optimizer, x, train_data)
        val_auc, val_ap = evaluate(
            model, x, val_data.edge_index,
            val_data.edge_label_index, val_data.edge_label
        )

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
    print(f"\n{'Method':<18} {'Split':<6} {'AUC':>7}  {'AP':>7}")
    print("-" * 42)

    for split_name, split_data in [("val", val_data), ("test", test_data)]:
        auc, ap = evaluate(
            model, x, split_data.edge_index,
            split_data.edge_label_index, split_data.edge_label
        )
        results[split_name] = {"auc": auc, "ap": ap}
        print(f"{'GraphSAGE':<18} {split_name:<6} {auc:>7.4f}  {ap:>7.4f}")

    save_results(results)
    torch.save(model.state_dict(), RESULTS_DIR / "graphsage_weights.pt")
    return results


def save_results(results: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "metrics.csv"

    existing_lines = []
    if csv_path.exists():
        all_lines = csv_path.read_text().splitlines()
        existing_lines = [l for l in all_lines[1:] if not l.startswith("GraphSAGE,3")]

    new_lines = [
        f"GraphSAGE,3,{split},{m['auc']:.6f},{m['ap']:.6f}"
        for split, m in results.items()
    ]

    header = "method,tier,split,auc,ap"
    csv_path.write_text("\n".join([header] + existing_lines + new_lines) + "\n")
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    run_tier3()
