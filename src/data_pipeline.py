import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

RAW_DIR = Path(__file__).parent.parent / "dataset"
PROCESSED_DIR = RAW_DIR / "processed"


def load_raw_data() -> Data:
    # --- Edges ---
    edges_df = pd.read_csv(RAW_DIR / "musae_git_edges.csv")
    src = torch.tensor(edges_df["id_1"].values, dtype=torch.long)
    dst = torch.tensor(edges_df["id_2"].values, dtype=torch.long)
    # Undirected: store both directions
    edge_index = torch.stack(
        [torch.cat([src, dst]), torch.cat([dst, src])], dim=0
    )

    # --- Node features (multi-hot binary, shape: num_nodes × 4005) ---
    features_df = pd.read_csv(RAW_DIR / "musae_git_features.csv")
    target_df = pd.read_csv(RAW_DIR / "musae_git_target.csv")

    num_nodes = int(target_df["id"].max()) + 1
    num_features = int(features_df["feature"].max()) + 1

    x = torch.zeros((num_nodes, num_features), dtype=torch.float)
    node_ids = torch.tensor(features_df["node"].values, dtype=torch.long)
    feat_ids = torch.tensor(features_df["feature"].values, dtype=torch.long)
    x[node_ids, feat_ids] = 1.0

    # --- Node labels (0 = web dev, 1 = ML dev) ---
    y = torch.tensor(target_df["ml_target"].values, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)


def split_and_save(data: Data) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Save the full graph for heuristics and Node2Vec (they need the plain graph)
    torch.save(data, PROCESSED_DIR / "graph.pt")
    print(f"Graph: {data.num_nodes} nodes, {data.num_edges // 2} undirected edges")
    print(f"Node feature dim: {data.x.shape[1]}")

    # Fix seed immediately before the split so the result is deterministic
    torch.manual_seed(SEED)

    transform = RandomLinkSplit(
        num_val=0.1,
        num_test=0.2,
        is_undirected=True,
        add_negative_train_samples=True,
        neg_sampling_ratio=1.0,
    )
    train_data, val_data, test_data = transform(data)

    splits = {"train": train_data, "val": val_data, "test": test_data}
    torch.save(splits, PROCESSED_DIR / "splits.pt")

    for name, split in splits.items():
        pos = (split.edge_label == 1).sum().item()
        neg = (split.edge_label == 0).sum().item()
        print(f"  {name:5s}: {pos} pos edges, {neg} neg edges")


def load_splits() -> tuple[Data, Data, Data]:
    splits = torch.load(PROCESSED_DIR / "splits.pt", weights_only=False)
    return splits["train"], splits["val"], splits["test"]


def load_graph() -> Data:
    return torch.load(PROCESSED_DIR / "graph.pt", weights_only=False)


if __name__ == "__main__":
    print("Loading raw data...")
    data = load_raw_data()
    print("Splitting edges and saving...")
    split_and_save(data)
    print("Done. Processed files written to dataset/processed/")
