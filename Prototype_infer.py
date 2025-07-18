# proto_single_graph_eval.py

import torch
import random
import numpy as np
from sklearn.metrics import classification_report
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GAT
# from proto_gat_main import GATEncoder, compute_prototypes, euclidean_distance

IN_CHANNELS = 18
OUT_CLASSES = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------- Compute Prototypes ------------------------
def compute_prototypes(embeddings, labels, num_classes=4):
    prototypes = []
    for c in range(num_classes):
        class_mask = (labels == c)
        if class_mask.sum() == 0:
            prototypes.append(torch.zeros_like(embeddings[0]))
        else:
            prototypes.append(embeddings[class_mask].mean(dim=0))
    return torch.stack(prototypes)

# --------- Compute Distances ------------------------
def euclidean_distance(a, b):
    return ((a.unsqueeze(1) - b.unsqueeze(0)) ** 2).sum(dim=2)


# --------- ProtoNet GAT Encoder --------------------
class GATEncoder(nn.Module):
    def __init__(self, hidden=256, heads=8, dropout=0.2, layers=2):
        super().__init__()
        self.gnn = GAT(
            in_channels=IN_CHANNELS,
            hidden_channels=hidden,
            out_channels=OUT_CLASSES,
            heads=heads,
            num_layers=layers,
            dropout=dropout,
            edge_dim=1,
            v2=True,
            jk='cat'
        )

    def forward(self, x, edge_index, edge_attr):
        return self.gnn(x, edge_index, edge_weight=edge_attr)

# --------- Load Trained GAT Proto Encoder ---------
def load_encoder(model_path):
    encoder = GATEncoder().to(DEVICE)
    encoder.load_state_dict(torch.load(model_path, map_location=DEVICE))
    encoder.eval()
    return encoder

# --------- Evaluate Using Graph(s) ---------
def evaluate_single_graph(graph_path, model_path, support_ratio=0.2):
    print(" Loading graph(s) from:", graph_path)
    data = torch.load(graph_path, map_location=DEVICE)
    encoder = load_encoder(model_path)

    from collections import defaultdict
    all_preds = []
    all_trues = []

    if isinstance(data, list):
        print(" Detected list of graphs. Splitting support/query at graph level.")
        random.shuffle(data)
        split = int(support_ratio * len(data))
        if split == 0:
            split = 1
        support_graphs = data[:split]
        query_graphs = data[split:]

        support_emb, support_y = [], []
        for g in support_graphs:
            g = g.to(DEVICE)
            emb = encoder(g.x, g.edge_index, g.edge_attr)
            support_emb.append(emb)
            support_y.append(g.y)
        if not support_emb or not support_y:
            print("No support graphs available. Please check your data or support_ratio.")
            return

        support_emb = torch.cat(support_emb, dim=0)
        support_y = torch.cat(support_y, dim=0)
        prototypes = compute_prototypes(support_emb, support_y)

        for i, g in enumerate(query_graphs):
            g = g.to(DEVICE)
            with torch.no_grad():
                emb = encoder(g.x, g.edge_index, g.edge_attr)
                dists = euclidean_distance(emb, prototypes)
                preds = dists.argmin(dim=1).cpu()
                all_preds.append(preds)
                all_trues.append(g.y.cpu())

                value_indices = (preds == 1).nonzero(as_tuple=True)[0].tolist()
                print(f"\n Graph {i+1} Predictions:")
                print(preds.tolist())
                print(f" Predicted VALUE nodes: {value_indices}")

                # Per-label accuracy for this query graph
                print(classification_report(g.y.cpu(), preds, zero_division=0))

        # Overall classification report across all query graphs
        all_preds_flat = torch.cat(all_preds).numpy()
        all_trues_flat = torch.cat(all_trues).numpy()
        print("\n===== Overall Classification Report (all query graphs) =====")
        print(classification_report(all_trues_flat, all_preds_flat, zero_division=0))

    else:
        print("🔎 Detected single graph. Splitting support/query at node level.")
        graph = data
        num_nodes = graph.x.size(0)
        indices = list(range(num_nodes))
        random.shuffle(indices)

        split = int(support_ratio * num_nodes)
        support_idx = indices[:split]
        query_idx = indices[split:]

        support_mask = torch.zeros(num_nodes, dtype=torch.bool)
        support_mask[support_idx] = True

        query_mask = torch.zeros(num_nodes, dtype=torch.bool)
        query_mask[query_idx] = True

        support_x = graph.x[support_mask]
        support_y = graph.y[support_mask]

        query_x = graph.x[query_mask]
        query_y = graph.y[query_mask]

        with torch.no_grad():
            embeddings = encoder(graph.x.to(DEVICE), graph.edge_index.to(DEVICE), graph.edge_attr.to(DEVICE))
            support_emb = embeddings[support_mask.to(DEVICE)]
            query_emb = embeddings[query_mask.to(DEVICE)]
            prototypes = compute_prototypes(support_emb, support_y.to(DEVICE))
            dists = euclidean_distance(query_emb, prototypes)
            preds = dists.argmin(dim=1).cpu()

        all_preds.append(preds)
        all_trues.append(query_y.cpu())

        print(f"Evaluation Completed on Single Graph")
        print(f"Predicted labels: {preds.tolist()}")
        print(f"True labels: {query_y.tolist()}")

        # Per-label accuracy for this query set
        print(classification_report(query_y, preds, zero_division=0))

        # Overall (just this graph)
        all_preds_flat = torch.cat(all_preds).numpy()
        all_trues_flat = torch.cat(all_trues).numpy()
        print("\n===== Overall Classification Report (this graph) =====")
        print(classification_report(all_trues_flat, all_preds_flat, zero_division=0))

        return all_preds, all_trues


# --------- Main ---------
if __name__ == "__main__":
    evaluate_single_graph(
        graph_path="data/test_data/test_dataset_loan.pt",
        model_path="models/prototypical/proto_gat_encoder.pt",
        support_ratio=0.2
    )