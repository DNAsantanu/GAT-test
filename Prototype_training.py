# proto_gat_main.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GAT

# ----------- Config -------------------
IN_CHANNELS = 18
OUT_CLASSES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# --------- Episode Sampler --------------------------
def sample_episode(data_list, task, k_shot=1,q_num=4):
    task_data = [d for d in data_list if getattr(d, 'task', None) == task]
    random.shuffle(task_data)
    return task_data[:k_shot], task_data[k_shot:k_shot + q_num]

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

# --------- Prototypical Loss ------------------------
def prototypical_loss(embeddings, labels, prototypes):
    dists = euclidean_distance(embeddings, prototypes)
    log_p_y = F.log_softmax(-dists, dim=1)
    loss = F.nll_loss(log_p_y, labels)
    preds = log_p_y.argmax(dim=1)
    acc = (preds == labels).float().mean().item()
    return loss, acc

# --------- Training Loop -----------------------------
def proto_train(data_list, encoder, optimizer, n_episodes=500, k_shot=1,q_num=4):
    encoder.train()
    tasks = list(set(d.task for d in data_list))

    for episode in range(n_episodes):
        task = random.choice(tasks)
        support_set, query_set = sample_episode(data_list, task, k_shot, q_num)

        support_x, support_y = [], []
        for g in support_set:
            g = g.to(DEVICE)
            emb = encoder(g.x, g.edge_index, g.edge_attr)
            support_x.append(emb)
            support_y.append(g.y)
        support_x = torch.cat(support_x, dim=0)
        support_y = torch.cat(support_y, dim=0)

        prototypes = compute_prototypes(support_x, support_y)

        query = query_set[0].to(DEVICE)
        query_emb = encoder(query.x, query.edge_index, query.edge_attr)
        loss, acc = prototypical_loss(query_emb, query.y, prototypes)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 5 == 0:
            print(f"[Episode {episode}] Loss: {loss.item():.4f} | Accuracy: {acc*100:.2f}% | Task: {task}")

# --------- Inference on a Graph -----------------------
def proto_predict(encoder, support_set, query_graph):
    encoder.eval()
    support_x, support_y = [], []

    for g in support_set:
        g = g.to(DEVICE)
        emb = encoder(g.x, g.edge_index, g.edge_attr)
        support_x.append(emb)
        support_y.append(g.y)

    support_x = torch.cat(support_x, dim=0)
    support_y = torch.cat(support_y, dim=0)
    prototypes = compute_prototypes(support_x, support_y)

    query = query_graph.to(DEVICE)
    query_emb = encoder(query.x, query.edge_index, query.edge_attr)
    dists = euclidean_distance(query_emb, prototypes)
    preds = dists.argmin(dim=1)
    return preds.cpu()

# # --------- Example Runner -----------------------------
print("Loading few-shot dataset...")
data_list = torch.load("data/training_data/training_dataset.pt", map_location=DEVICE, weights_only=False)  # please replace this path with your desired location

encoder = GATEncoder().to(DEVICE)
optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)

print("Starting ProtoNet training...")
proto_train(data_list, encoder, optimizer, n_episodes=250)

print("Saving trained encoder...")
torch.save(encoder.state_dict(), "models/prototypical/proto_gat_encoder.pt")