import torch
import torch.nn.functional as F
import random
import higher
from torch_geometric.nn import GAT

# ---------------- Config ------------------
IN_CHANNELS = 18
OUT_CLASSES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Model -------------------
def build_gat_model(hidden=256, heads=8, dropout=0.2, layers=2):
    return GAT(
        in_channels=IN_CHANNELS,
        hidden_channels=hidden,
        out_channels=OUT_CLASSES,
        heads=heads,
        num_layers=layers,
        dropout=dropout,
        edge_dim=1,
        v2=True,
        jk='lstm'
    ).to(DEVICE)

# ---------------- MAML Training ------------
def sample_episode(data_list, task, k_shot=2, q_num=3):
    task_data = [d for d in data_list if getattr(d, 'task', None) == task]
    random.shuffle(task_data)
    return task_data[:k_shot], task_data[k_shot:k_shot + q_num]

def maml_train(data_list, model, optimizer, inner_steps=1, n_episodes=500):
    model.train()
    tasks = list(set(d.task for d in data_list))

    for episode in range(n_episodes):
        task = random.choice(tasks)
        support_set, query_set = sample_episode(data_list, task)
        model.zero_grad()

        with torch.backends.cudnn.flags(enabled=False):
            with higher.innerloop_ctx(model, optimizer, copy_initial_weights=False) as (fmodel, diffopt):
                for _ in range(inner_steps):
                    for support in support_set:
                        support = support.to(DEVICE)
                        out = fmodel(support.x, support.edge_index, edge_weight=support.edge_attr)
                        loss = F.cross_entropy(out, support.y)
                        diffopt.step(loss)

                query = query_set[0].to(DEVICE)
                out = fmodel(query.x, query.edge_index, edge_weight=query.edge_attr)
                loss = F.cross_entropy(out, query.y)
                loss.backward()
                optimizer.step()

        if episode % 10 == 0:
            print(f"[Episode {episode}] Meta-loss: {loss.item():.4f} | Task: {task}")

# ---------------- Runner -------------------
if __name__ == "__main__":
    print("📦 Loading dataset...")
    data_list = torch.load("data/few-shot-dataset/fewshot_dataset.pt", map_location=DEVICE)

    print("🧠 Building model...")
    model = build_gat_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("🏃 Starting training...")
    maml_train(data_list, model, optimizer, inner_steps=1, n_episodes=500)

    print("💾 Saving model...")
    torch.save(model.state_dict(), "models/maml_gat_model.pt")
    print("✅ Model saved to models/maml_gat_model.pt")
