# maml_evaluator.py

import torch
import torch.nn.functional as F
from torch_geometric.nn import GAT
import higher
import random
from sklearn.metrics import classification_report

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- GAT Builder (same config as maml_runner) ----------
def build_gat_model(hidden=256, heads=8, dropout=0.2, layers=2):
    return GAT(
        in_channels=18,
        hidden_channels=hidden,
        out_channels=4,
        heads=heads,
        num_layers=layers,
        dropout=dropout,
        edge_dim=1,
        v2=True,
        jk='lstm'
    ).to(DEVICE)

# ---------- Episode Sampler ----------
def sample_episode(data_list, task, k_shot=1,q_num=4):
    task_data = [d for d in data_list if getattr(d, 'task', None) == task]
    random.shuffle(task_data)
    return task_data[:k_shot], task_data[k_shot:k_shot + q_num]

# ---------- Inference Logic ----------
def maml_infer(model, support_set, query_doc, optimizer, inner_steps=2):
    model.train()

    with higher.innerloop_ctx(model, optimizer, track_higher_grads=False) as (fmodel, diffopt):
        for _ in range(inner_steps):
            for support in support_set:
                support = support.to(DEVICE)
                out = fmodel(support.x, support.edge_index, edge_weight=support.edge_attr)
                loss = F.cross_entropy(out, support.y)
                diffopt.step(loss)

        query_doc = query_doc.to(DEVICE)
        out = fmodel(query_doc.x, query_doc.edge_index, edge_weight=query_doc.edge_attr)
        pred = out.argmax(dim=1)

    return pred.cpu(), query_doc.y.cpu()

# ---------- Evaluation Loop ----------
def evaluate_model(data_list, model_path):
    print("🔍 Loading model...")
    model = build_gat_model()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    tasks = list(set(d.task for d in data_list))
    all_preds, all_trues = [], []

    for task in tasks:
        support_set, query_set = sample_episode(data_list, task, k_shot=1, q_num=4)

        for query_doc in query_set:
            pred, true = maml_infer(model, support_set, query_doc, optimizer)
            all_preds.append(pred)
            all_trues.append(true)

            unique_labels = set(true.tolist())
            detected_labels = set(pred.tolist())
            print(f" Task: {task}")
            print(f"  True labels present: {sorted(unique_labels)}")
            print(f"  Predicted labels present: {sorted(detected_labels)}")
            print(classification_report(true, pred, zero_division=0))
            missing_labels = unique_labels - detected_labels
            if missing_labels:
                print(f" Missing labels in prediction: {sorted(missing_labels)}")
            else:
                print(f" All true labels detected in prediction.")

            unique_classes = set(true.tolist() + pred.tolist())
            for cls in unique_classes:
                n_pred = (pred == cls).sum().item()
                n_true = (true == cls).sum().item()
                print(f"Nodes of type {cls} predicted: {n_pred} | True: {n_true}")


    

    # Print overall classification report
    all_preds_flat = torch.cat(all_preds).numpy()
    all_trues_flat = torch.cat(all_trues).numpy()
    print("\n===== Overall Classification Report (all tasks) =====")
    print(classification_report(all_trues_flat, all_preds_flat, zero_division=0))

    return all_preds, all_trues

 
if __name__ == "__main__":
    data_list = torch.load("data/test_data/test_dataset_loan.pt", map_location=DEVICE)
    evaluate_model(data_list, model_path="models/maml_gat_model.pt")
