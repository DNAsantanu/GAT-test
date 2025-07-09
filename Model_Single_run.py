import json
import torch
import matplotlib.pyplot as plt
import pandas as pd
from torch.nn import CrossEntropyLoss
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import GAT
import os
from torch_geometric.data import Data
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
import torch.serialization


def smooth_curve(data, weight=0.9):
    smoothed = []
    last = data[0]
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def train_single_config(config, train_loader, val_loader, in_channels, num_classes, run_name, model_dir, results_dir, plots_dir):
    model = GAT(
        in_channels=in_channels,
        hidden_channels=config['hidden_channels'],
        num_layers=config['num_layers'],
        out_channels=num_classes,
        dropout=config['dropout'],
        heads=config['heads'],
        v2=True,
        edge_dim=1,
        jk='lstm'
    )

    all_labels = torch.cat([data.y for data in train_loader.dataset])
    class_counts = torch.bincount(all_labels, minlength=num_classes)
    class_weights = 1.0 / (class_counts.float() + 1e-6)
    class_weights = class_weights / class_weights.sum()

    criterion = CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    training_loss, validation_loss, validation_acc = [], [], []

    for epoch in range(500):
        model.train()
        total_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, edge_weight=data.edge_attr)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        training_loss.append(avg_train_loss)

        model.eval()
        val_total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data in val_loader:
                out = model(data.x, data.edge_index, edge_weight=data.edge_attr)
                loss = criterion(out, data.y)
                val_total_loss += loss.item()
                pred = out.argmax(dim=1)
                correct += (pred == data.y).sum().item()
                total += data.y.size(0)

        avg_val_loss = val_total_loss / len(val_loader)
        val_accuracy = correct / total
        validation_loss.append(avg_val_loss)
        validation_acc.append(val_accuracy)

        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch + 1:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f}")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    model_path = os.path.join(model_dir, f"{run_name}.pth")
    csv_path = os.path.join(results_dir, f"{run_name}.csv")
    plot_path = os.path.join(plots_dir, f"{run_name}.png")

    torch.save(model.state_dict(), model_path)

    df = pd.DataFrame({
        'Epoch': list(range(1, len(training_loss)+1)),
        'TrainLoss': training_loss,
        'ValLoss': validation_loss,
        'ValAcc': validation_acc
    })
    df.to_csv(csv_path, index=False)

    plt.figure()
    plt.plot(smooth_curve(training_loss), label='Train')
    plt.plot(validation_loss, label='Val')
    plt.title(run_name)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(plot_path)
    plt.close()


if __name__ == "__main__":

    with torch.serialization.safe_globals([Data, DataEdgeAttr, DataTensorAttr, GlobalStorage]):
        data_list = torch.load(f"E:/Graph_data_generation/DatacheckpointNew_Training.pt", map_location='cpu')

    labels = json.load(open("E:/DatasetGeneration_Updated/label_encoding.json"))
    batch_size = 1

    train_split = int(len(data_list) * 0.8)
    train_data = data_list[:train_split]
    val_data = data_list[train_split:]

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    in_channels = data_list[0].x.size(1)
    num_classes = len(labels)

    # 🔧 Use only one configuration here:
    config = {
        'hidden_channels':256,
        'num_layers': 2,
        'heads':8,
        'dropout': 0.2
    }

    run_name = f"SingleRun_H{config['hidden_channels']}_L{config['num_layers']}_HD{config['heads']}_DO{int(config['dropout']*10)}_Updated"

    model_dir = "E:/GATv2Experiments/models"
    results_dir = "E:/GATv2Experiments/results"
    plots_dir = "E:/GATv2Experiments/plots"

    print(f"\n🚀 Starting {run_name}")
    train_single_config(config, train_loader, val_loader, in_channels, num_classes, run_name, model_dir, results_dir, plots_dir)
