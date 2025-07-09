# # import networkx as nx
# # import numpy as np
# # import pandas as pd
# # from sklearn.preprocessing import MinMaxScaler
# # from torch_geometric.data import Data
# # import torch
# # import json
# # import ast
# # import os
# #
# # # === Update this list with your PDF numbers ===
# # # doc_ids = [
# # #     "2", "11", "12", "15", "16", "17", "18", "13", "14", "1",
# # #     "5", "6", "7", "8", "9", "10"
# # # ]
# # doc_ids = [
# #   "1","2","5","6","8","9","10","11","12","13","14","15","16","17","18"
# # ]
# # # === Folder Paths (update if needed) ===
# # base_path = r"E:\Graph_data_generation\Value_type_Corrected_csvs_for_training"
# # output_path = "DatacheckpointNew_Training.pt"
# #
# # # === Label Encoding ===
# # label_encoding = {
# #     "KEY": 0,
# #     "VALUE": 1,
# #     "OTHER_KEY": 2,
# #     "NON_RELATED": 3
# # }
# #
# # with open("label_encoding.json", "w") as f:
# #     json.dump(label_encoding, f, indent=4)
# #
# # # === Extract node features ===
# # def get_custom_node_features(node):
# #     value_type_str = node.get("ValueType", "[0]*9")
# #     try:
# #         value_type = [float(x) for x in ast.literal_eval(value_type_str)]
# #     except:
# #         value_type = [0.0] * 9
# #
# #     other_features = [
# #         float(node.get("IsVerticalNeighbourKey", 0)),
# #         float(node.get("IsHorizontalNeighbourKey", 0)),
# #         float(node.get("right_spacing", 0)),
# #         float(node.get("left_spacing", 0)),
# #         float(node.get("EndsWithColon", 0)),
# #         float(node.get("IsRightNeighbour",0)),
# #         float(node.get("IsLeftNeighbour", 0)),
# #         float(node.get("IsAboveNeighbour", 0)),
# #         float(node.get("IsBelowNeighbour", 0))
# #
# #     ]
# #
# #     return other_features + value_type
# #
# # # === Process one graphml+csv pair ===
# # def process_document_graph(graphml_file, csv_file):
# #     g = nx.read_graphml(graphml_file)
# #
# #     label_df = pd.read_csv(csv_file)
# #     print("label_df",label_df.to_string())
# #     label_df["Id"] = label_df["Id"].astype(str)
# #     label_map = dict(zip(label_df["Id"], label_df["Label"]))
# #
# #     node_ids = list(g.nodes)
# #     node_features = []
# #     node_labels = []
# #
# #     for node_id in node_ids:
# #         node_data = g.nodes[node_id]
# #         node_features.append(get_custom_node_features(node_data))
# #
# #         label_str = label_map.get(node_id, "NON_RELATED")
# #         label_index = label_encoding.get(label_str, label_encoding["NON_RELATED"])
# #         node_labels.append(label_index)
# #
# #     edge_index = []
# #     raw_distances = []
# #     node_id_to_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
# #
# #     for src, tgt in g.edges:
# #         src_data = g.nodes[src]
# #         tgt_data = g.nodes[tgt]
# #
# #         x1 = (float(src_data.get("xmin", 0)) + float(src_data.get("xmax", 0))) / 2
# #         y1 = (float(src_data.get("ymin", 0)) + float(src_data.get("ymax", 0))) / 2
# #         x2 = (float(tgt_data.get("xmin", 0)) + float(tgt_data.get("xmax", 0))) / 2
# #         y2 = (float(tgt_data.get("ymin", 0)) + float(tgt_data.get("ymax", 0))) / 2
# #
# #         dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
# #         raw_distances.append(dist)
# #         edge_index.append((node_id_to_index[src], node_id_to_index[tgt]))
# #
# #     norm_distances = MinMaxScaler().fit_transform(np.array(raw_distances).reshape(-1, 1)) if raw_distances else []
# #     edge_attr = (1.0 - np.array(norm_distances)).tolist() if raw_distances else []
# #
# #     data = Data(
# #         x=torch.tensor(np.array(node_features), dtype=torch.float),
# #         edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.empty((2, 0), dtype=torch.long),
# #         edge_attr=torch.tensor(edge_attr, dtype=torch.float32) if edge_attr else None,
# #         y=torch.tensor(node_labels, dtype=torch.long)
# #     )
# #
# #     return data
# #
# # # === Main: Process all documents and combine ===
# # data_list = []
# #
# # for doc_id in doc_ids:
# #     graphml_file = os.path.join(base_path, f"{doc_id}.graphml")
# #     csv_file = os.path.join(base_path, f"{doc_id}.csv")
# #
# #     print(f"📄 Processing {doc_id}.graphml + {doc_id}.csv")
# #     data = process_document_graph(graphml_file, csv_file)
# #     data_list.append(data)
# #
# # torch.save(data_list, output_path)
# # print(f"\n✅ Saved combined {len(data_list)} documents to '{output_path}'")
import os
import json
import ast
import pandas as pd
import numpy as np
import torch
import networkx as nx
from torch_geometric.data import Data
from sklearn.preprocessing import MinMaxScaler
#
# # === Document IDs to Process ===
# # doc_ids = [
# #     "1", "2", "5", "6","7","8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18"
# # ]
# #
# # # === Paths ===
# # base_path = r"E:\Graph_data_generation\Value_type_Corrected_csvs_for_training"
# # output_path = "DatacheckpointNew_Training.pt"
#
# # === Label Encoding ===
# label_encoding = {
#     "KEY": 0,
#     "VALUE": 1,
#     "OTHER_KEY": 2,
#     "NON_RELATED": 3
# }
#
# # Save label encoding for reference
# with open("label_encoding.json", "w") as f:
#     json.dump(label_encoding, f, indent=4)
#
# # === Function: Extract node features ===
# def get_custom_node_features(node):
#     value_type_str = node.get("ValueType", "[0]*9")
#     try:
#         value_type = [float(x) for x in ast.literal_eval(value_type_str)]
#     except Exception:
#         value_type = [0.0] * 9
#
#     other_features = [
#         float(node.get("IsVerticalNeighbourKey", 0)),
#         float(node.get("IsHorizontalNeighbourKey", 0)),
#         float(node.get("right_spacing", 0)),
#         float(node.get("left_spacing", 0)),
#         float(node.get("EndsWithColon", 0)),
#         float(node.get("IsRightNeighbour", 0)),
#         float(node.get("IsLeftNeighbour", 0)),
#         float(node.get("IsAboveNeighbour", 0)),
#         float(node.get("IsBelowNeighbour", 0))
#     ]
#
#     return other_features + value_type
#
# # === Function: Process a single graphml + csv file ===
# def process_document_graph(graphml_file, csv_file):
#     g = nx.read_graphml(graphml_file)
#
#     label_df = pd.read_csv(csv_file)
#     label_df["Id"] = label_df["Id"].astype(str)
#     label_map = dict(zip(label_df["Id"], label_df["Label"]))
#     print(label_map)
#
#     node_ids = list(g.nodes)
#     node_features = []
#     node_labels = []
#
#     for node_id in node_ids:
#         node_data = g.nodes[node_id]
#         node_features.append(get_custom_node_features(node_data))
#         label_str = label_map.get(node_id, "NON_RELATED")
#         print("label_str",label_str)
#         label_index = label_encoding.get(label_str, label_encoding["NON_RELATED"])
#         node_labels.append(label_index)
#
#     edge_index = []
#     raw_distances = []
#     node_id_to_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
#
#     for src, tgt in g.edges:
#         src_data = g.nodes[src]
#         tgt_data = g.nodes[tgt]
#
#         x1 = (float(src_data.get("xmin", 0)) + float(src_data.get("xmax", 0))) / 2
#         y1 = (float(src_data.get("ymin", 0)) + float(src_data.get("ymax", 0))) / 2
#         x2 = (float(tgt_data.get("xmin", 0)) + float(tgt_data.get("xmax", 0))) / 2
#         y2 = (float(tgt_data.get("ymin", 0)) + float(tgt_data.get("ymax", 0))) / 2
#
#         dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
#         raw_distances.append(dist)
#         edge_index.append((node_id_to_index[src], node_id_to_index[tgt]))
#
#     if raw_distances:
#         norm_distances = MinMaxScaler().fit_transform(np.array(raw_distances).reshape(-1, 1))
#         edge_attr = (1.0 - np.array(norm_distances)).tolist()
#     else:
#         edge_attr = []
#         edge_index = []
#
#     data = Data(
#         x=torch.tensor(np.array(node_features), dtype=torch.float),
#         edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.empty((2, 0), dtype=torch.long),
#         edge_attr=torch.tensor(edge_attr, dtype=torch.float32) if edge_attr else None,
#         y=torch.tensor(node_labels, dtype=torch.long)
#     )
#
#     return data
#
# # # === Main: Process all documents and combine ===
# # data_list = []
# #
# # for doc_id in doc_ids:
# #     graphml_file = os.path.join(base_path, f"{doc_id}.graphml")
# #     csv_file = os.path.join(base_path, f"{doc_id}.csv")
# #
# #     if os.path.exists(graphml_file) and os.path.exists(csv_file):
# #         print(f"📄 Processing {doc_id}.graphml + {doc_id}.csv")
# #         data = process_document_graph(graphml_file, csv_file)
# #         data_list.append(data)
# #     else:
# #         print(f"⚠️ Missing file for doc_id={doc_id}, skipping.")
# #
# # # === Save the Combined Data ===
# # torch.save(data_list, output_path)
# # print(f"\n✅ Saved combined {len(data_list)} documents to '{output_path}'")

import os

# === Update this list with your PDF numbers ===
# doc_ids = [
#     "2", "11", "12", "15", "16", "17", "18", "13", "14", "1",
#     "5", "6", "7", "8", "9", "10"
# ]
# doc_ids = [
#   "1","2","5","6","8","9","10","11","12","13","14","15","16","17","18"
# ]

doc_ids = [
  "14","15","16","17","18","19","20","21","22","23"
]

# === Folder Paths (update if needed) ===
base_path = r"E:\AnyDocTesting_HC256_L2_DO0.2_H8\UNIT TESTING\INVOICE\FAILED SAMPLES-INVOICE SAMPLES-KARL(113 No.)\Failed_sample_results_finetuning"
output_path = "DatacheckpointFinetuning.pt"

# === Label Encoding ===
label_encoding = {
    "KEY": 0,
    "VALUE": 1,
    "OTHER_KEY": 2,
    "NON_RELATED": 3
}

with open("label_encoding.json", "w") as f:
    json.dump(label_encoding, f, indent=4)

# === Extract node features ===
def get_custom_node_features(node):
    value_type_str = node.get("ValueType", "[0]*9")
    try:
        value_type = [float(x) for x in ast.literal_eval(value_type_str)]
    except:
        value_type = [0.0] * 9

    other_features = [
        float(node.get("IsVerticalNeighbourKey", 0)),
        float(node.get("IsHorizontalNeighbourKey", 0)),
        float(node.get("right_spacing", 0)),
        float(node.get("left_spacing", 0)),
        float(node.get("EndsWithColon", 0)),
        float(node.get("IsRightNeighbour",0)),
        float(node.get("IsLeftNeighbour", 0)),
        float(node.get("IsAboveNeighbour", 0)),
        float(node.get("IsBelowNeighbour", 0))

    ]

    return other_features + value_type

# === Process one graphml+csv pair ===
def process_document_graph(graphml_file, csv_file):
    g = nx.read_graphml(graphml_file)

    label_df = pd.read_csv(csv_file)
    print("label_df",label_df.to_string())
    label_df["Id"] = label_df["Id"].astype(str)
    label_map = dict(zip(label_df["Id"], label_df["Label"]))

    node_ids = list(g.nodes)
    node_features = []
    node_labels = []

    for node_id in node_ids:
        node_data = g.nodes[node_id]
        node_features.append(get_custom_node_features(node_data))

        label_str = label_map.get(node_id, "NON_RELATED")
        label_index = label_encoding.get(label_str, label_encoding["NON_RELATED"])
        node_labels.append(label_index)

    edge_index = []
    raw_distances = []
    node_id_to_index = {node_id: idx for idx, node_id in enumerate(node_ids)}

    for src, tgt in g.edges:
        src_data = g.nodes[src]
        tgt_data = g.nodes[tgt]

        x1 = (float(src_data.get("xmin", 0)) + float(src_data.get("xmax", 0))) / 2
        y1 = (float(src_data.get("ymin", 0)) + float(src_data.get("ymax", 0))) / 2
        x2 = (float(tgt_data.get("xmin", 0)) + float(tgt_data.get("xmax", 0))) / 2
        y2 = (float(tgt_data.get("ymin", 0)) + float(tgt_data.get("ymax", 0))) / 2

        dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        raw_distances.append(dist)
        edge_index.append((node_id_to_index[src], node_id_to_index[tgt]))

    norm_distances = MinMaxScaler().fit_transform(np.array(raw_distances).reshape(-1, 1)) if raw_distances else []
    edge_attr = (1.0 - np.array(norm_distances)).tolist() if raw_distances else []

    data = Data(
        x=torch.tensor(np.array(node_features), dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32) if edge_attr else None,
        y=torch.tensor(node_labels, dtype=torch.long)
    )

    return data

# === Main: Process all documents and combine ===
data_list = []

for doc_id in doc_ids:
    graphml_file = os.path.join(base_path, f"{doc_id}.graphml")
    csv_file = os.path.join(base_path, f"{doc_id}.csv")

    print(f"📄 Processing {doc_id}.graphml + {doc_id}.csv")
    data = process_document_graph(graphml_file, csv_file)
    data_list.append(data)

torch.save(data_list, output_path)
print(f"\n✅ Saved combined {len(data_list)} documents to '{output_path}'")