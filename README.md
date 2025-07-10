# GAT-test
testing phase...
#  Few-Shot GAT-Based Information Extraction Pipeline

This project uses Graph Attention Networks (GAT) with a MAML-based meta-learning loop to extract values from structured documents like invoices, loans, and reports in a few-shot setting.

---

##  Folder Structure & Dataset Organization

Each type of document is stored in its own subdirectory inside a `data/` folder. Each `.pt` file inside these folders is a PyTorch Geometric `Data` object representing one document graph.

```
data/
├── invoice/
│   ├── invoice_1.pt
│   ├── invoice_2.pt
│   └── ...
├── loan/
│   ├── loan_1.pt
│   └── ...
├── final_bill/
├── background_verification/
├── operative_report/
```

* Each `.pt` file contains:

  * `x`: node features (shape: \[N, 18])
  * `edge_index`: edge list (shape: \[2, E])
  * `edge_attr`: optional edge features (shape: \[E, 1])
  * `y`: node-level labels (0 = KEY, 1 = VALUE, 2 = OTHER\_KEY, 3 = NON\_RELATED)

---

##  Task Assignment

Each graph file is tagged with a `.task` attribute based on its folder name (e.g., `"Invoice"`).
This is done automatically in the dataset builder script.

```python
# Example:
data.task = "Invoice"
```

This `.task` field is used to:

* Sample few-shot episodes (support/query)
* Simulate K-shot learning problems during MAML training

---

## Workflow Summary

1. Prepare `.pt` files per document type in `data/<task_name>/`
2. Run dataset builder to tag `.task` and combine into one `fewshot_dataset.pt`
3. Train using `maml_runner.py`
4. Evaluate using `maml_evaluator.py`

---

## Supported Document Types

* Invoice
* Loan
* Final Bill
* Background Verification
* Operative Report

You can add more simply by adding new folders to `data/` and placing `.pt` graphs inside.

---

Need help creating graphs or preprocessing data? See `graph_builder.py` (optional helper coming soon).
