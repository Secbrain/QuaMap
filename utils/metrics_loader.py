# utils/metrics_loader.py

import os
import json

def load_metrics_from_json(json_path):
    """
    Load a single transpiled circuit's metrics from a JSON file.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def load_all_metrics(metrics_dir, backend_filter=None, qubit_filter=None):
    """
    Load all JSON metric files under a directory.
    Optional:
      - backend_filter: only load files for specified backend
      - qubit_filter: only load circuits with specific qubit count
    """
    records = []
    for root, _, files in os.walk(metrics_dir):
        for file in files:
            if not file.endswith(".json"):
                continue
            path = os.path.join(root, file)
            data = load_metrics_from_json(path)

            # Apply backend filter
            if backend_filter and data.get("backend") != backend_filter:
                continue

            # Apply qubit count filter
            if qubit_filter and len(data.get("layout", [])) != qubit_filter:
                continue

            records.append(data)
    return records

if __name__ == "__main__":
    # Example usage
    metric_dir = "../metrics/"
    all_data = load_all_metrics(metric_dir, backend_filter="ibmq_lima", qubit_filter=4)
    print(f"Loaded {len(all_data)} records.")
    print("Example record:")
    print(all_data[0])
