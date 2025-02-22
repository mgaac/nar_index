from mlx.utils import tree_flatten
import argparse
import pandas as pd
import numpy as np
import json

# Helper: Compute activation statistics per layer
def compute_activation_stats(activations):
    stats = {}
    for act_key, act_tensor in activations.items():
        # Convert MLX array to NumPy array using np.array()
        act_np = np.array(act_tensor)
        hist, _ = np.histogram(act_np, bins=10)
        stats[f"{act_key}_histogram"] = hist.tolist()
        stats[f"{act_key}_std"] = float(np.std(act_np))
        #saturated = np.sum((act_np >= 0.99) | (act_np <= -0.99))
        stats[f"{act_key}_norm"] = float(np.linalg.norm(act_np))
    return stats

# Helper: Compute gradient statistics using tree_flatten to iterate over all gradients
def compute_gradient_stats(grads):
    stats = {}
    flat_grads = tree_flatten(grads)
    for key, grad in flat_grads:
        grad_np = np.array(grad)  # Convert to NumPy array
        stats[f"{key}_grad_norm"] = float(np.linalg.norm(grad_np))
        stats[f"{key}_grad_variance"] = float(np.var(grad_np))
        hist, _ = np.histogram(grad_np, bins=10)
        stats[f"{key}_grad_histogram"] = hist.tolist()
    return stats

# Helper: Compute parameter statistics using tree_flatten to iterate over all parameters
def compute_param_stats(params):
    stats = {}
    flat_params = tree_flatten(params)
    for key, param in flat_params:
        param_np = np.array(param)  # Convert to NumPy array
        stats[f"{key}_weight_norm"] = float(np.linalg.norm(param_np))
        hist, _ = np.histogram(param_np, bins=10)
        stats[f"{key}_weight_histogram"] = hist.tolist()
    return stats

# Helper: Compute weight update ratios by comparing current parameters to previous ones
def compute_update_ratios(prev_params, current_params):
    update_ratios = {}
    flat_prev = tree_flatten(prev_params)
    flat_curr = tree_flatten(current_params)
    for (key_prev, prev_val), (key_curr, curr_val) in zip(flat_prev, flat_curr):
        prev_np = np.array(prev_val)
        curr_np = np.array(curr_val)
        update = np.linalg.norm(curr_np - prev_np)
        param_norm = np.linalg.norm(prev_np) + 1e-8
        update_ratios[f"{key_curr}_update_ratio"] = float(update / param_norm)
    return update_ratios

def get_table_scalar_metric(csv_file, column_name, metric_name):
    """
    Reads a scalar metric (like loss or accuracy) from the CSV log.
    Returns a DataFrame with one row (named by metric_name) and columns corresponding to iterations.
    """
    df = pd.read_csv(csv_file)
    df['Iteration'] = df['Iteration'].astype(int)
    data = {}
    for _, row in df.iterrows():
        iteration = row['Iteration']
        data[iteration] = row[column_name]
    # Create a DataFrame with a single row (the metric name) and sorted iteration columns.
    df_table = pd.DataFrame([data], index=[metric_name])
    df_table = df_table.reindex(sorted(df_table.columns), axis=1)
    return df_table

def get_table_metric(csv_file, column_name, metric_suffix):
    """
    Reads the CSV log and extracts a metric (e.g., weight_norm, grad_norm, update_ratio, std, saturation_pct)
    for each layer.

    Returns a DataFrame with rows = layer names, columns = iterations, and values = metric value.
    """
    df = pd.read_csv(csv_file)
    df['Iteration'] = df['Iteration'].astype(int)
    table = {}

    for _, row in df.iterrows():
        iteration = row["Iteration"]
        try:
            data = json.loads(row[column_name])
        except (TypeError, json.JSONDecodeError):
            continue
        for key, value in data.items():
            if key.endswith(metric_suffix):
                # Remove the trailing "_" plus the metric_suffix to extract the parameter/activation name.
                param_name = key[:-len(metric_suffix)-1]
                if param_name not in table:
                    table[param_name] = {}
                table[param_name][iteration] = value

    # Create a DataFrame: rows = parameter names, columns = iterations.
    df_table = pd.DataFrame(table).T
    # Ensure the columns (iterations) are sorted.
    df_table = df_table.reindex(sorted(df_table.columns), axis=1)
    return df_table

def table_weight_norm(csv_file, column_name="Parameter Stats"):
    """Creates a table of weight norms per layer over iterations."""
    return get_table_metric(csv_file, column_name, "weight_norm")

def get_table_activation_threshold(csv_file, column_name="Activation Stats", threshold=0.9):
    """
    Computes, for each layer, how many activation values exceed the given threshold.
    It assumes keys in the JSON include a sequence prefix (e.g. "seq0_") that is removed.

    Returns a DataFrame with rows = layer names and columns = iterations.
    """
    df = pd.read_csv(csv_file)
    df['Iteration'] = df['Iteration'].astype(int)
    table = {}
    for _, row in df.iterrows():
        iteration = row["Iteration"]
        try:
            data = json.loads(row[column_name])
        except (TypeError, json.JSONDecodeError):
            continue
        counts_per_layer = {}
        for key, value in data.items():
            try:
                act_array = np.array(value)
            except Exception:
                continue
            count = np.sum(act_array > threshold)
            parts = key.split("_", 1)
            layer_name = parts[1] if len(parts) > 1 else key
            counts_per_layer[layer_name] = counts_per_layer.get(layer_name, 0) + int(count)
        for layer, count in counts_per_layer.items():
            if layer not in table:
                table[layer] = {}
            table[layer][iteration] = count
    df_table = pd.DataFrame(table).T
    df_table = df_table.reindex(sorted(df_table.columns), axis=1)
    return df_table

def get_table_activation_std(csv_file, column_name="Activation Stats"):
    """Extracts activation standard deviations (keys ending in 'std') from the CSV log."""
    return get_table_metric(csv_file, column_name, "std")

def get_table_activation_norm(csv_file, column_name="Activation Stats"):
    """Extracts activation saturation percentages (keys ending in 'saturation_pct') from the CSV log."""
    return get_table_metric(csv_file, column_name, "norm")

def get_table_grad_metric(csv_file, column_name="Gradient Stats", metric_suffix="grad_norm"):
    """
    Extracts a gradient metric (e.g., grad_norm or grad_variance) from the CSV log.
    Returns a DataFrame with rows = parameter names and columns = iterations.
    """
    return get_table_metric(csv_file, column_name, metric_suffix)

def table_weight_update_ratio(csv_file, column_name="Weight Update Ratio"):
    """
    Extracts the weight update ratio from the CSV log.
    It expects keys ending in 'update_ratio'.
    Returns a DataFrame with rows = parameter names and columns = iterations.
    """
    return get_table_metric(csv_file, column_name, "update_ratio")

def table_loss(csv_file, column_name="Training Loss"):
    """Creates a table of training loss over iterations."""
    return get_table_scalar_metric(csv_file, column_name, metric_name="loss")

def table_acc_score(csv_file, column_name="Accuracy Score"):
    """Creates a table of accuracy scores over iterations."""
    return get_table_scalar_metric(csv_file, column_name, metric_name="acc")

# --- New functions for Prediction vs Target visualization ---

def table_pred_target(csv_file, column_name="Prediction vs Target"):
    """
    Reads the CSV log and extracts the Prediction vs Target pairs for each logged iteration.
    Returns a DataFrame with index as iteration and two columns: 'predictions' and 'targets'.
    """
    df = pd.read_csv(csv_file)
    if column_name not in df.columns:
        print(f"Column '{column_name}' not found in CSV.")
        return pd.DataFrame()

    data = []
    for _, row in df.iterrows():
        iteration = row["Iteration"]
        try:
            pred_target = json.loads(row[column_name])
            predictions = pred_target.get("predictions", [])
            targets = pred_target.get("targets", [])
        except Exception as e:
            predictions = []
            targets = []
        data.append({"Iteration": iteration, "predictions": predictions, "targets": targets})

    df_table = pd.DataFrame(data).set_index("Iteration")
    df_table = df_table.sort_index()
    return df_table

def print_pred_target_table(csv_file, column_name="Prediction vs Target"):
    """
    Prints a formatted view of Prediction vs Target data for each iteration from the CSV log.
    """
    df = table_pred_target(csv_file, column_name=column_name)
    if df.empty:
        print("No Prediction vs Target data available.")
        return

    print("\nPrediction vs Target Table")
    print("=" * 30)
    for iteration, row in df.iterrows():
        # Convert list entries to space-separated strings.
        pred_str = " ".join(str(p) for p in row["predictions"])
        targ_str = " ".join(str(t) for t in row["targets"])
        print(f"Iteration {iteration}:")
        print("  Predictions: " + pred_str)
        print("  Targets:     " + targ_str)
        print("-" * 30)

def print_table_with_title(df_table, title, max_cols=10):
    """
    Prints the DataFrame with a title. If there are more than max_cols columns,
    only the first half and the last half are printed with an ellipsis column in between.
    """
    print("\n" + title)
    print("=" * len(title))
    if df_table.empty:
        print("No data available.\n")
        return

    num_cols = len(df_table.columns)
    if num_cols > max_cols:
        half = max_cols // 2
        first_cols = list(df_table.columns[:half])
        last_cols = list(df_table.columns[-half:])
        df_trunc = pd.concat([df_table[first_cols], df_table[last_cols]], axis=1)
        # Insert a column of ellipsis in the middle.
        df_trunc.insert(half, '...', ['...'] * len(df_trunc))
        print(df_trunc.to_string())
    else:
        print(df_table.to_string())
    print("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", type=str)
    args = parser.parse_args()
    csv_file = args.csv_file # Update the path if needed

    loss_table = table_loss(csv_file, column_name="Training Loss")
    print_table_with_title(loss_table, "Loss Table (per layer over iterations)")

    acc_table = table_acc_score(csv_file, column_name="Accuracy Score")
    print_table_with_title(acc_table, "Evaluation Table (per layer over iterations)")

    weight_norm_table = table_weight_norm(csv_file, column_name="Parameter Stats")
    print_table_with_title(weight_norm_table, "Weight Norm Table (per layer over iterations)")

    grad_norm_table = get_table_grad_metric(csv_file, column_name="Gradient Stats", metric_suffix="grad_norm")
    print_table_with_title(grad_norm_table, "Gradient Norm Table (per layer over iterations)")

    activation_std_table = get_table_activation_std(csv_file, column_name="Activation Stats")
    print_table_with_title(activation_std_table, "Activation Std Table (per layer over iterations)")

    grad_variance_table = get_table_grad_metric(csv_file, column_name="Gradient Stats", metric_suffix="grad_variance")
    print_table_with_title(grad_variance_table, "Gradient Variance Table (per layer over iterations)")

    weight_update_ratio_table = table_weight_update_ratio(csv_file, column_name="Weight Update Ratio")
    print_table_with_title(weight_update_ratio_table, "Weight Update Ratio Table (per layer over iterations)")

    activation_norm_table = get_table_activation_norm(csv_file, column_name="Activation Stats")
    print_table_with_title(activation_norm_table, "Activation Norm Table (per layer over iterations)")
