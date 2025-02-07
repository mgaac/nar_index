import matplotlib.pyplot as plt
import numpy as np

def visualize_training_history(loss_history, activation_history, gradient_history):
    """
    Visualizes training loss, activations, and gradients over the training iterations.

    Parameters:
      loss_history: list of loss values per iteration.
      activation_history: list of dicts (per iteration) containing activations.
      gradient_history: list of dicts (per iteration) containing gradient information.
    """

    # --- Set the style to ensure a white background ---
    plt.style.use('default')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'

    iterations = np.arange(len(loss_history))

    # 1. Plot the training loss.
    plt.figure(figsize=(10, 4), facecolor='white')
    plt.plot(iterations, loss_history, label="Loss", color='tab:blue')
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Iterations")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2. Plot evolution of each activation (mean and std).
    for key in activation_history[0].keys():
        values = []
        for act in activation_history:
            if key in act:
                try:
                    value_np = np.array(act[key]).flatten()
                    values.append(value_np)
                except Exception as e:
                    print(f"Skipping key '{key}' due to conversion error: {e}")
                    continue
        if not values:
            continue
        means = np.array([np.mean(v) for v in values])
        stds  = np.array([np.std(v) for v in values])
        plt.figure(figsize=(10, 4), facecolor='white')
        plt.plot(iterations, means, label=f"Mean of {key}", marker='o')
        plt.fill_between(iterations, means - stds, means + stds, alpha=0.2, label="Std Dev")
        plt.xlabel("Iteration")
        plt.ylabel("Activation Value")
        plt.title(f"Evolution of Activation: {key}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # 3. Special treatment for attention weights "w"
    if "w" in activation_history[0]:
        w_history = []
        for act in activation_history:
            if "w" in act:
                try:
                    w_np = np.array(act["w"]).flatten()
                    w_history.append(w_np)
                except Exception as e:
                    print(f"Skipping 'w' due to conversion error: {e}")
                    continue
        if w_history:
            w_history = np.array(w_history)  # Shape: (iterations, memory_size)
            plt.figure(figsize=(10, 4), facecolor='white')
            for i in range(w_history.shape[1]):
                plt.plot(iterations, w_history[:, i], label=f"w[{i}]")
            plt.xlabel("Iteration")
            plt.ylabel("Weight Value")
            plt.title("Evolution of Addressing Weights 'w'")
            plt.legend(ncol=2)
            plt.tight_layout()
            plt.show()

    # 4. Plot gradient norms over iterations.
    # We expect gradient_history to be a list of dictionaries with keys: "memory", "shared_layers", etc.
    if not gradient_history or not isinstance(gradient_history[0], dict):
        print("Skipping gradient visualization because gradient entries are not dictionaries.")
    else:
        for key in gradient_history[0].keys():
            if key == "memory":
                norms = []
                for grad in gradient_history:
                    if isinstance(grad, dict) and key in grad:
                        try:
                            g = np.array(grad[key])
                            norms.append(np.linalg.norm(g))
                        except Exception as e:
                            norms.append(np.nan)
                    else:
                        norms.append(np.nan)
                plt.figure(figsize=(10, 4), facecolor='white')
                plt.plot(iterations, norms, label=f"Gradient norm of {key}", color='tab:orange', marker='o')
                plt.xlabel("Iteration")
                plt.ylabel("L2 Norm")
                plt.title(f"Gradient Norm Evolution: {key}")
                plt.legend()
                plt.tight_layout()
                plt.show()
            else:
                # For keys such as "shared_layers", "controller_layers", and "out_layers"
                blocks = gradient_history[0][key]
                num_blocks = len(blocks)
                for block_idx in range(num_blocks):
                    block_example = blocks[block_idx]
                    # If the block is a Sequential module, expect a nested "layers" list.
                    if isinstance(block_example, dict) and "layers" in block_example:
                        sublayers = block_example["layers"]
                        num_sublayers = len(sublayers)
                        for sublayer_idx in range(num_sublayers):
                            norms = []
                            for grad in gradient_history:
                                try:
                                    block_grad = grad[key][block_idx]
                                    sublayer_grad = block_grad["layers"][sublayer_idx]
                                    norm_val = 0
                                    if "weight" in sublayer_grad:
                                        norm_val += np.linalg.norm(np.array(sublayer_grad["weight"]))
                                    if "bias" in sublayer_grad:
                                        norm_val += np.linalg.norm(np.array(sublayer_grad["bias"]))
                                    norms.append(norm_val)
                                except Exception as e:
                                    norms.append(np.nan)
                            plt.figure(figsize=(10, 4), facecolor='white')
                            plt.plot(iterations, norms,
                                     label=f"Gradient norm of {key} block {block_idx} layer {sublayer_idx}",
                                     marker='o')
                            plt.xlabel("Iteration")
                            plt.ylabel("L2 Norm")
                            plt.title(f"Gradient Norm Evolution: {key} block {block_idx} layer {sublayer_idx}")
                            plt.legend()
                            plt.tight_layout()
                            plt.show()
                    else:
                        # Fallback: if the block isn't structured as a Sequential module.
                        norms = []
                        for grad in gradient_history:
                            try:
                                subgrad = grad[key][block_idx]
                                norm_val = 0
                                if "weight" in subgrad:
                                    norm_val += np.linalg.norm(np.array(subgrad["weight"]))
                                if "bias" in subgrad:
                                    norm_val += np.linalg.norm(np.array(subgrad["bias"]))
                                norms.append(norm_val)
                            except Exception as e:
                                norms.append(np.nan)
                        plt.figure(figsize=(10, 4), facecolor='white')
                        plt.plot(iterations, norms,
                                 label=f"Gradient norm of {key} block {block_idx}",
                                 marker='o')
                        plt.xlabel("Iteration")
                        plt.ylabel("L2 Norm")
                        plt.title(f"Gradient Norm Evolution: {key} block {block_idx}")
                        plt.legend()
                        plt.tight_layout()
                        plt.show()
