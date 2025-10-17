import json
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

# ---- load JSON ----
with open("results.json", "r") as f:
    data = json.load(f)

beta_vals = [0.01]
explore_period_vals = [100]
gamma_vals = [0.98]

# ---- helper function for plotting ----
def plot_runs(ax, runs, title):
    for run in runs:
        ax.plot(run, linewidth=0.8, alpha=0.6, color="gray")  # individual runs

    # Pad runs to same length (convert to float for NaN padding)
    max_len = max(len(run) for run in runs)
    runs_padded = [np.pad(np.array(run, dtype=float),
                          (0, max_len - len(run)),
                          constant_values=np.nan) 
                   for run in runs]

    runs_padded = np.vstack(runs_padded)
    mean_curve = np.nanmean(runs_padded, axis=0)
    std_curve = np.nanstd(runs_padded, axis=0)

    # Print average and std for reference
    print(f"{title}: mean={np.nanmean(mean_curve):.3f}, std={np.nanmean(std_curve):.3f}")

    # Plot mean curve
    ax.plot(mean_curve, linewidth=2.5, color="blue", label="Average")
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Lifetime")
    ax.legend()

# ---- plotting loop ----
for beta, explore_period, gamma in product(beta_vals, explore_period_vals, gamma_vals):
    key = str((beta, explore_period, gamma))
    
    runs_train = data[key]["train"]
    runs_test = data[key]["test"]

    print(f"\n=== Parameters: beta={beta}, explore_period={explore_period}, gamma={gamma} ===")
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.suptitle(f"Parameters: beta={beta}, explore_period={explore_period}, gamma={gamma}", fontsize=14)

    plot_runs(axs[0], runs_train, "Train")
    plot_runs(axs[1], runs_test, "Test")

    plt.tight_layout()
    plt.show()
