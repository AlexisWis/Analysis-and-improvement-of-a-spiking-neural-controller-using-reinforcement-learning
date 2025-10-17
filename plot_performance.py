import json
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

# Choose which parameter to plot on x-axis
x_param = "explore_period"  # could be "beta", "explore_period", or "gamma"

# ---- load JSON ----
with open(f"{x_param}_results.json", "r") as f:
    data = json.load(f)

# Example parameter lists
beta_vals = [0.0001, 0.001, 0.01, 0.03, 0.05, 0.08, 0.1] #[0.005, 0.01, 0.015, 0.02]
explore_period_vals = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
gamma_vals = [0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1]


# Fixed values for the other two parameters
fixed_values = {
    "beta": 0.01,              # if x-axis is beta, leave it None
    "explore_period": 100,     
    "gamma": 0.98              
}

# Map param names to their value lists
param_lists = {
    "beta": beta_vals,
    "explore_period": explore_period_vals,
    "gamma": gamma_vals
}

# ---- compute mean and std test lifetime per parameter value ----
x_values = param_lists[x_param]
y_means = []
y_stds = []

for x_val in x_values:
    # Determine the key for this combination
    key_dict = fixed_values.copy()
    key_dict[x_param] = x_val
    key = str((key_dict["beta"], key_dict["explore_period"], key_dict["gamma"]))

    # Extract test runs
    runs_test = data[key]["test"]

    # Pad runs (to handle different lengths) and convert to float
    max_len = max(len(run) for run in runs_test)
    runs_padded = [np.pad(np.array(run, dtype=float), 
                          (0, max_len - len(run)), 
                          constant_values=np.nan)
                   for run in runs_test]
    
    # Flatten all runs into one array to get overall mean and std
    all_values = np.concatenate(runs_padded)
    mean_lifetime = np.nanmean(all_values)
    std_lifetime = np.nanstd(all_values)

    y_means.append(mean_lifetime)
    y_stds.append(std_lifetime)

print(y_means)
# ---- plot with error bars ----
plt.figure(figsize=(6,4))
plt.errorbar(x_values, y_means, yerr=y_stds, fmt='o-', capsize=5, linewidth=2)
plt.xlabel(x_param)
plt.ylabel("Mean lifetime of test runs")
plt.title(f"Mean Test Lifetime vs {x_param} (±1 std)")
plt.grid(True)
plt.show()
