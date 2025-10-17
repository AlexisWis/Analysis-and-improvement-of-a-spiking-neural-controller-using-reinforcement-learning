from simulation import run_simulation
from agent import build_code
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from collections import defaultdict

def main():
    input_layer_module_name, \
    input_layer_neuron_model_name, \
    output_layer_module_name, \
    output_layer_neuron_model_name, \
    output_layer_synapse_model_name = build_code()
    
    # Parameters
    beta_vals = [0.01]#[0.0001, 0.001, 0.01, 0.03, 0.05, 0.08, 0.1]
    explore_period_vals = [100] #[0, 50, 100, 150]
    gamma_vals = [0.98] #[0.96, 0.97, 0.98, 0.99]
    trials = 10

    param_combos = [(b, ep, g, t, 
                    input_layer_module_name, \
                    input_layer_neuron_model_name, \
                    output_layer_module_name, \
                    output_layer_neuron_model_name, \
                    output_layer_synapse_model_name) for b, ep, g in itertools.product(beta_vals, explore_period_vals, gamma_vals) for t in range(trials)]

    worker_results = []

    # Run simulations in parallel
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(run_simulation, *params): params for params in param_combos}
        for future in as_completed(futures):
            worker_results.append(future.result())  # each result is (train_dict, test_dict)

    print("Total runs:", len(worker_results))

    # Aggregate results per parameter combination
    results = defaultdict(lambda: {"train": [], "test": []})

    for train_dict, test_dict, state_log in worker_results:
        # Add training results
        for combo, value in train_dict.items():
            results[combo]["train"].append(value)
        
        # Add testing results
        for combo, value in test_dict.items():
            results[combo]["test"].append(value)
        
        import matplotlib.pyplot as plt
        import numpy as np
        from collections import Counter

        # Example list of tuples

        # Count occurrences
        counts = Counter(state_log)

        # Extract unique x and y values
        xs = sorted(set(x for x, y in counts.keys()))
        ys = sorted(set(y for x, y in counts.keys()))

        # Create a 2D array for heatmap
        heatmap = np.zeros((len(ys), len(xs)), dtype=int)

        for (x, y), c in counts.items():
            i = ys.index(y)
            j = xs.index(x)
            heatmap[i, j] = c

        # Plot
        plt.imshow(heatmap, cmap="viridis", origin="lower")
        plt.colorbar(label="visit count")

        # Label axes with the actual x, y values
        plt.xticks(range(len(xs)), xs)
        plt.yticks(range(len(ys)), ys)

        plt.xlabel("angle")
        plt.ylabel("angular velocity")
        plt.title("Heatmap of visited states")
        plt.show()

    # Convert to normal dict for JSON serialization
    results = dict(results)
    print(results)
    # Save to JSON
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
