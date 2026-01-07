import flwr as fl
import numpy as np
import json
from sklearn.metrics import confusion_matrix
from client import CreditCardClient
from dataset import prepare_partitions, load_partition
import centralized_training
import utils
import visualization

# Set random seed for reproducibility
np.random.seed(42)

# Ensure partitions are ready
prepare_partitions()

# Configuration
AGGREGATION_METHOD = "median" # Options: "mean", "median"

def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    recalls = [num_examples * m["recall"] for num_examples, m in metrics]
    precisions = [num_examples * m["precision"] for num_examples, m in metrics]
    aucs = [num_examples * m["auc"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    return {
        "accuracy": sum(accuracies) / sum(examples),
        "recall": sum(recalls) / sum(examples),
        "precision": sum(precisions) / sum(examples),
        "auc": sum(aucs) / sum(examples)
    }

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.final_parameters = None

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_parameters is not None:
            self.final_parameters = aggregated_parameters
            
        return aggregated_parameters, aggregated_metrics

class FedMedian(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.final_parameters = None

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        # Get weights from results
        weights_results = [
            (fl.common.parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        
        # Extract just the weights (ignoring num_examples for median)
        weights = [w for w, _ in weights_results]
        
        # Calculate median per layer
        # zip(*weights) gives us an iterator over layers ((client1_layer0, client2_layer0, ...), ...)
        new_weights = [
            np.median(layer_updates, axis=0) 
            for layer_updates in zip(*weights)
        ]

        # Convert back to parameters
        parameters_aggregated = fl.common.ndarrays_to_parameters(new_weights)
        
        # Aggregate metrics (delegated to helper or empty)
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
            
        self.final_parameters = parameters_aggregated
        return parameters_aggregated, metrics_aggregated

def run_simulation(scenario_name, attack_type, num_malicious_clients):
    print(f"\n--- Starting Simulation: {scenario_name} ---")
    print(f"Attack Type: {attack_type}, Malicious Clients: {num_malicious_clients}")
    
    def client_fn(cid):
        # Determine if this client is malicious
        # We designate the first 'num_malicious_clients' as malicious
        if int(cid) < num_malicious_clients:
            client_attack_type = attack_type
        else:
            client_attack_type = "none"
            
        return CreditCardClient(cid, attack_type=client_attack_type)

    # Strategy
    strategy_args = dict(
        fraction_fit=1.0, # All clients participate
        fraction_evaluate=1.0,
        min_fit_clients=10,
        min_evaluate_clients=10,
        min_available_clients=10,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    if AGGREGATION_METHOD == "median":
        strategy = FedMedian(**strategy_args)
    else:
        strategy = SaveModelStrategy(**strategy_args)

    # Start Simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=10,
        config=fl.server.ServerConfig(num_rounds=50),
        strategy=strategy,
        client_resources=None, # Use default resources
    )
    
    # Return history AND the final parameters from the strategy
    return history, strategy.final_parameters

def calculate_attack_stats(scenario_name, attack_type, num_malicious_clients):
    print(f"Calculating stats for: {scenario_name}")
    
    stats = {
        "scenario": scenario_name,
        "total_samples_global": 0,
        "total_poisoned_samples": 0,
        "global_poison_ratio": 0.0,
        "flip_counts": {"0_to_1": 0, "1_to_0": 0},
        "affected_clients": 0
    }
    
    # Iterate through all clients
    for cid in range(10): # Assuming 10 clients
        # Load data
        X_train, y_train, _, _ = load_partition(cid)
        
        stats["total_samples_global"] += len(y_train)
        
        if cid < num_malicious_clients:
            stats["affected_clients"] += 1
            
            if attack_type == "label_flip":
                # 0->1 and 1->0
                n_zeros = (y_train == 0).sum()
                n_ones = (y_train == 1).sum()
                stats["flip_counts"]["0_to_1"] += int(n_zeros)
                stats["flip_counts"]["1_to_0"] += int(n_ones)
                stats["total_poisoned_samples"] += len(y_train)
                
            elif attack_type == "targeted_flip":
                # Only 1->0. 0s are untouched.
                n_ones = (y_train == 1).sum()
                stats["flip_counts"]["1_to_0"] += int(n_ones)
                stats["total_poisoned_samples"] += int(n_ones) # Only flipped samples count as poisoned? 
                # Or is the whole dataset considered "poisoned" because it's from a malicious client?
                # Usually "poisoned samples" refers to the ones actually modified.
                # But for consistency with previous logic, let's count the modified ones.
                
            elif attack_type == "none":
                pass
            
    if stats["total_samples_global"] > 0:
        stats["global_poison_ratio"] = (stats["total_poisoned_samples"] / stats["total_samples_global"]) * 100
        
    return stats

if __name__ == "__main__":
    SCENARIOS = [
        {"name": "Baseline", "attack": "none", "n_malicious": 0},
        {"name": "Untargeted (Label Flip)", "attack": "label_flip", "n_malicious": 2},
        #{"name": "Targeted (5 Clients)", "attack": "targeted_flip", "n_malicious": 5},
        #{"name": "Targeted (7 Clients)", "attack": "targeted_flip", "n_malicious": 7},
        #{"name": "Targeted (9 Clients)", "attack": "targeted_flip", "n_malicious": 9}
    ]
    
    all_results = {}
    all_attack_stats = {}
    all_confusion_matrices = {}
    all_roc_data = {}
    
    # Load Centralized Test Data for Global Evaluation
    _, X_test_global, _, y_test_global = centralized_training.load_data()
    
    for scenario in SCENARIOS:
        name = scenario["name"]
        attack = scenario["attack"]
        n_mal = scenario["n_malicious"]
        
        # 1. Run Simulation
        hist, params = run_simulation(name, attack, n_mal)
        
        # 2. Extract Metrics
        metrics = {}
        for key, val_list in hist.metrics_distributed.items():
            metrics[key] = [[int(r), float(v)] for r, v in val_list]
        all_results[name] = metrics
        
        # 3. Calculate Attack Stats
        stats = calculate_attack_stats(name, attack, n_mal)
        all_attack_stats[name] = stats
        
        # 4. Global Confusion Matrix & ROC Data
        if params is not None:
            weights = fl.common.parameters_to_ndarrays(params)
            model = centralized_training.get_model_pipeline()
            utils.set_initial_params(model, X_test_global.iloc[:10], y_test_global.iloc[:10])
            utils.set_model_parameters(model, weights)
            
            y_prob = model.predict_proba(X_test_global)[:, 1]
            
            # Dynamic Thresholding for Global Model
            threshold = utils.find_optimal_threshold(y_test_global, y_prob)
            print(f"Global Model (Scenario: {name}) - Optimal Threshold: {threshold:.4f}")
            
            y_pred = (y_prob >= threshold).astype(int)
            
            # CM
            cm = confusion_matrix(y_test_global, y_pred)
            all_confusion_matrices[name] = cm.tolist()
            
            # ROC
            all_roc_data[f"{name}_y_true"] = y_test_global.tolist()
            all_roc_data[f"{name}_y_prob"] = y_prob.tolist()
            
    # Save All Data
    suffix = f"_{AGGREGATION_METHOD}"
    
    with open(f"simulation_results{suffix}.json", "w") as f:
        json.dump(all_results, f, indent=4)
        
    with open(f"attack_stats{suffix}.json", "w") as f:
        json.dump(all_attack_stats, f, indent=4)
        
    with open(f"confusion_matrices{suffix}.json", "w") as f:
        json.dump(all_confusion_matrices, f, indent=4)
        
    with open(f"roc_data{suffix}.json", "w") as f:
        json.dump(all_roc_data, f, indent=4)
        
    print(f"\nAll simulations complete ({AGGREGATION_METHOD}). Data saved.")
    
    # Visualization
    print("Generating Visualizations...")
    visualization.plot_from_json(f"simulation_results{suffix}.json", f"comparison_results{suffix}.png")
    visualization.plot_roc_curves(f"roc_data{suffix}.json", f"roc_comparison{suffix}.png")
    
    # Generate dynamic plots for each scenario
    visualization.plot_attack_matrix(f"attack_stats{suffix}.json")
    visualization.plot_poison_breakdown(f"attack_stats{suffix}.json")
    visualization.plot_global_confusion_matrix(f"confusion_matrices{suffix}.json", f"global_confusion_matrix{suffix}.png")
    
    # t-SNE Generation for All Scenarios
    print("\nGenerating t-SNE plots for all scenarios...")
    for scenario in SCENARIOS:
        name = scenario["name"]
        attack = scenario["attack"]
        n_mal = scenario["n_malicious"]
        
        # Determine attack type for Client 0
        # If the scenario has malicious clients, we assume Client 0 is one of them (since we start from 0)
        # If n_malicious == 0, then the attack is "none" anyway.
        current_attack_type = attack if n_mal > 0 else "none"
        
        # Instantiate Client 0 to get its data (poisoned or clean)
        # We use Client 0 as the representative client.
        client = CreditCardClient(cid=0, attack_type=current_attack_type)
        
        # Prepare title and filename
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        title = f"Data Distribution: {name} (Client 0)"
        filename = f"tsne_{safe_name}{suffix}.png"
        
        # Plot
        visualization.plot_tsne(
            client.X_train.values, 
            client.y_train_poisoned, 
            title=title, 
            output_file=filename
        )
