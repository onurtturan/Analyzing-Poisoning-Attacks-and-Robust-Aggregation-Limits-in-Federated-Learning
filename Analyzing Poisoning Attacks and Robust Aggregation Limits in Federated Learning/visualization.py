import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc

def plot_from_json(json_path="simulation_results.json", output_file="comparison_results.png"):
    """
    Loads simulation results from a JSON file and plots Accuracy, Recall, Precision, and AUC.
    Saves the plot as 'output_file'.
    """
    try:
        with open(json_path, 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"Error: {json_path} not found.")
        return

    plt.figure(figsize=(12, 10))

    metrics_to_plot = [
        ('accuracy', 'Global Accuracy', 'Accuracy'),
        ('recall', 'Global Recall (Class 1)', 'Recall'),
        ('precision', 'Global Precision (Class 1)', 'Precision'),
        ('auc', 'Global AUC-ROC', 'AUC')
    ]

    for i, (metric_key, title, ylabel) in enumerate(metrics_to_plot):
        plt.subplot(2, 2, i+1)
        for scenario_name, metrics in results.items():
            if metric_key in metrics:
                data = metrics[metric_key]
                if isinstance(data[0], list) or isinstance(data[0], tuple):
                    rounds = [x[0] for x in data]
                    values = [x[1] for x in data]
                else:
                    values = data
                    rounds = range(1, len(values) + 1)
                
                plt.plot(rounds, values, 'o-', label=scenario_name)
        
        plt.title(title)
        plt.xlabel('Round')
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Comparison plot saved to '{output_file}'.")

def plot_roc_curves(json_path="roc_data.json", output_file="roc_comparison.png"):
    """
    Plots ROC Curves for all scenarios.
    """
    try:
        with open(json_path, 'r') as f:
            roc_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {json_path} not found.")
        return
        
    plt.figure(figsize=(10, 8))
    
    # Extract scenario names from keys (assuming format "{name}_y_true")
    keys = roc_data.keys()
    scenario_names = set([k.replace("_y_true", "").replace("_y_prob", "") for k in keys if "_y_true" in k])
    
    for name in scenario_names:
        y_true_key = f"{name}_y_true"
        y_prob_key = f"{name}_y_prob"
        
        if y_true_key in roc_data and y_prob_key in roc_data:
            y_true = roc_data[y_true_key]
            y_prob = roc_data[y_prob_key]
            
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
            
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    plt.savefig(output_file)
    print(f"ROC comparison plot saved to '{output_file}'.")

def plot_tsne(X, y, title, output_file):
    """
    Plots t-SNE visualization of the data.
    """
    print(f"Generating t-SNE plot: {title}...")
    
    # Sampling if data is too large
    if len(X) > 2000:
        indices = np.random.choice(len(X), 2000, replace=False)
        X_subset = X[indices]
        y_subset = y[indices]
    else:
        X_subset = X
        y_subset = y

    # t-SNE Algorithm
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_embedded = tsne.fit_transform(X_subset)

    # Plotting
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=X_embedded[:, 0], 
        y=X_embedded[:, 1], 
        hue=y_subset, 
        palette="viridis", 
        s=60, 
        alpha=0.8
    )
    
    plt.title(title)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(title='Class')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.savefig(output_file)
    print(f"t-SNE plot saved to '{output_file}'.")

def plot_attack_matrix(json_path="attack_stats.json"):
    """
    Plots a confusion matrix style heatmap for label flipping for EACH scenario.
    """
    try:
        with open(json_path, 'r') as f:
            all_stats = json.load(f)
    except FileNotFoundError:
        print(f"Error: {json_path} not found.")
        return
        
    for scenario_name, stats in all_stats.items():
        flip_counts = stats.get("flip_counts", {})
        n_0_to_1 = flip_counts.get("0_to_1", 0)
        n_1_to_0 = flip_counts.get("1_to_0", 0)
        
        matrix_data = np.array([
            [0, n_0_to_1],
            [n_1_to_0, 0]
        ])
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(matrix_data, annot=True, fmt="d", cmap="Reds", cbar=False,
                    xticklabels=["0", "1"], yticklabels=["0", "1"])
        
        plt.title(f"Attack Impact: {scenario_name}")
        plt.xlabel("Poisoned Label")
        plt.ylabel("Original Label")
        
        # Sanitize filename
        safe_name = scenario_name.replace(" ", "_").replace("(", "").replace(")", "")
        filename = f"attack_matrix_{safe_name}.png"
        
        plt.savefig(filename)
        print(f"Attack matrix saved to '{filename}'.")
        plt.close()

def plot_poison_breakdown(json_path="attack_stats.json"):
    """
    Plots a pie chart of Clean vs Poisoned Data for EACH scenario.
    """
    try:
        with open(json_path, 'r') as f:
            all_stats = json.load(f)
    except FileNotFoundError:
        print(f"Error: {json_path} not found.")
        return
        
    for scenario_name, stats in all_stats.items():
        total = stats.get("total_samples_global", 1)
        poisoned = stats.get("total_poisoned_samples", 0)
        clean = total - poisoned
        
        flip_counts = stats.get("flip_counts", {})
        n_0_to_1 = flip_counts.get("0_to_1", 0)
        n_1_to_0 = flip_counts.get("1_to_0", 0)
        
        labels = ['Clean Data', 'Poisoned (0->1)', 'Poisoned (1->0)']
        sizes = [clean, n_0_to_1, n_1_to_0]
        colors = ['#2ca02c', '#ff7f0e', '#d62728'] # Green, Orange, Red
        
        # Filter out zero slices
        final_labels = []
        final_sizes = []
        final_colors = []
        
        for l, s, c in zip(labels, sizes, colors):
            if s > 0:
                final_labels.append(l)
                final_sizes.append(s)
                final_colors.append(c)
                
        if not final_sizes:
            continue
            
        plt.figure(figsize=(7, 7))
        plt.pie(final_sizes, labels=final_labels, colors=final_colors, autopct='%1.1f%%', startangle=140)
        plt.title(f"Data Integrity: {scenario_name}")
        
        # Sanitize filename
        safe_name = scenario_name.replace(" ", "_").replace("(", "").replace(")", "")
        filename = f"pie_chart_{safe_name}.png"
        
        plt.savefig(filename)
        print(f"Poison breakdown pie chart saved to '{filename}'.")
        plt.close()

def plot_global_confusion_matrix(json_path="confusion_matrices.json", output_file="global_confusion_matrix.png"):
    """
    Plots the Global Confusion Matrices for all scenarios.
    """
    try:
        with open(json_path, 'r') as f:
            cms = json.load(f)
    except FileNotFoundError:
        print(f"Error: {json_path} not found.")
        return
        
    # Determine grid size
    n_scenarios = len(cms)
    cols = 2
    rows = (n_scenarios + 1) // 2
    
    plt.figure(figsize=(12, 5 * rows))
    
    for i, (name, cm_list) in enumerate(cms.items()):
        cm = np.array(cm_list)
        
        plt.subplot(rows, cols, i+1)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Predicted 0", "Predicted 1"],
                    yticklabels=["Actual 0", "Actual 1"])
        plt.title(f"{name} Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        
    plt.suptitle("Global Model Confusion Matrix (Threshold=0.43)")
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Global confusion matrix plot saved to '{output_file}'.")
