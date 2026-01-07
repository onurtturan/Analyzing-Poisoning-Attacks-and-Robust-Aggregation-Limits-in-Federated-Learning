import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from centralized_training import load_data

# Global variable to store partitions
PARTITIONS = None

def partition_data(X, y, n_clients=10, beta=4.0):
    """
    Partitions the data into n_clients using Dirichlet distribution for Non-IIDness.
    Ensures each client has at least 100 samples.
    """
    min_size = 0
    min_require_size = 1000
    y = y.values # Convert to numpy array for indexing
    N = len(y)
    
    # We partition indices, then split X and y
    # Actually, we should partition the original dataset before splitting into train/test?
    # Or partition X_train and X_test separately?
    # Usually in FL, each client has their own local dataset (Train + Test).
    # But here we are simulating from a central dataset.
    # To be consistent, let's partition the indices of the WHOLE dataset (X, y) 
    # and then each client will do their own train/test split?
    # OR, as per `centralized_training.py`, `load_data` returns X_train, X_test...
    # The user said: "load_data fonksiyonunu kullan".
    # And "load_partition... X_train, y_train, X_test, y_test döndürmeli".
    # So we should partition the TRAIN set and TEST set separately using the same distribution logic?
    # Or partition the whole and then split?
    # If we partition separately, we might get different distributions.
    # Best approach for simulation: Partition the indices of the combined data, then split each partition.
    # BUT `load_data` already splits.
    # Let's take X_train, y_train from load_data and partition them. 
    # And X_test, y_test and partition them? 
    # Or just partition X_train/y_train for clients and keep a global test set?
    # User said: "load_partition... döndürmeli... (X_train, y_train, X_test, y_test)".
    # This implies each client has a local test set.
    # So I will partition X_train and X_test SEPARATELY using the same beta, 
    # OR better: Concatenate, partition, then split.
    # Let's Concatenate first to ensure consistent distribution per client.
    
    # Wait, `load_data` returns split data.
    # I will recombine them to partition, then split again per client? 
    # Or just partition the training data and split that into train/test for each client?
    # The latter is more realistic (client has data, splits it).
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from centralized_training import load_data

# Global variable to store partitions
PARTITIONS = None

def partition_data(X, y, n_clients=10, beta=2.0):
    """
    Partitions the data into n_clients using Dirichlet distribution for Non-IIDness.
    Strict Safety Checks:
    - Min Total Samples: 1000
    - Min Class 1 (Default): 50
    - Min Class 0 (Non-Default): 200
    """
    y = y.values # Convert to numpy array for indexing
    N = len(y)
    
    while True:
        idx_batch = [[] for _ in range(n_clients)]
        for k in range(2): # For each class (0 and 1)
            idx_k = np.where(y == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, n_clients))
            
            # Balance check: avoid very small proportions if possible
            proportions = np.array([p * (len(idx_j) < N / n_clients) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            
            idx_batch_split = np.split(idx_k, proportions)
            for i in range(n_clients):
                idx_batch[i] = np.concatenate((idx_batch[i], idx_batch_split[i]), axis=0)
        
        # Check constraints
        min_total = min([len(idx) for idx in idx_batch])
        
        # Check class counts per client
        min_class_1 = float('inf')
        min_class_0 = float('inf')
        
        valid_partition = True
        for idx in idx_batch:
            y_client = y[idx.astype(int)]
            c1_count = np.sum(y_client == 1)
            c0_count = np.sum(y_client == 0)

    
            
            if c1_count < 100 or c0_count < 500:
                valid_partition = False
                break
            
            if c1_count < min_class_1: min_class_1 = c1_count
            if c0_count < min_class_0: min_class_0 = c0_count
            
        if min_total >= 1250 and valid_partition:
            print(f"Partition Accepted! Min Total: {min_total}, Min Class 1: {min_class_1}, Min Class 0: {min_class_0}")
            break
        else:
            # print(f"Retry... Min Total: {min_total}, Min C1: {min_class_1}, Min C0: {min_class_0}")
            pass
            
    # Convert indices to int
    client_indices = [idx.astype(int) for idx in idx_batch]
    
    return client_indices

def prepare_partitions():
    """
    Orchestrates the data loading, partitioning, and storage.
    """
    global PARTITIONS
    
    # 1. Load Data (Centralized Split)
    X_train, X_test, y_train, y_test = load_data()
    
    # 2. Recombine to partition properly for each client
    X_full = pd.concat([X_train, X_test])
    y_full = pd.concat([y_train, y_test])
    
    # Reset index to ensure iloc works with partition indices
    X_full = X_full.reset_index(drop=True)
    y_full = y_full.reset_index(drop=True)
    
    # 3. Partition
    client_indices = partition_data(X_full, y_full, n_clients=10, beta=3.0)
    
    PARTITIONS = []
    for indices in client_indices:
        # Extract client data
        X_client = X_full.iloc[indices]
        y_client = y_full.iloc[indices]
        
        # Split into local Train/Test (80/20)
        # We use stratify to maintain the local class ratio in train/test
        # But if a client has very few of one class, stratify might fail.
        # Try/except or check count?
        # With 100 samples min, it should be okay, but let's be safe.
        try:
            X_c_train, X_c_test, y_c_train, y_c_test = \
                from_sklearn_split(X_client, y_client)
        except ValueError:
            # Fallback without stratify if classes are too few
             X_c_train, X_c_test, y_c_train, y_c_test = \
                from_sklearn_split(X_client, y_client, stratify=None)
                
        PARTITIONS.append((X_c_train, y_c_train, X_c_test, y_c_test))

def from_sklearn_split(X, y, stratify=True):
    from sklearn.model_selection import train_test_split
    strat = y if stratify else None
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=strat)

def load_partition(client_id: int):
    """
    Returns (X_train, y_train, X_test, y_test) for the given client_id.
    """
    global PARTITIONS
    if PARTITIONS is None:
        prepare_partitions()
        
    if client_id < 0 or client_id >= len(PARTITIONS):
        raise ValueError("Invalid client_id")
        
    return PARTITIONS[client_id]

def plot_class_distribution(partitions):
    """
    Plots the class distribution for each client.
    """
    n_clients = len(partitions)
    client_ids = range(n_clients)
    
    zeros = []
    ones = []
    ratios = []
    totals = []
    
    for p in partitions:
        # Combine train and test to see total distribution
        y_total = pd.concat([p[1], p[3]])
        counts = y_total.value_counts()
        n_0 = counts.get(0, 0)
        n_1 = counts.get(1, 0)
        total = n_0 + n_1
        
        zeros.append(n_0)
        ones.append(n_1)
        ratios.append(n_1 / total * 100)
        totals.append(total)
        
    plt.figure(figsize=(14, 7))
    
    # Stacked Bar Chart
    p1 = plt.bar(client_ids, zeros, label='Class 0 (Non-Default)', color='#1f77b4')
    p2 = plt.bar(client_ids, ones, bottom=zeros, label='Class 1 (Default)', color='#ff7f0e')
    
    plt.xlabel('Client ID')
    plt.ylabel('Number of Samples')
    plt.title('Data Distribution per Client (Non-IID)')
    plt.xticks(client_ids)
    plt.legend()
    
    # Annotations
    for i in range(n_clients):
        text_label = f'1: %{ratios[i]:.1f}\n(N={totals[i]})'
        plt.text(i, zeros[i] + ones[i] + 50, text_label, 
                 ha='center', va='bottom', fontsize=9, rotation=0)
                 
    plt.tight_layout()
    plt.savefig('data_distribution.png')
    print("Distribution plot saved to 'data_distribution.png'")

if __name__ == "__main__":
    # Initialize partitions
    prepare_partitions()
    
    # Plot
    plot_class_distribution(PARTITIONS)
    
    # Calculate Stats
    min_samples = float('inf')
    min_sample_client = -1
    
    min_rate = float('inf')
    min_rate_client = -1
    
    print("\n--- Client Data Statistics ---")
    for i, p in enumerate(PARTITIONS):
        y_total = pd.concat([p[1], p[3]])
        n_1 = y_total.sum()
        total = len(y_total)
        ratio = n_1 / total * 100
        
        print(f"Client {i}: Total={total}, Default Rate={ratio:.2f}% (Defaults={n_1})")
        
        if total < min_samples:
            min_samples = total
            min_sample_client = i
            
        if ratio < min_rate:
            min_rate = ratio
            min_rate_client = i
            
    print("\n--- Summary ---")
    print(f"Client with Fewest Samples: Client {min_sample_client} ({min_samples} samples)")
    print(f"Client with Lowest Default Rate: Client {min_rate_client} ({min_rate:.2f}%)")
    
    # Test load_partition
    print("\nTesting load_partition(2)...")
    X_tr, y_tr, X_te, y_te = load_partition(2)
    print(f"Client 2 Shapes: Train={X_tr.shape}, Test={X_te.shape}")
