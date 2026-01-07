import warnings
import flwr as fl
import numpy as np
from sklearn.metrics import log_loss, accuracy_score, recall_score, precision_score, roc_auc_score

import centralized_training
import dataset
import utils

# Ignore warnings
warnings.filterwarnings("ignore")

class CreditCardClient(fl.client.NumPyClient):
    def __init__(self, cid, attack_type="none"):
        self.cid = cid
        self.attack_type = attack_type
        
        # Load data for this client
        self.X_train, self.y_train, self.X_test, self.y_test = dataset.load_partition(int(cid))
        
        # Initialize Poisoned Labels
        # Convert to numpy array to avoid pandas index mismatch errors during assignment
        self.y_train_poisoned = self.y_train.copy().values
        
        if self.attack_type == "label_flip":
            # Untargeted: Flip all labels (0->1, 1->0)
            self.y_train_poisoned = 1 - self.y_train_poisoned
            
        elif self.attack_type == "targeted_flip":
            # Targeted: Flip 90% of 1s to 0s (Fraud -> Legitimate)
            # Ensure at least one '1' remains to prevent Single Class Crash
            idx_ones = np.where(self.y_train_poisoned == 1)[0]
            total_ones = len(idx_ones)
            
            if total_ones > 1:
                # Calculate number to flip (90%)
                n_flip = int(total_ones * 0.9)
                
                # Safety check: Ensure we leave at least one '1'
                if n_flip >= total_ones:
                    n_flip = total_ones - 1
                
                # Randomly select indices to flip
                flip_indices = np.random.choice(idx_ones, size=n_flip, replace=False)
                self.y_train_poisoned[flip_indices] = 0
                
                # print(f"Client {cid}: Targeted Attack - Flipped {n_flip}/{total_ones} fraud cases to legitimate.")
            
        # Create model pipeline
        self.model = centralized_training.get_model_pipeline()
        
        # Initialize model parameters (Warm Start) using utils
        # Use poisoned labels for initialization to be consistent
        utils.set_initial_params(self.model, self.X_train, self.y_train_poisoned)

    def get_parameters(self, config):
        return utils.get_model_parameters(self.model)

    def fit(self, parameters, config):
        # Update local model parameters with global parameters
        utils.set_model_parameters(self.model, parameters)
        
        # Train the model on local data (using pre-calculated poisoned labels)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.X_train, self.y_train_poisoned)
        
        # Return updated parameters
        return utils.get_model_parameters(self.model), len(self.X_train), {}

    def evaluate(self, parameters, config):
        # Update local model with global parameters
        utils.set_model_parameters(self.model, parameters)
        
        # Predict probabilities on Test Set
        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        
        # Calculate Loss
        loss = log_loss(self.y_test, y_prob)
        
        # Dynamic Thresholding:
        # Calculate optimal threshold based on Training Data (what the client knows)
        # We predict on X_train to find the best separation the model has learned so far.
        y_prob_train = self.model.predict_proba(self.X_train)[:, 1]
        threshold = utils.find_optimal_threshold(self.y_train_poisoned, y_prob_train)
        
        # Apply Dynamic Threshold
        y_pred = (y_prob >= threshold).astype(int)
        
        # Calculate Metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        
        try:
            auc = roc_auc_score(self.y_test, y_prob)
        except ValueError:
            auc = 0.5
        
        return loss, len(self.X_test), {"accuracy": accuracy, "recall": recall, "precision": precision, "auc": auc}

if __name__ == "__main__":
    # Test the client
    print("Initializing Client 0 (Honest)...")
    client = CreditCardClient(0, attack_type="none")
    params = client.get_parameters(config={})
    client.fit(params, config={})
    
    print("Initializing Client 1 (Malicious)...")
    client_mal = CreditCardClient(1, attack_type="label_flip")
    params_mal = client_mal.get_parameters(config={})
    client_mal.fit(params_mal, config={})
    print("Malicious client fit complete.")
