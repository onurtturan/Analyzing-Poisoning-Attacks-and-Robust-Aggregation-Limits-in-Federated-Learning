from sklearn.metrics import roc_curve
from numpy import sqrt, argmax
import numpy as np
from sklearn.linear_model import LogisticRegression
import warnings

def find_optimal_threshold(y_true, y_prob):
    """
    Calculates the optimal threshold using the Geometric Mean (G-Mean) 
    of Sensitivity (TPR) and Specificity (1 - FPR).
    
    Returns:
        float: Optimal threshold value.
    """
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        # Calculate the Geometric Mean for each threshold
        gmeans = sqrt(tpr * (1 - fpr))
        
        # Find the index of the largest G-Mean
        ix = argmax(gmeans)
        
        return thresholds[ix]
    except Exception as e:
        print(f"Error calculating threshold: {e}")
        return 0.5

def get_model_parameters(model):
    """
    Returns the parameters of a sklearn LogisticRegression model.
    """
    clf = model.named_steps['model']
    
    if clf.fit_intercept:
        params = [
            clf.coef_,
            clf.intercept_,
        ]
    else:
        params = [
            clf.coef_,
        ]
    return params

def set_model_parameters(model, parameters):
    """
    Sets the parameters of a sklearn LogisticRegression model.
    """
    clf = model.named_steps['model']
    
    # Set coefficients
    clf.coef_ = parameters[0]
    
    # Set intercept if it exists
    if clf.fit_intercept:
        clf.intercept_ = parameters[1]
        
    return model

def set_initial_params(model, X_sample, y_sample):
    """
    Sets initial parameters for the model to ensure shapes are correct.
    Fits the model on a tiny sample just to initialize coef_ and intercept_.
    """
    # Enable warm_start to preserve weights in future training rounds
    model.named_steps['model'].warm_start = True
    
    # Save original max_iter
    original_max_iter = model.named_steps['model'].max_iter
    
    # Set max_iter to 1 for quick initialization
    model.named_steps['model'].max_iter = 1
    
    # Fit on the provided sample to initialize shapes
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_sample, y_sample)
    
    # Restore original max_iter
    model.named_steps['model'].max_iter = original_max_iter
    
    return model