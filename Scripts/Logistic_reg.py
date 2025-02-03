import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from Plots import *

def train_and_evaluate(X_train, X_val, y_train, y_val, feature_name):
    """Train a Logistic Regression model and evaluate it."""
    if feature_name != 'both':
        X_train = X_train.values.reshape(-1, 1)
        X_val = X_val.values.reshape(-1, 1)
    
    model = LogisticRegression().fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    
    print(f'Feature: {feature_name}')
    print(f'Accuracy = {accuracy_score(y_val, y_pred)}')
    print(f'Confusion Matrix:\n{confusion_matrix(y_val, y_pred)}')
    
    return model, y_prob


def run_regression(X, y):
    """Main function to run Logistic Regression models on different features."""
    # Load Data


    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    # Train models
    models = {}
    probabilities = {}
    features = ['shot_distance', 'shot_angle', 'both']
    
    for feature in features:
        if feature == 'both':
            X_train_feature = X_train
            X_val_feature = X_val
        else:
            X_train_feature = X_train[feature]
            X_val_feature = X_val[feature]
        
        model, y_prob = train_and_evaluate(X_train_feature, X_val_feature, y_train, y_val, feature)
        models[feature] = model
        probabilities[feature] = y_prob
    
    # Generate plots
    Ys = [
        ["Distance", probabilities['shot_distance'], "blue", False],
        ["Angle", probabilities['shot_angle'], "orange", False],
        ["Distance & Angle", probabilities['both'], "green", False],
        ["Random Baseline", np.random.uniform(0, 1, len(y_val)), "red", True]
    ]
    all_plots(y_val, Ys, [[[models['shot_distance']], X_val[['shot_distance']], 'Distance'],
                         [[models['shot_angle']], X_val[['shot_angle']], 'Angle'],
                         [[models['both']], X_val, 'Distance & Angle']], y_val)
    
    return models

if __name__ == "__main__":
    run_regression()