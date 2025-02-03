import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from Plots import *
import Data_collection


def random_forest(isGridSearch):
    def gridSearch(model, X2_train, y2_train):
        # Define the hyperparameters to tune and their possible values
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2, 3]
        }

        # Create the GridSearchCV object with cross-validation (e.g., 3-fold cross-validation)
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring='roc_auc',
            cv=3
        )

        print("Training with the Grid Search")
        # Train with the Grid Search
        grid_search.fit(X2_train, y2_train)

        # Output the best parameters that were found
        print("Output the best hyperparams")
        best_params = grid_search.best_params_
        print(f"Best Hyperparameters: {best_params}")

        # Uncomment the following line if you want to generate graphs for hyperparameters
        # generateGraphsGridSearch()

        return grid_search

    def featureSelection(best_rf_classifier, X_train, y_train):
        # Random Forest has feature importance scores that we can use for feature selection
        rf_classifier = RandomForestClassifier(
            n_estimators=best_rf_classifier.n_estimators,
            max_depth=best_rf_classifier.max_depth,
            min_samples_split=best_rf_classifier.min_samples_split,
            min_samples_leaf=best_rf_classifier.min_samples_leaf
        )
        rf_classifier.fit(X_train, y_train)

        importance_scores = rf_classifier.feature_importances_
        feature_names = X_train.columns
        feature_importance = list(zip(feature_names, importance_scores))
        feature_importance.sort(key=lambda x: x[1], reverse=True)

        threshold = np.percentile(importance_scores, 60)

        top_features = []
        for feature, importance in feature_importance:
            if importance > threshold:
                top_features.append(feature)

        X_train_top = X_train[top_features]

        rf_classifier.fit(X_train_top, y_train)

        return rf_classifier, top_features

    # 1. Load and preprocess data
    data = pd.read_csv('../data/derivatives/DATA.csv')

    X, y = Data_collection.preprocessing(data, "goalFlag")
    X = X.drop(columns=['evt_idx', 'shotBy'], errors='ignore')  # Drop 'shotBy'

    # Split data 80/20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train initial Random Forest model
    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(X_train[['shot_distance', 'shot_angle']], y_train)

    y_probs = rf_classifier.predict_proba(X_test[['shot_distance', 'shot_angle']])

    # 2. Grid Search for hyperparameter tuning
    rf_classifier2 = RandomForestClassifier()
    if isGridSearch:
        grid_search = gridSearch(rf_classifier2, X_train, y_train)
        best_rf_classifier = grid_search.best_estimator_
        y_probs1 = best_rf_classifier.predict_proba(X_test)

        new_best_rf_classifier = RandomForestClassifier(
            n_estimators=best_rf_classifier.n_estimators,
            max_depth=best_rf_classifier.max_depth,
            min_samples_split=best_rf_classifier.min_samples_split,
            min_samples_leaf=best_rf_classifier.min_samples_leaf
        )
    else:
        new_best_rf_classifier = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_split=2, min_samples_leaf=1)

    # 3. Feature selection
    featureSelectionRf, top_features = featureSelection(new_best_rf_classifier, X_train, y_train)
    y_probs2 = featureSelectionRf.predict_proba(X_test[top_features])
    print(top_features)

    CLFS = [
        [[best_rf_classifier], X_test, 'Best Random Forest GridSearch'],
        [[featureSelectionRf], X_test[top_features], 'Random Forest Feature Selection'],
        [[rf_classifier], X_test[['shot_distance', 'shot_angle']], 'Random Forest']
    ]
    Ys = [
        ["Best Random Forest GridSearch", y_probs1[:, 1], "blue", True],
        ["Random Forest Feature Selection", y_probs2[:, 1], "orange", True],
        ["Random Forest", y_probs[:, 1], "green", True]
    ]
    all_plots(y_test, Ys, CLFS, y_test)

    return featureSelectionRf


if __name__ == "__main__":
    random_forest(True)