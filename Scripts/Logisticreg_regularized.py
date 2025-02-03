import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from Data_collection import preprocessing
from Plots import all_plots  # Ensure this function is defined in Plots.py


def runBoosted_Logistic_reg(isRandomizedSearchCV):
    # Load and preprocess data
    df = pd.read_csv('../data/derivatives/DATA.csv')
    X, y = preprocessing(df, 'goalFlag')
    X = X.drop(columns=['evt_idx', 'shotBy'], errors='ignore')  # Drop 'shotBy'

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Elasticnet regularization (combines L1 and L2 penalties)
    # Only 'saga' solver supports this regularization
    clf = LogisticRegression(penalty='elasticnet', max_iter=1000, solver='saga', l1_ratio=0.5)
    clf.fit(X_train, y_train)

    # Use XGBoost as the boosting classifier
    xgb_model = XGBClassifier(random_state=42)

    # Hyperparameter grid for RandomizedSearchCV
    param_grid = {
        'n_estimators': [50, 100, 150, 200, 250],
        'learning_rate': [0.01, 0.1, 0.2, 0.3, 0.5],
        'max_depth': [3, 4, 5, 6, 7],
    }

    # RandomizedSearchCV or default XGBoost
    if isRandomizedSearchCV:
        random_sch = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_grid,
            n_iter=10,  # Number of parameter settings to sample
            scoring='roc_auc',  # Scoring metric
            cv=3,  # Cross-validation folds
            n_jobs=-1,  # Use all available CPU cores
            random_state=42,  # For reproducibility
            refit=True  # Refit the best model on the entire dataset
        )
        # Fit RandomizedSearchCV
        print("Fitting RandomizedSearchCV...")
        random_sch.fit(X_train, y_train)

        # Get the best hyperparameters and model
        best_params = random_sch.best_params_
        print("Best Hyperparameters:", best_params)
        best_model = random_sch.best_estimator_
    else:
        # Use default hyperparameters
        best_model = XGBClassifier(random_state=42, n_estimators=50, max_depth=3, learning_rate=0.3)
        best_model.fit(X_train, y_train)

    # Make predictions with the best model
    y_prob = best_model.predict_proba(X_test)

    # Prepare data for plotting
    Y = [["Boosted_Logistic_reg", y_prob[:, 1], "blue", True]]
    CLF = [[[best_model], X_test, 'Boosted_Logistic_reg']]

    # Generate plots
    all_plots(y_test, Y, CLF, y_test)

    return best_model


if __name__ == "__main__":
    runBoosted_Logistic_reg(False)