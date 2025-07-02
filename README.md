# Predicting NHL goals using classification models

## Introduction

In this project, I set out to analyze and predict outcomes in the National Hockey League (NHL) using data science techniques. The goal was to leverage historical data, advanced analytics, and machine learning to gain insights into team performance and forecast future results.

## Project Motivation

The NHL is a fast-paced, data-rich sport where small advantages can make a big difference. By applying data science, we can uncover hidden patterns, improve predictions, and potentially inform coaching or betting strategies.

## Data Collection

The first step was gathering data. I sourced historical NHL game data, including:
- Game results (win/loss, scores)
- Team statistics (shots, goals, penalties, etc.)
- Player statistics (goals, assists, time on ice, etc.)

## Data Cleaning and Preprocessing

Raw sports data is often messy. I performed several preprocessing steps:
- Handling missing values
- Normalizing statistics
- Feature engineering (e.g., calculating rolling averages, home/away splits)
- Encoding categorical variables

## Exploratory Data Analysis (EDA)

Before modeling, I explored the data to find trends and correlations:
- Visualized team and player performance over time
- Identified key features influencing game outcomes
- Checked for data imbalances

## Modeling Approach

I experimented with several machine learning models, including:
- Logistic Regression
- Random Forests
- Gradient Boosting Machines

The target variable was typically whether a team would win a given game.

## Model Evaluation

Models were evaluated using metrics such as accuracy, precision, recall, and ROC-AUC. I used cross-validation to ensure robustness and avoid overfitting.

## Key Results

- The best-performing model achieved an accuracy of X% on the test set.
- Important features included recent team form, home/away status, and special teams performance.
- The model was able to predict upsets and close games with reasonable accuracy.

## Challenges

- Data quality: Incomplete or inconsistent records required careful cleaning.
- Feature selection: Sports outcomes are influenced by many subtle factors.
- Overfitting: Avoiding models that were too closely tailored to past seasons.

## Conclusion

This project demonstrated the power of data science in sports analytics. While no model can predict every outcome, the insights gained can inform strategies and deepen our understanding of the game.

## Next Steps

- Incorporate real-time data for live predictions
- Explore player-level modeling for fantasy sports
- Share findings with the hockey analytics community

## Code and Notebooks

All code and analysis are available in this [GitHub repository](https://github.com/yourusername/NHL-Project). The main pipeline can be found in the `Notebooks/Full_pipeline.ipynb` notebook. 
