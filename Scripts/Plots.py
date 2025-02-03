import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay , roc_curve, auc
from sklearn.calibration import CalibrationDisplay




def ROC_plot(y, Y, ax):
    curves = []
    AUC = {}
    for model, pred, color, chance_lev in Y:
        curves.append(model) 
        RocCurveDisplay.from_predictions(
            y,
            pred,
            name=model,
            color=color,
            plot_chance_level=chance_lev,
            ax=ax,
        )
        fpr, tpr, _ = roc_curve(np.array(y), pred, pos_label=1)
        AUC[model] = auc(fpr, tpr) 
    print(f'Métrique AUC : {AUC}')
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs")
    ax.set_title("Courbes ROC but vs non-but ")
    ax.legend()
    


def Centiles_plot(y, Y, ax):
    curves = []
    for model, pred, _, __ in Y:
        centiles = np.percentile(pred, np.arange(0, 101, 5))
        taux_buts = []
        curves.append(model)
        for i in range(20):
            lower_bound = centiles[i]
            upper_bound = centiles[i + 1]
            indices = np.where((pred >= lower_bound) & (pred <= upper_bound))
            goal_rate = np.sum(y[indices]) / len(y[indices]) * 100
            taux_buts.append(goal_rate)
        ax.plot(np.arange(0, 100, 5), taux_buts, linestyle='-', label=model)
    ax.set_xlabel("Centile de la probabilité de tir")
    ax.set_ylabel("Taux de buts")
    ax.set_title("Taux de buts en fonction du centile de probabilité de tir")
    ax.grid(True)
    ax.set_xlim(100, 0)
    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_yticks(np.arange(0, 101, 10))
    ax.legend()


def cumulative_centiles_plot(y, Y, ax):
    n = len(y)
    curves = []
    for model, pred, _, __ in Y:
        curves.append(model)
        x_axis = np.arange(n)[::-1] * (100 / n)
        reverse_prob = pred[::-1]
        reverse_prob[::-1].sort()
        cum_percentile = np.cumsum(reverse_prob) * 100
        ax.plot(x_axis, cum_percentile / sum(pred), label=model)
    ax.set_xlabel("Centile de la probabilité de tir")
    ax.set_ylabel("Proportion")
    ax.set_title("Cumulatif des buts (en %)")
    ax.grid(True)
    ax.set_xlim(100, 0)
    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_yticks(np.arange(0, 101, 10))
    ax.legend()

def calibrate_display(classifier, y_val, n_bin=50, ax=None):
    curves = []
    for model, X, name in classifier: 
        curves.append(name)
        if len(model) != 2:
            CalibrationDisplay.from_estimator(model[0], X, y_val, n_bins=n_bin, name=name, ax=ax)
        else:
            CalibrationDisplay.from_predictions(model[0], model[1], n_bins=n_bin, name=name, ax=ax)
    ax.set_xlabel("Probabilité moyenne prédite (Classe 1 )")
    ax.set_ylabel("Proportion des positifs (Classe 1 )")
    ax.set_title("Diagramme de fiabilité : Courbes de calibration")
    ax.legend()


def all_plots(y, Y, classifier, y_val):
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))  # Create 2x2 grid
    ROC_plot(y, Y, axs[0, 0])  # Top-left plot
    Centiles_plot(y, Y, axs[0, 1])  # Top-right plot
    cumulative_centiles_plot(y, Y, axs[1, 0])  # Bottom-left plot
    calibrate_display(classifier, y_val, n_bin=50, ax=axs[1, 1])  # Bottom-right plot
    plt.tight_layout()  # Adjust layout
    plt.savefig(f"../figures/all_plots{classifier[0][2]}.png")  # Save the figure
    plt.show()  # Show the figure