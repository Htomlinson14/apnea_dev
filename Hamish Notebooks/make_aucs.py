"""
Simple script to plot the ROC and AUC for a range of models.
"""

# Imports
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt

# Imports scikit-learn
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics


def main(data=None, models=None, kfolds=5):

    """
    Run the main script.

    data is in format [xdata, ydata]
    models is in format dict(model_name:sckit-learn model)
    """

    # Set up plot
    _, plot_axis = plt.subplots()
    plot_axis.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='Luck')

    # Create training and cross validation set
    kfold_index = make_kfold_index(data, kfolds)

    for model_name, model in models.items():

        print("Fitting and testing model: {}".format(model_name))

        # Set up parameters
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        mean_auc = []

        # Iterate through cross validation sets: fit and test models
        for i in range(kfolds):

            # Train model; update fpr and auc
            fpr, tpr, auc = train_and_test(data, kfold_index[i], model)
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            mean_auc.append(auc)

        # Plot average ROC curve
        mean_tpr /= kfolds
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(mean_auc)
        plot_axis.plot(mean_fpr, mean_tpr, linestyle='-', \
        label=model_name + ' (AUC = {:.3f})'.format(mean_auc), lw=2)

    plot_axis.set_xlim([-0.05, 1.05])
    plot_axis.set_ylim([-0.05, 1.05])
    plot_axis.set_xlabel('False Positive Rate')
    plot_axis.set_ylabel('True Positive Rate')
    plot_axis.legend(loc="lower right")


def make_kfold_index(data, kfolds):

    """
    Create index vectors that can be used to create stratified
    cross-validation folds.
    """

    skf = StratifiedKFold(n_splits=kfolds, random_state=None, shuffle=False)
    kfold_index = [[i, j] for i, j in skf.split(data[0], data[1])]

    return kfold_index


def train_and_test(data, kfold_index, model):

    """Train and test the model for a given fold of cross-validation."""

    # Create cross validation training and test sets
    data_train = [data[0].loc[kfold_index[0]], data[1].loc[kfold_index[0]]]
    data_test = [data[0].loc[kfold_index[1]], data[1].loc[kfold_index[1]]]

    # Fit model
    model.fit(data_train[0], data_train[1])

    # Score models
    pred_proba = model.predict_proba(data_test[0])[:, 1]

    # Calculate metrics
    fpr, tpr, _ = metrics.roc_curve(data_test[1], pred_proba)
    auc = metrics.roc_auc_score(data_test[1], pred_proba)

    # Return performance metrics
    return fpr, tpr, auc


if __name__ == "__main__":
    # execute only if run as a script
    main()
