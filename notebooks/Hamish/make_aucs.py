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
from sklearn.model_selection import GridSearchCV


def main(data=None, models=None, kfolds=5):

    """
    Run the main script.

    data is in format [xdata, ydata]

    models is in format dict(model_name:sckit-learn model)

    If model is to be regularised, model name should end in 'reg'
    """

    # Set up plot
    _, plot_axis = plt.subplots()
    plot_axis.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='Luck')

    # Create training and cross validation set
    kfold_index = make_kfold_index(data, kfolds)

    for model_name, model in models.items():

        # Set up parameters
        mean_tpr = 0.0
        mean_auc = []

        # Iterate through cross validation sets: fit and test models
        for i in range(kfolds):

            # Train model; update fpr and auc
            fpr, tpr, auc = train_and_test(data, kfold_index[i], \
            model, regularise=model_name[-3:] == 'reg')
            mean_tpr += interp(np.linspace(0, 1, 100), fpr, tpr)
            mean_tpr[0] = 0.0
            mean_auc.append(auc)

        # Plot average ROC curve
        mean_tpr /= kfolds
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(mean_auc)
        label = make_label(model_name, mean_tpr, np.linspace(0, 1, 100), mean_auc)
        plot_axis.plot(np.linspace(0, 1, 100), mean_tpr, linestyle='-', label=label, lw=2)

    plot_axis.set_xlim([-0.05, 1.05])
    plot_axis.set_ylim([-0.05, 1.05])
    plot_axis.set_xlabel('False Positive Rate')
    plot_axis.set_ylabel('True Positive Rate')
    plot_axis.legend(loc='center left', bbox_to_anchor=(1, 0.5))


def make_label(model_name, mean_tpr, mean_fpr, mean_auc):

    """
    Create label for the plot with the mean auc, sensitivity and specificity.

    The sensitivity and specificity are chosen at the optimal threshold
    determined by Youden's J Statistic.
    """

    mean_j_statistic = mean_tpr - mean_fpr
    j_idx = np.argmax(mean_j_statistic)
    sensitivity = mean_tpr[j_idx]
    specificity = 1-mean_fpr[j_idx]
    label = model_name + ' (AUC: {:.3f}, Sens: {:.3f}, Spec: {:.3f})'.format(mean_auc, \
    sensitivity, specificity)

    return label


def make_kfold_index(data, kfolds):

    """
    Create index vectors that can be used to create stratified
    cross-validation folds.
    """

    skf = StratifiedKFold(n_splits=kfolds, random_state=None, shuffle=False)
    kfold_index = [[i, j] for i, j in skf.split(data[0], data[1])]

    return kfold_index


def train_and_test(data, kfold_index, model, regularise=False):

    """Train and test the model for a given fold of cross-validation."""

    # Create cross validation training and test sets
    data_train = [data[0].loc[kfold_index[0]], data[1].loc[kfold_index[0]]]
    data_test = [data[0].loc[kfold_index[1]], data[1].loc[kfold_index[1]]]

    if regularise:
        reg_coeffs = np.logspace(-6, 1, 10)
        model = GridSearchCV(estimator=model, param_grid=dict(C=reg_coeffs), n_jobs=-1)

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
