"""
A few functions to help with the RAIL code.

sens_and_spec: Compute sensitivity and specificity.

match_sens: Find the threshold that best matches a reference sensitivity, and
            print the sensitivity and specificity at this threshold.

match_spec: Find the threshold that best matches a reference specificity, and
            print the sensitivity and specificity at this threshold.

PlotModels: Plot a number of model ROCs in one plot, including the
average AUC over k-fold cross-validation, as well as sensitivity and specificity
calculated with Youden's J-Statistic.
"""

# Imports
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
import pandas as pd

# Imports scikit-learn
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.model_selection import GridSearchCV


def sens_and_spec(y_true, y_pred):
    """ Calculate sensitivity and specificity comparing predictions with reference results"""

    true_negatives, false_positives, false_negatives, true_positives = metrics.confusion_matrix(
        y_true, y_pred).ravel()
    sensitivity = true_positives / (true_positives + false_negatives)
    specificity = true_negatives / (true_negatives + false_positives)

    return sensitivity, specificity


def match_sens(fpr, tpr, ref_sens):
    """ Calculate specificity for the sensitivity value that best matches a reference sensitivity"""

    # Find idx of closest tpr value greater than the reference sensitivity
    idx_greater = tpr >= ref_sens
    fpr = fpr[idx_greater]
    tpr = tpr[idx_greater]
    idx_closest = np.argmin(tpr - ref_sens)
    sensitivity = tpr[idx_closest]
    specificity = 1 - fpr[idx_closest]

    return sensitivity, specificity


def match_spec(fpr, tpr, ref_spec):
    """ Calculate sensitivity for the specificity value that best matches a reference specificity"""

    # Find idx of closest tpr value greater than the reference sensitivity
    spec = 1 - fpr
    idx_greater = spec >= ref_spec
    spec = spec[idx_greater]
    tpr = tpr[idx_greater]
    idx_closest = np.argmin(spec - ref_spec)
    sensitivity = tpr[idx_closest]
    specificity = spec[idx_closest]

    return sensitivity, specificity


class PlotModels(object):

    """
    Build an ROC curve for a range of models using kfold cross-validation.

    Parameters
    ----------
    data: array, shape = [n_samples, 2]
        Array of data with x values in column 1 and y values in column 2.
    models: dict, shape = {n_models}
        Dictionary of models specificied by model_name: model_object.
    kfolds: int (default 5)
        Number of cross-validation folds.
    pipeline: list
        The function objects required to pre-process the data.
    """

    def __init__(self, data=None, models=None, kfolds=0, pipeline=None):

        self.data = data
        self.models = models
        self.kfolds = kfolds
        self.pipeline = pipeline
        self._plot_vars = {}
        self._kfold_index = []

    def _make_kfold_index(self):
        """
        Create index vectors that can be used to create stratified
        cross-validation folds.
        """
        skf = StratifiedKFold(n_splits=self.kfolds,
                              random_state=None, shuffle=False)
        self._kfold_index = [[i, j]
                             for i, j in skf.split(self.data[0], self.data[1])]

    def _make_label(self):
        """
        Create label for the plot with the mean auc, sensitivity and specificity.

        The sensitivity and specificity are chosen at the optimal threshold
        determined by Youden's J Statistic.
        """

        # Iterate through models
        for model_name, _ in self.models.items():

            # Calculate j-statistic = tpr - fpr
            mean_j_statistic = self._plot_vars[model_name][1] - \
                self._plot_vars[model_name][0]

            # Calculate max of j-statistic and find associate sensitivity and
            # specificity
            j_idx = np.argmax(mean_j_statistic)
            sensitivity = self._plot_vars[model_name][1][j_idx]
            specificity = 1 - self._plot_vars[model_name][0][j_idx]

            # Add label to self._plot_vars
            self._plot_vars[model_name][3] = \
                model_name + ' (AUC: {:.3f}, Sens: {:.3f}, Spec: {:.3f})'.format(
                    self._plot_vars[model_name][2], sensitivity, specificity)

    def _process_pipeline(self, data_train, data_test):
        """
        Transform the data using the functions given in self.pipeline.
        """

        x_train = data_train[0].copy()
        x_test = data_test[0].copy()

        if self.pipeline:
            for process, variables in self.pipeline:

                if variables == 'all':
                    process.fit(x_train)
                    x_train = pd.DataFrame(process.transform(x_train),
                                           index=x_train.index)
                    x_test = pd.DataFrame(process.transform(x_test),
                                          index=x_test.index)
                    columns_left = list(x_train.columns)

                elif variables == 'from before':
                    columns_hold = list(
                        set(x_train.columns) - set(columns_left))
                    x_train_temp = x_train.copy()
                    x_test_temp = x_test.copy()
                    process.fit(x_train[columns_left])
                    x_train = pd.DataFrame(process.transform(x_train[columns_left]),
                                           index=x_train.index)
                    x_train[columns_hold] = x_train_temp[columns_hold]
                    x_test = pd.DataFrame(process.transform(x_test[columns_left]),
                                          index=x_test.index)
                    columns_left = list(x_test.columns)
                    x_test[columns_hold] = x_test_temp[columns_hold]

                else:
                    columns_hold = list(set(x_train.columns) - set(variables))
                    x_train_temp = x_train.copy()
                    x_test_temp = x_test.copy()
                    process.fit(x_train[variables])
                    x_train = pd.DataFrame(process.transform(x_train[variables]),
                                           index=x_train.index)
                    columns_left = list(x_train.columns)
                    x_train[columns_hold] = x_train_temp[columns_hold]
                    x_test = pd.DataFrame(process.transform(x_test[variables]),
                                          index=x_test.index)
                    x_test[columns_hold] = x_test_temp[columns_hold]

            data_train[0] = x_train
            data_test[0] = x_test

        return data_train, data_test

    def _compute_results(self, model, data_train, data_test):
        """ Compute the vectors of false positive rate and true positive rate,
        as well as the auc, for a given model and training and testing data"""

        # Pre-process data
        data_train, data_test = self._process_pipeline(
            data_train, data_test)

        # Fit model
        model.fit(data_train[0], data_train[1])

        # Score models
        pred_proba = model.predict_proba(data_test[0])[:, 1]
        fpr, tpr, _ = metrics.roc_curve(data_test[1], pred_proba)
        auc = metrics.roc_auc_score(data_test[1], pred_proba)

        return fpr, tpr, auc

    def fit(self, data=None):
        """Train and test the model for each fold of cross-validation."""

        if data:
            self.data = data

        # Define stratified kfold indices
        self._make_kfold_index()

        # Initialise plot variables
        self._plot_vars = {model_name: [[], [], 0, '']
                           for model_name in self.models.keys()}

        # Iterate through models
        for model_name, model in self.models.items():

            # Initialise for-loop updates
            mean_fpr = np.linspace(0, 1, 100)
            mean_tpr = 0.0
            mean_auc = 0.0

            # regularise
            if model_name[-3:] == 'reg':
                model = GridSearchCV(estimator=model, refit=True,
                                     param_grid=dict(C=np.logspace(-6, 1, 10)),
                                     scoring='roc_auc', n_jobs=-1)
            if self.kfolds:

                for i in range(self.kfolds):

                    # Create cross validation training and test sets
                    data_train = [self.data[0].loc[self._kfold_index[i][0]],
                                  self.data[1].loc[self._kfold_index[i][0]]]
                    data_test = [self.data[0].loc[self._kfold_index[i][1]],
                                 self.data[1].loc[self._kfold_index[i][1]]]

                    fpr, tpr, auc = self._compute_results(
                        model, data_train, data_test)

                    # Update
                    mean_tpr += interp(mean_fpr, fpr, tpr)
                    mean_tpr[0] = 0.0
                    mean_auc += auc

                # Divide by kfolds to get average
                mean_tpr /= self.kfolds
                mean_tpr[-1] = 1.0
                mean_auc /= self.kfolds

            # Add to self._plot_vars
            self._plot_vars[model_name][:3] = [mean_fpr, mean_tpr, mean_auc]

        self._make_label()

        return self

    def plot(self):
        """
        Plot the ROC for all the models.
        Legend includes AUC, sensitivity and specificity,
        calculated with Youden's J-statistic
        """

        # Set up plot
        _, plot_axis = plt.subplots()
        plot_axis.plot([0, 1], [0, 1], linestyle='--',
                       lw=2, color='k', label='Luck')

        for _, (plt_x, plt_y, _, plt_label) in self._plot_vars.items():
            plot_axis.plot(plt_x, plt_y, linestyle='-',
                           label=plt_label, lw=2)

        plot_axis.set_xlim([-0.05, 1.05])
        plot_axis.set_ylim([-0.05, 1.05])
        plot_axis.set_xlabel('False Positive Rate')
        plot_axis.set_ylabel('True Positive Rate')
        plot_axis.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        max_auc_idx = np.argmax([list(self._plot_vars.values())[i][2]
                                 for i in range(len(self.models))])
        print('Best model: ' + list(self._plot_vars.values())[max_auc_idx][3])
