import pandas as pd
import numpy as np
import statsmodels.discrete.discrete_model as sm


class PointsModel():

    def __init__(self):
        self.bin_bmi = ([[0, 20], [20, 25], [25, 30], [
                        30, 35], [35, float('inf')]], 5)
        self.bin_age = ([[0, 3], [3, 6], [6, 9], [9, 12], [
                        12, 15], [15, float('inf')]][::-1], 3)
        self.bin_zscore = ([[-float('inf'), -1], [-1, 0],
                            [0, 1], [1, 2], [2, float('inf')]][::-1], 1)
        self.bin_dict = {'bmi': self.bin_bmi,
                         'age': self.bin_age,
                         'zscore': self.bin_zscore}
        self.X = None
        self.y = None
        self.coefficients = None
        self.fitted = False

    def _getCoefficients(self, X, y):
        """
        This function builds the reduced logistic regression model,
        finds the significant variables, performs another logistic
        regression using just those variables, and keeps the
        coefficients in a dict for later use.
        """
        logit = sm.Logit(y, X)
        f = logit.fit()
        keep_variables = list(f.params[f.pvalues <= 0.05].index)
        self.keep_variables = keep_variables
        new_X = X[keep_variables]
        new_logit = sm.Logit(y, new_X)
        new_logit.fit().params
        self.coefficients = dict(new_logit.fit().params)
        self.X_old = self.X.copy()
        self.X = new_X

    def logistic(self, summed):
        """
        The logistic function.
        """
        return 1. / (1 + np.exp(-(summed)))

    def _getDiffScore(self, value, rangelist, scale_factor):
        """
        Calculates Sullivan et al. difference score. Unnormalized.
        """
        for i, group in enumerate(rangelist):
            if (value >= group[0]) and (value < group[1]):
                return scale_factor * float(i)

    def predict(self, X_input):
        X_input = X_input.copy()
        if not isinstance(X_input, pd.core.frame.DataFrame):
            raise Exception("""X must be a Pandas dataframe. \
                            (use pd.DataFrame(arr) if X is numpy array, \
                            and add column titles (for interpretability)""")
        X_input['total_points'] = 0
        for column in self.coefficients:
            vector = X_input[column]
            if column in self.bin_dict:
                points = map(lambda x:
                             self._getDiffScore(x,
                                                self.bin_dict[column][0],
                                                self.bin_dict[column][1]), vector)
            else:
                points = vector
            final_points = self.coefficients[column] * np.array(points)
            X_input['total_points'] += final_points
        X_input['risk_score'] = X_input['total_points'].apply(self.logistic)
        return X_input['total_points'], X_input['risk_score']

    def fit(self, X, y):
        """
        Calculates the total points score of each entry
        and then associates them with a risk score.
        """
        if isinstance(X, pd.core.frame.DataFrame):
            self.X = X
        else:
            raise Exception("""X must be a Pandas dataframe. \
                            (use pd.DataFrame(arr) if X is numpy array, \
                            and add column titles (for interpretability)""")
        self.y = y
        self._getCoefficients(self.X, self.y)
        self.X['total_points'], self.X['risk_score'] = self.predict(self.X)
        self.fitted = True

    def predict_proba(self, X_input):
        """
        Calculates the points core and risk likelihood of a new input.
        """
        probabilities = np.array(self.predict(X_input)[1])
        inverse = 1 - probabilities
        return np.vstack((probabilities, inverse)).T
