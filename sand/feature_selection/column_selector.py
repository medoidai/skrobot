from sklearn.base import BaseEstimator

import numpy as np

class ColumnSelector(BaseEstimator):
    def __init__(self, cols=None, drop_axis=False):
        self.drop_axis = drop_axis

        self.cols = cols

    def fit_transform(self, X, y=None):
        return self.transform(X=X, y=y)

    def transform(self, X, y=None):
        if hasattr(X, 'loc') or hasattr(X, 'iloc'):
            if type(self.cols) == tuple:
                self.cols = list(self.cols)

            types = {type(i) for i in self.cols}

            if len(types) > 1: raise Exception('Elements in `cols` should be all of the same data type.')

            if isinstance(self.cols[0], int):
                t = X.iloc[:, self.cols].values
            elif isinstance(self.cols[0], str):
                t = X.loc[:, self.cols].values
            else:
                raise Exception('Elements in `cols` should be either `int` or `str`.')
        else:
            t = X[:, self.cols]

        if t.shape[-1] == 1 and self.drop_axis:
            t = t.reshape(-1)

        if len(t.shape) == 1 and not self.drop_axis:
            t = t[:, np.newaxis]

        return t

    def fit(self, X, y=None):
        return self