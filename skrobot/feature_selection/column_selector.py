from sklearn.base import BaseEstimator

import numpy as np

class ColumnSelector(BaseEstimator):
    """
    The :class:`.ColumnSelector` class is an implementation of a column selector for scikit-learn pipelines.

    It can be used for manual feature selection to select specific columns from an input data set.

    It can select columns either by integer indices or by names.
    """
    def __init__(self, cols, drop_axis=False):
        """
        This is the constructor method and can be used to create a new object instance of :class:`.ColumnSelector` class.

        :param cols: A non-empty list specifying the columns to be selected. For example, [1, 4, 5] to select the 2nd, 5th, and 6th columns, and ['A','C','D'] to select the columns A, C and D.
        :type cols: list

        :param drop_axis: Can be used to reshape the output data set from (n_samples, 1) to (n_samples) by dropping the last axis. It defaults to False.
        :type drop_axis: bool, optional
        """

        self.drop_axis = drop_axis

        self.cols = cols

    def fit_transform(self, X, y=None):
        """
        Returns a slice of the input data set.

        :param X: Input vectors of shape (n_samples, n_features), where n_samples is the number of samples and n_features is the number of features.
        :type X: {NumPy array, pandas DataFrame, SciPy sparse matrix}

        :param y: Ignored.
        :type y: None

        :return: Subset of the input data set of shape (n_samples, k_features), where n_samples is the number of samples and k_features <= n_features.
        :rtype: {NumPy array, SciPy sparse matrix}
        """

        return self.transform(X=X, y=y)

    def transform(self, X, y=None):
        """
        Returns a slice of the input data set.

        :param X: Input vectors of shape (n_samples, n_features), where n_samples is the number of samples and n_features is the number of features.
        :type X: {NumPy array, pandas DataFrame, SciPy sparse matrix}

        :param y: Ignored.
        :type y: None

        :return: Subset of the input data set of shape (n_samples, k_features), where n_samples is the number of samples and k_features <= n_features.
        :rtype: {NumPy array, SciPy sparse matrix}
        """

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
        """
        This is a mock method and does nothing.

        :param X: Ignored.
        :type X: None

        :param y: Ignored.
        :type y: None

        :return: The object instance itself.
        :rtype: :class:`.ColumnSelector`
        """

        return self