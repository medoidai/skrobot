import os, copy

import pandas as pd

import numpy as np

from sklearn.feature_selection import RFECV

from sklearn.pipeline import Pipeline

from . import BaseCrossValidationTask

class FeatureSelectionCrossValidationTask(BaseCrossValidationTask):
  """
  The :class:`.FeatureSelectionCrossValidationTask` class can be used to perform feature selection with Recursive Feature Elimination using a scikit-learn estimator on some data.

  A scikit-learn preprocessor can be used on the input train data set before feature selection runs.

  It can support both stratified k-fold cross-validation as well as cross-validation with user-defined folds.

  By default, stratified k-fold cross-validation is used with the default parameters of :meth:`.stratified_folds` method.
  """
  def __init__ (self, estimator, train_data_set, estimator_params=None, field_delimiter=',', preprocessor=None, preprocessor_params=None, min_features_to_select=1, scoring='f1', feature_columns='all', id_column='id', label_column='label', random_seed=42, verbose=3, n_jobs=1):
    """
    This is the constructor method and can be used to create a new object instance of :class:`.FeatureSelectionCrossValidationTask` class.

    :param estimator: An estimator (e.g., LogisticRegression). It needs to provide feature importances through either a ``coef_`` or a ``feature_importances_`` attribute.
    :type estimator: scikit-learn estimator

    :param train_data_set: The input train data set. It can be either a URL, a disk file path or a pandas DataFrame.
    :type train_data_set: {str, pandas DataFrame}

    :param estimator_params: The parameters to override in the provided estimator. It defaults to None.
    :type estimator_params: dict, optional

    :param field_delimiter: The separation delimiter (comma for CSV, tab for TSV, etc.) used in the input train data set file. It defaults to ','.
    :type field_delimiter: str, optional

    :param preprocessor: The preprocessor you want to run on the input train data set before feature selection. You can set for example a scikit-learn ColumnTransformer, OneHotEncoder, etc. It defaults to None.
    :type preprocessor: scikit-learn preprocessor, optional

    :param preprocessor_params: The parameters to override in the provided preprocessor. It defaults to None.
    :type preprocessor_params: dict, optional

    :param min_features_to_select: The minimum number of features to be selected. This number of features will always be scored. It defaults to 1.
    :type min_features_to_select: int, optional

    :param scoring: A single scikit-learn scorer string (e.g., 'f1') or a callable that is built with scikit-learn ``make_scorer``. Note that when using custom scorers, each scorer should return a single value. It defaults to 'f1'.
    :type scoring: {str, callable}, optional

    :param feature_columns: Either 'all' to use from the input train data set file all the columns or a list of column names to select specific columns. It defaults to 'all'.
    :type feature_columns: {str, list}, optional

    :param id_column: The name of the column in the input train data set file containing the sample IDs. It defaults to 'id'.
    :type id_column: str, optional

    :param label_column: The name of the column in the input train data set file containing the ground truth labels. It defaults to 'label'.
    :type label_column: str, optional

    :param random_seed: The random seed used in the random number generator. It can be used to reproduce the output. It defaults to 42.
    :type random_seed: int, optional

    :param verbose: Controls the verbosity of output. The higher, the more messages. It defaults to 3.
    :type verbose: int, optional

    :param n_jobs: Number of jobs to run in parallel. -1 means using all processors. It defaults to 1.
    :type n_jobs: int, optional
    """

    super(FeatureSelectionCrossValidationTask, self).__init__(FeatureSelectionCrossValidationTask.__name__, locals())

  def run(self, output_directory):
    """
    Run the task.

    The selected features are returned as a result and also stored in a *features_selected.txt* file under the output directory path.

    :param output_directory: The output directory path under which task-related generated files are stored.
    :type output_directory: str
    
    :return: The task's result. Specifically, the selected features, which can be either column names from the input train data set or column indexes from the preprocessed data set, depending on whether a ``preprocessor`` was used or not.
    :rtype: list
    """
    if isinstance(self.train_data_set, str):
      self.train_data_set_data_frame = pd.read_csv(self.train_data_set, delimiter=self.field_delimiter)
    else:
      self.train_data_set_data_frame = self.train_data_set.copy()

      self.train_data_set_data_frame.reset_index(inplace=True, drop=True)

    y = self.train_data_set_data_frame[self.label_column]

    X = self.train_data_set_data_frame.drop(columns=[self.label_column, self.id_column])

    if self.feature_columns != 'all':
      X = X[self.feature_columns]

    np.random.seed(self.random_seed)

    model = RFECV(self._build_estimator(), min_features_to_select=self.min_features_to_select, step=1, cv=self._build_cv_splits(X, y), scoring=self.scoring, verbose=self.verbose, n_jobs=self.n_jobs)

    if self.preprocessor:
      model = Pipeline(steps=[('preprocessor', self._build_preprocessor()), ('selection', model)])

    model.fit(X, y)

    if self.preprocessor:
      features_selected = np.nonzero(model.named_steps.selection.support_)[0].tolist()
    else:
      features_selected = X.columns[model.support_].values.tolist()

    with open(os.path.join(output_directory, 'features_selected.txt'), "w") as f: f.writelines('\n'.join(map(str, features_selected)))

    return features_selected

  def _build_preprocessor (self):
    preprocessor = copy.deepcopy(self.preprocessor)

    if self.preprocessor_params: preprocessor.set_params(**self.preprocessor_params)

    return preprocessor

  def _build_estimator (self):
    estimator = copy.deepcopy(self.estimator)

    if self.estimator_params: estimator.set_params(**self.estimator_params)

    return estimator