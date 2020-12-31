import os, copy

import pandas as pd

import numpy as np

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from . import BaseCrossValidationTask

class HyperParametersSearchCrossValidationTask(BaseCrossValidationTask):
  """
  The :class:`.HyperParametersSearchCrossValidationTask` class can be used to search the best hyperparameters of a scikit-learn estimator/pipeline on some data.

  **Cross-Validation**

  It can support both stratified k-fold cross-validation as well as cross-validation with user-defined folds.

  By default, stratified k-fold cross-validation is used with the default parameters of :meth:`.stratified_folds` method.

  **Search**

  It can support both grid search as well as random search.

  By default, grid search is used.
  """
  def __init__ (self, estimator, search_params, train_data_set, estimator_params=None, field_delimiter=',', scorers=['roc_auc', 'average_precision', 'f1', 'precision', 'recall', 'accuracy'], feature_columns='all', id_column='id', label_column='label', objective_score='f1', random_seed=42, verbose=3, n_jobs=1, return_train_score=True):
    """
    This is the constructor method and can be used to create a new object instance of :class:`.HyperParametersSearchCrossValidationTask` class.

    :param estimator: It can be either an estimator (e.g., LogisticRegression) or a pipeline ending with an estimator.
    :type estimator: scikit-learn {estimator, pipeline}

    :param search_params: Dictionary with hyperparameters names as keys and lists of hyperparameter settings to try as values, or a list of such dictionaries, in which case the grids spanned by each dictionary in the list are explored. This enables searching over any sequence of hyperparameter settings.
    :type search_params: {dict, list of dictionaries}

    :param train_data_set: The input train data set. It can be either a URL, a disk file path or a pandas DataFrame.
    :type train_data_set: {str, pandas DataFrame}

    :param estimator_params: The parameters to override in the provided estimator/pipeline. It defaults to None.
    :type estimator_params: dict, optional

    :param field_delimiter: The separation delimiter (comma for CSV, tab for TSV, etc.) used in the input train data set file. It defaults to ','.
    :type field_delimiter: str, optional

    :param scorers: Multiple metrics to evaluate the predictions on the hold-out data. Either give a list of (unique) strings or a dict with names as keys and callables as values. The callables should be scorers built using scikit-learn ``make_scorer``. Note that when using custom scorers, each scorer should return a single value. It defaults to ['roc_auc', 'average_precision', 'f1', 'precision', 'recall', 'accuracy'].
    :type scorers: {list, dict}, optional

    :param feature_columns: Either 'all' to use from the input train data set file all the columns or a list of column names to select specific columns. It defaults to 'all'.
    :type feature_columns: {str, list}, optional

    :param id_column: The name of the column in the input train data set file containing the sample IDs. It defaults to 'id'.
    :type id_column: str, optional

    :param label_column: The name of the column in the input train data set file containing the ground truth labels. It defaults to 'label'.
    :type label_column: str, optional

    :param objective_score: The scorer that would be used to find the best hyperparameters for refitting the best estimator/pipeline at the end. It defaults to 'f1'.
    :type objective_score: str, optional

    :param random_seed: The random seed used in the random number generator. It can be used to reproduce the output. It defaults to 42.
    :type random_seed: int, optional

    :param verbose: Controls the verbosity of output. The higher, the more messages. It defaults to 3.
    :type verbose: int, optional

    :param n_jobs: Number of jobs to run in parallel. -1 means using all processors. It defaults to 1.
    :type n_jobs: int, optional

    :param return_train_score: If False, training scores will not be computed and returned. Computing training scores is used to get insights on how different parameter settings impact the overfitting/underfitting trade-off. It defaults to True.
    :type return_train_score: bool, optional
    """

    super(HyperParametersSearchCrossValidationTask, self).__init__(HyperParametersSearchCrossValidationTask.__name__, locals())

    self.grid_search()

    pd.set_option('display.max_colwidth', None)

  def grid_search(self):
    """
    Optional method.

    Use the grid search method when searching the best hyperparameters.

    :return: The object instance itself.
    :rtype: :class:`.HyperParametersSearchCrossValidationTask`
    """

    options = self._filter_arguments(locals())

    self._update_arguments({ 'search_options' : options})

    self._update_arguments({ 'search_method': 'grid' })

    return self

  def random_search(self, n_iters=200):
    """
    Optional method.

    Use the random search method when searching the best hyperparameters.

    :param n_iters: Number of hyperparameter settings that are sampled. ``n_iters`` trades off runtime vs quality of the solution. It defaults to 200.
    :type n_iters: int, optional

    :return: The object instance itself.
    :rtype: :class:`.HyperParametersSearchCrossValidationTask`
    """

    options = self._filter_arguments(locals())

    self._update_arguments({ 'search_options' : options})

    self._update_arguments({ 'search_method': 'random' })

    return self

  def run(self, output_directory):
    """
    Run the task.

    The search results (``search_results``) are stored also in a *search_results.html* file as a static HTML table under the output directory path.

    :param output_directory: The output directory path under which task-related generated files are stored.
    :type output_directory: str
    
    :return: The task's result. Specifically, **1)** ``best_estimator``: The estimator/pipeline that was chosen by the search, i.e. estimator/pipeline which gave best score on the hold-out data. **2)** ``best_params``: The hyperparameters setting that gave the best results on the hold-out data. **3)** ``best_score``: Mean cross-validated score of the ``best_estimator``. **4)** ``search_results``: Metrics measured for each of the hyperparameters setting in the search. **5)** ``best_index``: The index (of the ``search_results``) which corresponds to the best candidate hyperparameters setting.
    :rtype: dict
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

    search = self._build_search_method(self._build_cv_splits(X, y))

    search.fit(X, y)

    cv_results = pd.DataFrame(search.cv_results_).reset_index()

    cv_results.columns = cv_results.columns.str.replace('_test_', '_validation_')

    cv_results.to_html(os.path.join(output_directory, 'search_results.html'), index=False)

    return { 'best_estimator': search.best_estimator_, 'best_params': search.best_params_, 'best_index': search.best_index_, 'best_score': search.best_score_, 'search_results': cv_results }

  def _build_search_method (self, cv):
    if self.search_method == 'random':
      return RandomizedSearchCV(self._build_estimator(), self.search_params, cv=cv, scoring=self.scorers, refit=self.objective_score, return_train_score=self.return_train_score, n_jobs=self.n_jobs, verbose=self.verbose, n_iter=self.search_options['n_iters'], random_state=self.random_seed)
    else:
      return GridSearchCV(self._build_estimator(), self.search_params, cv=cv, scoring=self.scorers, refit=self.objective_score, return_train_score=self.return_train_score, n_jobs=self.n_jobs, verbose=self.verbose)

  def _build_estimator (self):
    estimator = copy.deepcopy(self.estimator)

    if self.estimator_params: estimator.set_params(**self.estimator_params)

    return estimator