import os, copy

import pandas as pd

import numpy as np

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from . import BaseCrossValidationTask

class HyperParametersSearchCrossValidationTask(BaseCrossValidationTask):
  """
  The :class:`.HyperParametersSearchCrossValidationTask` class can be used to do the feature selection task on some data. It extends the :class:`.BaseCrossValidationTask` class.


  """
  def __init__ (self, estimator, search_params, train_data_set_file_path, estimator_params=None, field_delimiter=',', scorers=['roc_auc', 'average_precision', 'f1', 'precision', 'recall', 'accuracy'], feature_columns='all', id_column='id', label_column='label', objective_score='f1', random_seed=42, verbose=3, n_jobs=1, return_train_score=True):
    """
	This is the constructor method and can be used to create a new object instance of :class:`.HyperParametersSearchCrossValidationTask` class.
	
	:param estimator: It can be either a scikit-learn estimator (e.g., LogisticRegression) or a scikit-learn pipeline ending with an estimator. The estimator needs to be able to predict probabilities through a ``predict_proba`` method.
    :type estimator: scikit-learn {estimator, pipeline}
	
	:param search_params : Dictionary with parameters names (str) as keys and lists of parameter settings to try as values, or a list of such dictionaries, in which case the grids spanned by each dictionary in the list are explored. This enables searching over any sequence of parameter settings. 
	:type search_params : dict or list of dictionaries
	
	:param train_data_set_file_path: The file path of the training data set. It can be either a URL or a disk file path.
    :type train_data_set_file_path: str
	
	:param estimator_params: The parameters to override in the provided estimator. It can be either a URL or a disk file path. It defaults to None.
    :type estimator_params: dict, optional
	
	:param field_delimiter: The separation delimiter (comma for CSV, tab for TSV, etc.) used in the input data set file. It defaults to ','.
    :type field_delimiter: str, optional
	
	:param scorers: The scorers with respect to, to calculate the performance. It defaults to ['roc_auc', 'average_precision', 'f1', 'precision', 'recall', 'accuracy'].
	:type : str, callable, list/tuple or dict
	
	:param feature_columns: Either 'all' to use from the input data set file all the columns or a list of column names to select specific columns. It defaults to 'all'.
    :type feature_columns: {str, list}, optional
	
	:param id_column: The name of the column in the input data set file containing the sample IDs. It defaults to 'id'.
    :type id_column: str, optional
	
	:param label_column: The name of the column in the input data set file containing the ground truth labels. It defaults to 'label'.
    :type label_column: str, optional
	
	:param objective_score : 
	:type objective_score :
	
	:param random_seed: The random seed used in the random number generator. It can be used to reproduce the output. It defaults to 42.
    :type random_seed: int, optional
	
	:param verbose : Controls verbosity of output. It defaults to 3.
	:type verbose : int, optional 
	
	:param n_jobs : Number of cores to run in parallel while fitting across folds. It defaults to 1.
	:type n_jobs : int, optional
	
	:param return_train_score : If you want to return the score on train set. It defaults to True.
	:type return_train_score : bool
	"""
	super(HyperParametersSearchCrossValidationTask, self).__init__(HyperParametersSearchCrossValidationTask.__name__, locals())

    self.grid_search()

    pd.set_option('display.max_colwidth', None)

  def grid_search(self):
    """
	This task performs the grid search on the search parameters and update the arguments.
	"""
    options = self._filter_arguments(locals())

    self._update_arguments({ 'search_options' : options})

    self._update_arguments({ 'search_method': 'grid' })

    return self

  def random_search(self, n_iters=200):
    """
	This task performs random search on the search parameters and update the arguments.
	
	:param n_iter : The number of iterations. It defaults to 200.
	:type n_iter : int
	"""
    options = self._filter_arguments(locals())

    self._update_arguments({ 'search_options' : options})

    self._update_arguments({ 'search_method': 'random' })

    return self

  def run(self, output_directory):
    """
    A method for running the task.
    
    :param output_directory: The output directory path under which task-related generated files are stored.
    :type output_directory: str
    
    :return: A dictionary contains the best estimator, best params, best index, best score and search results. 
	:rtype: dict
  	"""
    self.train_data_set_data_frame = pd.read_csv(self.train_data_set_file_path, delimiter=self.field_delimiter)

    y = self.train_data_set_data_frame[self.label_column]

    X = self.train_data_set_data_frame.drop(columns=[self.label_column, self.id_column])

    if self.feature_columns != 'all':
      X = X[self.feature_columns]

    np.random.seed(self.random_seed)

    search = self._build_search_method(self._build_cv_splits(X, y))

    search.fit(X, y)

    cv_results = pd.DataFrame(search.cv_results_).reset_index()

    cv_results.columns = cv_results.columns.str.replace('_test_', '_validation_')

    cv_results.to_html(os.path.join(output_directory, f'search_results_optimized_for_{self.objective_score}.html'), index=False)

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