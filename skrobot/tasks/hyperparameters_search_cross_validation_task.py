import os, copy

import pandas as pd

import numpy as np

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from . import BaseCrossValidationTask

class HyperParametersSearchCrossValidationTask(BaseCrossValidationTask):
  def __init__ (self, estimator, search_params, train_data_set_file_path, estimator_params=None, field_delimiter=',', scorers=['roc_auc', 'average_precision', 'f1', 'precision', 'recall', 'accuracy'], feature_columns='all', id_column='id', label_column='label', objective_score='f1', random_seed=123456789, verbose=3, n_jobs=1, return_train_score=True):
    super(HyperParametersSearchCrossValidationTask, self).__init__(HyperParametersSearchCrossValidationTask.__name__, locals())

    self.grid_search()

    pd.set_option('display.max_colwidth', -1)

  def grid_search(self):
    options = self.filter_arguments(locals())

    self.update_arguments({ 'search_options' : options})

    self.update_arguments({ 'search_method': 'grid' })

    return self

  def random_search(self, n_iters=200):
    options = self.filter_arguments(locals())

    self.update_arguments({ 'search_options' : options})

    self.update_arguments({ 'search_method': 'random' })

    return self

  def run(self, output_directory):
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