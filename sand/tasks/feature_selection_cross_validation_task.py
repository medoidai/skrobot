import os, copy

import pandas as pd

import numpy as np

from sklearn.feature_selection import RFECV

from sklearn.pipeline import Pipeline

from . import BaseCrossValidationTask

class FeatureSelectionCrossValidationTask(BaseCrossValidationTask):
  def __init__ (self, estimator, train_data_set_file_path, estimator_params=None, field_delimiter=',', preprocessor=None, preprocessor_params=None, min_features_to_select=1, scoring='f1', feature_columns='all', id_column='id', label_column='label', random_seed=123456789, verbose=3, n_jobs=1):
    super(FeatureSelectionCrossValidationTask, self).__init__(FeatureSelectionCrossValidationTask.__name__, locals())

  def run(self, output_directory):
    self.train_data_set_data_frame = pd.read_csv(self.train_data_set_file_path, delimiter=self.field_delimiter)

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