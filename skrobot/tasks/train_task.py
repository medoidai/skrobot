import os, joblib, copy

import pandas as pd

import numpy as np

from . import BaseTask

class TrainTask(BaseTask):
  def __init__ (self, estimator, train_data_set_file_path, estimator_params=None, field_delimiter=',', feature_columns='all', id_column='id', label_column='label', random_seed=123456789):
    arguments = copy.deepcopy(locals())

    super(TrainTask, self).__init__(TrainTask.__name__, arguments)

  def run(self, output_directory):
    data_set_data_frame = pd.read_csv(self.train_data_set_file_path, delimiter=self.field_delimiter)

    y = data_set_data_frame[self.label_column]

    X = data_set_data_frame.drop(columns=[self.label_column, self.id_column])

    if self.feature_columns != 'all':
      X = X[self.feature_columns]

    np.random.seed(self.random_seed)

    estimator = self._build_estimator()

    estimator.fit(X, y)

    joblib.dump(estimator, os.path.join(output_directory, 'trained_model.pkl'))

    return { 'estimator': estimator }

  def _build_estimator (self):
    estimator = copy.deepcopy(self.estimator)

    if self.estimator_params: estimator.set_params(**self.estimator_params)

    return estimator