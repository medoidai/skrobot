import os, copy

import pandas as pd

from . import BaseTask

class PredictionTask(BaseTask):
  def __init__ (self, estimator, data_set_file_path, field_delimiter=',', feature_columns='all', id_column='id', prediction_column='prediction', threshold=0.5):
    arguments = copy.deepcopy(locals())

    super(PredictionTask, self).__init__(PredictionTask.__name__, arguments)

  def run(self, output_directory):
    data_set_data_frame = pd.read_csv(self.data_set_file_path, delimiter=self.field_delimiter)

    ids = data_set_data_frame[self.id_column]

    X = data_set_data_frame.drop(columns=[self.id_column])

    if self.feature_columns != 'all':
      X = X[self.feature_columns]

    predictions = pd.DataFrame({ self.id_column : ids, self.prediction_column : self._calculate_y_hat_for_threshold(self.estimator, X, self.threshold) })

    predictions.to_csv(os.path.join(output_directory, f'predictions.csv'), index=False)

    return predictions

  def _calculate_y_hat_for_threshold (self, estimator, X, threshold):
    y_proba = estimator.predict_proba(X)

    y_hat = y_proba[:, 1] >= threshold

    return y_hat.astype(int)