import os, copy

import pandas as pd

from . import BaseTask

class PredictionTask(BaseTask):
  def __init__ (self, estimator, data_set_file_path, field_delimiter=',', feature_columns='all', id_column='id', prediction_column='prediction'):
    arguments = copy.deepcopy(locals())

    super(PredictionTask, self).__init__(PredictionTask.__name__, arguments)

  def run(self, output_directory):
    data_set_data_frame = pd.read_csv(self.data_set_file_path, delimiter=self.field_delimiter)

    ids = data_set_data_frame[self.id_column]

    X = data_set_data_frame.drop(columns=[self.id_column])

    if self.feature_columns != 'all':
      X = X[self.feature_columns]

    predictions = pd.DataFrame({ self.id_column : ids, self.prediction_column : self.estimator.predict(X) })

    predictions.to_csv(os.path.join(output_directory, f'predictions.csv'), index=False)

    return predictions