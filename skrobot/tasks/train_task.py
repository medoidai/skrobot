import os, joblib, copy

import pandas as pd

import numpy as np

from . import BaseTask

class TrainTask(BaseTask):
  """
  The :class:`.TrainTask` class can be used to fit a scikit-learn estimator/pipeline on train data.
  """
  def __init__ (self, estimator, train_data_set, estimator_params=None, field_delimiter=',', feature_columns='all', id_column='id', label_column='label', random_seed=42):
    """
    This is the constructor method and can be used to create a new object instance of :class:`.TrainTask` class.

    :param estimator: It can be either an estimator (e.g., LogisticRegression) or a pipeline ending with an estimator.
    :type estimator: scikit-learn {estimator, pipeline}

    :param train_data_set: The input train data set. It can be either a URL, a disk file path or a pandas DataFrame.
    :type train_data_set: {str, pandas DataFrame}

    :param estimator_params: The parameters to override in the provided estimator/pipeline. It defaults to None.
    :type estimator_params: dict, optional

    :param field_delimiter: The separation delimiter (comma for CSV, tab for TSV, etc.) used in the input train data set file. It defaults to ','.
    :type field_delimiter: str, optional

    :param feature_columns: Either 'all' to use from the input train data set file all the columns or a list of column names to select specific columns. It defaults to 'all'.
    :type feature_columns: {str, list}, optional

    :param id_column: The name of the column in the input train data set file containing the sample IDs. It defaults to 'id'.
    :type id_column: str, optional

    :param label_column: The name of the column in the input train data set file containing the ground truth labels. It defaults to 'label'.
    :type label_column: str, optional

    :param random_seed: The random seed used in the random number generator. It can be used to reproduce the output. It defaults to 42.
    :type random_seed: int, optional
    """
    arguments = copy.deepcopy(locals())

    super(TrainTask, self).__init__(TrainTask.__name__, arguments)

  def run(self, output_directory):
    """
    Run the task.

    The fitted estimator/pipeline is returned as a result and also stored in a *trained_model.pkl* pickle file under the output directory path.

    :param output_directory: The output directory path under which task-related generated files are stored.
    :type output_directory: str

    :return: The task's result. Specifically, the fitted estimator/pipeline.
    :rtype: dict
    """

    if isinstance(self.train_data_set, str):
      data_set_data_frame = pd.read_csv(self.train_data_set, delimiter=self.field_delimiter)
    else:
      data_set_data_frame = self.train_data_set.copy()

      data_set_data_frame.reset_index(inplace=True, drop=True)

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