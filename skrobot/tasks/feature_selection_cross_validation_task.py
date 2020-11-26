import os, copy

import pandas as pd

import numpy as np

from sklearn.feature_selection import RFECV

from sklearn.pipeline import Pipeline

from . import BaseCrossValidationTask

class FeatureSelectionCrossValidationTask(BaseCrossValidationTask):
  """
  The :class:`.FeatureSelectionCrossValidationTask` class can be used to do the feature selection task on some data. It extends the :class:`.BaseCrossValidationTask` class.
  
  """
  def __init__ (self, estimator, train_data_set_file_path, estimator_params=None, field_delimiter=',', preprocessor=None, preprocessor_params=None, min_features_to_select=1, scoring='f1', feature_columns='all', id_column='id', label_column='label', random_seed=42, verbose=3, n_jobs=1):
    """
    This is the constructor method and can be used to create a new object instance of :class:`.FeatureSelectionCrossValidationTask` class.
	
	:param estimator: It can be either a scikit-learn estimator (e.g., LogisticRegression) or a scikit-learn pipeline ending with an estimator. The estimator needs to be able to predict probabilities through a ``predict_proba`` method.
    :type estimator: scikit-learn {estimator, pipeline}
	
	:param train_data_set_file_path: The file path of the training data set. It can be either a URL or a disk file path.
    :type train_data_set_file_path: str
	
	:param estimator_params: The parameters to override in the provided estimator. It can be either a URL or a disk file path. It defaults to None.
    :type estimator_params: dict, optional
	
	:param field_delimiter: The separation delimiter (comma for CSV, tab for TSV, etc.) used in the input data set file. It defaults to ','.
    :type field_delimiter: str, optional
	
	:param preprocessor : The preprocessor you want to run on the data. You can set for example a ColumnTransformer or OneHotEncoder. It defaults to None.
	:type preprocessor : scikit-learn {preprocessor}, optional
	
	:param preprocessor_params : the parameters of preprocessor. It defaults to None.
	:type preprocessor_params: dict, optional
	
	:param min_features_to_select : The minimum number of features to be selected. This number of features will always be scored. It defaults to 1.
	:type min_features_to_select : int, optional
	
	:param scoring : A scorer callable object or a string with the name of the scorer. It defaults to 'f1'
	:type scoring :	str, optional
	
	:param feature_columns: Either 'all' to use from the input data set file all the columns or a list of column names to select specific columns. It defaults to 'all'.
    :type feature_columns: {str, list}, optional
	
	:param id_column: The name of the column in the input data set file containing the sample IDs. It defaults to 'id'.
    :type id_column: str, optional
	
	:param label_column: The name of the column in the input data set file containing the ground truth labels. It defaults to 'label'.
    :type label_column: str, optional
	
	:param random_seed: The random seed used in the random number generator. It can be used to reproduce the output. It defaults to 42.
    :type random_seed: int, optional
	
	:param verbose : Controls verbosity of output. It defaults to 3.
	:type verbose : int, optional 
	
	:param n_jobs : Number of cores to run in parallel while fitting across folds. It defaults to 1.
	:type n_jobs : int, optional
	"""
	super(FeatureSelectionCrossValidationTask, self).__init__(FeatureSelectionCrossValidationTask.__name__, locals())

  def run(self, output_directory):
    """
    A method for running the task.
    
    :param output_directory: The output directory path under which task-related generated files are stored.
    :type output_directory: str
    
    :return: The task's results. Specifically, the features that has been selected from the task.
    :rtype: {pandas.DataFrame}
  	"""
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