import pandas as pd

from sklearn.model_selection import StratifiedKFold

from . import BaseTask

class BaseCrossValidationTask(BaseTask):
  """
  The :class:`.BaseCrossValidationTask` is an abstract base class for implementing tasks that use cross-validation functionality.

  It can support both stratified k-fold cross-validation as well as cross-validation with user-defined folds.

  By default, stratified k-fold cross-validation is used with the default parameters of :meth:`.stratified_folds` method.
  """
  def __init__ (self, type_name, args):
    """
    This is the constructor method and can be used from child :class:`.BaseCrossValidationTask` implementations.

    :param type_name: The task's type name. A common practice is to pass the name of the task's class.
    :type type_name: str

    :param args: The task's parameters. A common practice is to pass the parameters at the time of task's object creation. It is a dictionary of key-value pairs, where the key is the parameter name and the value is the parameter value.
    :type args: dict
    """

    super(BaseCrossValidationTask, self).__init__(type_name, args)

    self.stratified_folds()

  def custom_folds(self, folds_data, fold_column='fold'):
    """
    Optional method.

    Use cross-validation with user-defined custom folds.

    :param folds_data: The input folds data. It can be either a URL, a disk file path or a pandas DataFrame. The folds data contain the user-defined folds for the samples. If a URL or a disk file path is provided the data must be formatted with the same separation delimiter (comma for CSV, tab for TSV, etc.) as the one used in the input data set files provided to the task. The data must contain two columns and the first row must be the header. The first column is for the sample IDs and needs to be the same as the one used in the input data set files provided to the task. The second column is for the fold IDs (e.g., 1 through 5, A through D, etc.).
    :type folds_data: {str or pandas DataFrame}

    :param fold_column: The column name for the fold IDs. It defaults to 'fold'.
    :type fold_column: str, optional

    :return: The object instance itself.
    :rtype: :class:`.BaseCrossValidationTask`
    """

    options = self._filter_arguments(locals())

    self._update_arguments({ 'fold_options' : options})

    self._update_arguments({ 'fold_method': 'custom' })

    return self

  def stratified_folds(self, total_folds=3, shuffle=False):
    """
    Optional method.

    Use stratified k-fold cross-validation.

    The folds are made by preserving the percentage of samples for each class.

    :param total_folds: Number of folds. Must be at least 2. It defaults to 3.
    :type total_folds: int, optional

    :param shuffle: Whether to shuffle each class's samples before splitting into batches. Note that the samples within each split will not be shuffled. It defaults to False.
    :type shuffle: bool, optional

    :return: The object instance itself.
    :rtype: :class:`.BaseCrossValidationTask`
    """

    options = self._filter_arguments(locals())

    self._update_arguments({ 'fold_options' : options})

    self._update_arguments({ 'fold_method': 'stratified' })

    return self

  def _build_cv_splits (self, X, y):
    if self.fold_method == 'custom':
      folds_data = self.fold_options['folds_data']

      if isinstance(folds_data, str):
        folds_data_frame = pd.read_csv(folds_data, delimiter=self.field_delimiter)
      else:
        folds_data_frame = folds_data.copy()

        folds_data_frame.reset_index(inplace=True, drop=True)

      return self._get_cv_splits(self.train_data_set_data_frame.merge(folds_data_frame, how='inner', on=self.id_column))
    else:
      return StratifiedKFold(n_splits=self.fold_options['total_folds'], shuffle=self.fold_options['shuffle'], random_state=self.random_seed if self.fold_options['shuffle'] else None).split(X, y)

  def _get_cv_splits(self, data_set_data_frame_with_folds):
    cv = []

    for fold_id in sorted(data_set_data_frame_with_folds[self.fold_options['fold_column']].unique()):
      train_indexes = data_set_data_frame_with_folds[data_set_data_frame_with_folds[self.fold_options['fold_column']] != fold_id].index.values

      validation_indexes = data_set_data_frame_with_folds[data_set_data_frame_with_folds[self.fold_options['fold_column']] == fold_id].index.values

      cv.append((train_indexes, validation_indexes))

    return cv