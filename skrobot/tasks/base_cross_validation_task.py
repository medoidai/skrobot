import pandas as pd

from sklearn.model_selection import StratifiedKFold

from . import BaseTask

class BaseCrossValidationTask(BaseTask):
  def __init__ (self, type_name, args):
    super(BaseCrossValidationTask, self).__init__(type_name, args)

    self.stratified_folds()

  def custom_folds(self, folds_file_path, fold_column='fold'):
    options = self.filter_arguments(locals())

    self.update_arguments({ 'fold_options' : options})

    self.update_arguments({ 'fold_method': 'custom' })

    return self

  def stratified_folds(self, total_folds=3, shuffle=False):
    options = self.filter_arguments(locals())

    self.update_arguments({ 'fold_options' : options})

    self.update_arguments({ 'fold_method': 'stratified' })

    return self

  def _build_cv_splits (self, X, y):
    if self.fold_method == 'custom':
      folds_data_frame = pd.read_csv(self.fold_options['folds_file_path'], delimiter=self.field_delimiter)

      return self._get_cv_splits(self.train_data_set_data_frame.merge(folds_data_frame, how='inner', on=self.id_column))
    else:
      return StratifiedKFold(n_splits=self.fold_options['total_folds'], shuffle=self.fold_options['shuffle'], random_state=self.random_seed).split(X, y)

  def _get_cv_splits(self, data_set_data_frame_with_folds):
    cv = []

    for fold_id in sorted(data_set_data_frame_with_folds[self.fold_options['fold_column']].unique()):
      train_indexes = data_set_data_frame_with_folds[data_set_data_frame_with_folds[self.fold_options['fold_column']] != fold_id].index.values

      validation_indexes = data_set_data_frame_with_folds[data_set_data_frame_with_folds[self.fold_options['fold_column']] == fold_id].index.values

      cv.append((train_indexes, validation_indexes))

    return cv