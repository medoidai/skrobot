import os, copy

import pandas as pd

import numpy as np

from math import floor

from functools import reduce

from plotly.offline import plot

from plotly import express

from . import BaseCrossValidationTask

class EvaluationCrossValidationTask(BaseCrossValidationTask):
  """
  The :class:`.EvaluationCrossValidationTask` class can be used to evaluate a scikit-learn estimator/pipeline on some data.

  The following evaluation results can be generated on-demand for hold-out test data set as well as train/validation cross-validation folds:

  * PR / ROC Curves
  * Confusion Matrixes
  * Classification Reports
  * Performance Metrics
  * False Positives
  * False Negatives

  It can support both stratified k-fold cross-validation as well as cross-validation with user-defined folds.

  By default, stratified k-fold cross-validation is used with the default parameters of :meth:`.stratified_folds` method.
  """
  def __init__ (self, estimator, train_data_set, test_data_set=None, estimator_params=None, field_delimiter=',', feature_columns='all', id_column='id', label_column='label', random_seed=42, threshold_selection_by='f1', metric_greater_is_better=True, threshold_tuning_range=(0.01, 1.0, 0.01), export_classification_reports=False, export_confusion_matrixes=False, export_roc_curves=False, export_pr_curves=False, export_false_positives_reports=False, export_false_negatives_reports=False, export_also_for_train_folds=False, fscore_beta=1):
    """
    This is the constructor method and can be used to create a new object instance of :class:`.EvaluationCrossValidationTask` class.

    :param estimator: It can be either an estimator (e.g., LogisticRegression) or a pipeline ending with an estimator. The estimator needs to be able to predict probabilities through a ``predict_proba`` method.
    :type estimator: scikit-learn {estimator, pipeline}

    :param train_data_set: The input train data set. It can be either a URL, a disk file path or a pandas DataFrame.
    :type train_data_set: {str or pandas DataFrame}

    :param test_data_set: The input test data set. It can be either a URL, a disk file path or a pandas DataFrame. It defaults to None.
    :type test_data_set: {str or pandas DataFrame}, optional

    :param estimator_params: The parameters to override in the provided estimator/pipeline. It defaults to None.
    :type estimator_params: dict, optional

    :param field_delimiter: The separation delimiter (comma for CSV, tab for TSV, etc.) used in the input train/test data set files. It defaults to ','.
    :type field_delimiter: str, optional

    :param feature_columns: Either 'all' to use from the input train/test data set files all the columns or a list of column names to select specific columns. It defaults to 'all'.
    :type feature_columns: {str, list}, optional

    :param id_column: The name of the column in the input train/test data set files containing the sample IDs. It defaults to 'id'.
    :type id_column: str, optional

    :param label_column: The name of the column in the input train/test data set files containing the ground truth labels. It defaults to 'label'.
    :type label_column: str, optional

    :param random_seed: The random seed used in the random number generator. It can be used to reproduce the output. It defaults to 42.
    :type random_seed: int, optional

    :param threshold_selection_by: The evaluation results will be generated either for a specific provided threshold value (e.g., 0.49) or for the best threshold found from threshold tuning, based on a specific provided metric (e.g., 'f1', 'f0.55'). It defaults to 'f1'.
    :type threshold_selection_by: {str, float}, optional

    :param metric_greater_is_better: This flag will control the direction of searching of the best threshold and it depends on the provided metric in ``threshold_selection_by``. True, means that greater metric values is better and False means the opposite. It defaults to True.
    :type metric_greater_is_better: bool, optional

    :param threshold_tuning_range: A range in form (start_value, stop_value, step_size) for generating a sequence of threshold values in threshold tuning. It generates the sequence by incrementing the start value using the step size until it reaches the stop value. It defaults to (0.01, 1.0, 0.01).
    :type threshold_tuning_range: tuple, optional

    :param export_classification_reports: If this task will export classification reports. It defaults to False.
    :type export_classification_reports: bool, optional

    :param export_confusion_matrixes:  If this task will export confusion matrixes. It defaults to False.
    :type export_confusion_matrixes: bool, optional

    :param export_roc_curves:  If this task will export ROC curves. It defaults to False.
    :type export_roc_curves: bool, optional

    :param export_pr_curves: If this task will export PR curves. It defaults to False.
    :type export_pr_curves: bool, optional

    :param export_false_positives_reports: If this task will export false positives reports. It defaults to False.
    :type export_false_positives_reports: bool, optional

    :param export_false_negatives_reports: If this task will export false negatives reports. It defaults to False.
    :type export_false_negatives_reports: bool, optional

    :param export_also_for_train_folds: If this task will export the evaluation results also for the train folds of cross-validation. It defaults to False.
    :type export_also_for_train_folds: bool, optional

    :param fscore_beta: The beta parameter in F-measure. It determines the weight of recall in the score. *beta < 1* lends more weight to precision, while *beta > 1* favors recall (*beta -> 0* considers only precision, *beta -> +inf* only recall). It defaults to 1.
    :type fscore_beta: float, optional
    """

    super(EvaluationCrossValidationTask, self).__init__(EvaluationCrossValidationTask.__name__, locals())

    pd.set_option('display.max_colwidth', None)

    self.fscore_beta_text = f'f{str(self.fscore_beta)}'

    self.train_text = 'train'

    self.validation_text = 'validation'

    self.test_text = 'test'

  def run(self, output_directory):
    """
    Run the task.

    All of the evaluation results are stored as files under the output directory path.

    :param output_directory: The output directory path under which task-related generated files are stored.
    :type output_directory: str

    :return: The task's result. Specifically, the threshold used along with its related performance metrics and summary metrics from all cross-validation splits as well as hold-out test data set.
    :rtype: dict
    """

    if isinstance(self.train_data_set, str):
      self.train_data_set_data_frame = pd.read_csv(self.train_data_set, delimiter=self.field_delimiter)
    else:
      self.train_data_set_data_frame = self.train_data_set.copy()

      self.train_data_set_data_frame.reset_index(inplace=True, drop=True)

    train_ids = self.train_data_set_data_frame[self.id_column]
    train_y = self.train_data_set_data_frame[self.label_column]
    train_X = self.train_data_set_data_frame.drop(columns=[self.label_column, self.id_column])
    if self.feature_columns != 'all': train_X = train_X[self.feature_columns]

    labels = sorted(train_y.unique())
    if len(labels) != 2: raise Exception('The unique values for the y target should be exactly two.')

    splits_threshold_metrics = self._get_splits_threshold_metrics(train_X, train_y, self._build_cv_splits(train_X, train_y))
    splits_threshold_metrics.to_html(os.path.join(output_directory, 'cv_splits_threshold_metrics.html'), index=False)
    splits_threshold_metrics_summary = self._get_splits_threshold_metrics_summary(splits_threshold_metrics)
    splits_threshold_metrics_summary.to_html(os.path.join(output_directory, 'cv_splits_threshold_metrics_summary.html'), index=False)

    self.threshold, threshold_metrics = self._get_threshold_and_its_metrics(splits_threshold_metrics_summary)
    splits_y_and_y_hat_validation, splits_y_and_y_hat_train = self._get_splits_y_and_y_hat_results(train_X, train_y, train_ids, self._build_cv_splits(train_X, train_y))

    result = { 'threshold' : self.threshold, 'cv_threshold_metrics' : threshold_metrics, 'cv_splits_threshold_metrics' : splits_threshold_metrics, 'cv_splits_threshold_metrics_summary' : splits_threshold_metrics_summary }

    y_and_y_hat_test = None
    test_threshold_metrics = None

    if self.test_data_set is not None:
      if isinstance(self.test_data_set, str):
        self.test_data_set_data_frame = pd.read_csv(self.test_data_set, delimiter=self.field_delimiter)
      else:
        self.test_data_set_data_frame = self.test_data_set.copy()

        self.test_data_set_data_frame.reset_index(inplace=True, drop=True)

      test_ids = self.test_data_set_data_frame[self.id_column]
      test_y = self.test_data_set_data_frame[self.label_column]
      test_X = self.test_data_set_data_frame.drop(columns=[self.label_column, self.id_column])
      if self.feature_columns != 'all': test_X = test_X[self.feature_columns]

      np.random.seed(self.random_seed)
      estimator = self._build_estimator()
      estimator.fit(train_X, train_y)

      test_threshold_metrics = self._get_threshold_metrics(estimator, test_X, test_y, self.test_text)
      test_threshold_metrics.to_html(os.path.join(output_directory, 'test_threshold_metrics.html'), index=False)
      y_and_y_hat_test = self._get_y_and_y_hat_results(estimator, test_X, test_y, test_ids)

      result.update({'test_threshold_metrics' : test_threshold_metrics})

    if self.export_pr_curves: self._build_all_precision_recall_curves(splits_threshold_metrics, splits_threshold_metrics_summary, self._create_directory_path(output_directory, 'pr_curves'), test_threshold_metrics)
    if self.export_roc_curves: self._build_all_roc_curves(splits_threshold_metrics, splits_threshold_metrics_summary, self._create_directory_path(output_directory, 'roc_curves'), test_threshold_metrics)
    if self.export_confusion_matrixes: self._build_all_confusion_matrixes(splits_y_and_y_hat_validation, splits_y_and_y_hat_train, labels, self._create_directory_path(output_directory, 'confusion_matrixes'), y_and_y_hat_test)
    if self.export_classification_reports: self._build_all_classification_reports(splits_y_and_y_hat_validation, splits_y_and_y_hat_train, labels, self._create_directory_path(output_directory, 'classification_reports'), y_and_y_hat_test)
    if self.export_false_positives_reports: self._build_all_false_positives_reports(splits_y_and_y_hat_validation, splits_y_and_y_hat_train, self._create_directory_path(output_directory, 'false_positives_reports'), y_and_y_hat_test)
    if self.export_false_negatives_reports: self._build_all_false_negatives_reports(splits_y_and_y_hat_validation, splits_y_and_y_hat_train, self._create_directory_path(output_directory, 'false_negatives_reports'), y_and_y_hat_test)

    return result

  def _get_threshold_and_its_metrics (self, splits_threshold_metrics_summary):
    if isinstance(self.threshold_selection_by, float):
      row = splits_threshold_metrics_summary.loc[splits_threshold_metrics_summary['threshold'] == self.threshold_selection_by].squeeze()
    elif isinstance(self.threshold_selection_by, str):
      mean_column = f'validation_{self.threshold_selection_by}_mean'

      std_column = f'validation_{self.threshold_selection_by}_std'

      if mean_column not in splits_threshold_metrics_summary.columns or std_column not in splits_threshold_metrics_summary.columns:
        raise Exception(f"The specified metric [{self.threshold_selection_by}] for threshold selection is invalid.")

      row = splits_threshold_metrics_summary.sort_values([mean_column, std_column], ascending = (not self.metric_greater_is_better, True)).head(1).squeeze()
    else:
      raise Exception(f"The specified threshold selection criteria [{self.threshold_selection_by}] is invalid.")

    if row.empty: raise Exception(f"For the specified threshold selection criteria [{self.threshold_selection_by}] no threshold can be selected.")

    return self._truncate_number(row['threshold'], 3), row.drop('threshold')

  def _build_all_false_negatives_reports(self, splits_y_and_y_hat_validation, splits_y_and_y_hat_train, output_directory, y_and_y_hat_test=None):
    self._build_false_negatives_reports(splits_y_and_y_hat_validation, output_directory, self.validation_text)

    if self.export_also_for_train_folds: self._build_false_negatives_reports(splits_y_and_y_hat_train, output_directory, self.train_text)

    if y_and_y_hat_test: self._build_false_negatives_report(y_and_y_hat_test[0], y_and_y_hat_test[1], y_and_y_hat_test[2], output_directory, f'for {self.test_text} data', f'{self.test_text}_data')

  def _build_false_negatives_reports(self, splits_y_and_y_hat, output_directory, data_split):
    for split_index, (y, y_hat, ids) in enumerate(splits_y_and_y_hat):
      self._build_false_negatives_report(y, y_hat, ids, output_directory, f'for CV split {split_index} {data_split} data', f'cv_split_{split_index}_{data_split}_data')

  def _build_false_negatives_report(self, y, y_hat, ids, output_directory, title_part_text, file_part_text):
    data = np.vstack((y, y_hat, ids)).T

    false_negatives = data[(data[:,0] == 1) & (data[:,1] == 0)][:,2]

    if false_negatives.size != 0:
      with open(os.path.join(output_directory, f'false_negatives_report_{file_part_text}.txt'), "w") as f: f.write(f"False negatives report {title_part_text} (Threshold = {self.threshold})" + '\n'*2 + '\n'.join(map(str, false_negatives)))

  def _build_all_false_positives_reports(self, splits_y_and_y_hat_validation, splits_y_and_y_hat_train, output_directory, y_and_y_hat_test=None):
    self._build_false_positives_reports(splits_y_and_y_hat_validation, output_directory, self.validation_text)

    if self.export_also_for_train_folds: self._build_false_positives_reports(splits_y_and_y_hat_train, output_directory, self.train_text)

    if y_and_y_hat_test: self._build_false_positives_report(y_and_y_hat_test[0], y_and_y_hat_test[1], y_and_y_hat_test[2], output_directory, f'for {self.test_text} data', f'{self.test_text}_data')

  def _build_false_positives_reports(self, splits_y_and_y_hat, output_directory, data_split):
    for split_index, (y, y_hat, ids) in enumerate(splits_y_and_y_hat):
      self._build_false_positives_report(y, y_hat, ids, output_directory, f'for CV split {split_index} {data_split} data', f'cv_split_{split_index}_{data_split}_data')

  def _build_false_positives_report(self, y, y_hat, ids, output_directory, title_part_text, file_part_text):
    data = np.vstack((y, y_hat, ids)).T

    false_positives = data[(data[:,0] == 0) & (data[:,1] == 1)][:,2]

    if false_positives.size != 0:
      with open(os.path.join(output_directory, f'false_positives_report_{file_part_text}.txt'), "w") as f: f.write(f"False positives report {title_part_text} (Threshold = {self.threshold})" + '\n'*2 + '\n'.join(map(str, false_positives)))

  def _build_all_confusion_matrixes(self, splits_y_and_y_hat_validation, splits_y_and_y_hat_train, labels, output_directory, y_and_y_hat_test=None):
    self._build_confusion_matrixes(splits_y_and_y_hat_validation, output_directory, self.validation_text, labels)

    if self.export_also_for_train_folds: self._build_confusion_matrixes(splits_y_and_y_hat_train, output_directory, self.train_text, labels)

    if y_and_y_hat_test: self._build_confusion_matrix(y_and_y_hat_test[0], y_and_y_hat_test[1], output_directory, f'for {self.test_text} data', f'{self.test_text}_data', labels)

  def _build_confusion_matrixes(self, splits_y_and_y_hat, output_directory, data_split, labels):
    for split_index, (y, y_hat, _) in enumerate(splits_y_and_y_hat):
      self._build_confusion_matrix(y, y_hat, output_directory, f'for CV split {split_index} {data_split} data', f'cv_split_{split_index}_{data_split}_data', labels)

    y = reduce(lambda x, y: np.concatenate((x, y)), map(lambda x: x[0], splits_y_and_y_hat))
    y_hat = reduce(lambda x, y: np.concatenate((x, y)), map(lambda x: x[1], splits_y_and_y_hat))

    self._build_confusion_matrix(y, y_hat, output_directory, f'for CV {data_split} data', f'cv_{data_split}_data', labels)

  def _build_confusion_matrix(self, y, y_hat, output_directory, title_part_text, file_part_text, labels):
    import matplotlib.pyplot as plt

    from scikitplot.metrics import plot_confusion_matrix

    plot_confusion_matrix(y, y_hat, labels=labels, title=f'Confusion matrix {title_part_text} (Threshold = {self.threshold})')

    plt.savefig(os.path.join(output_directory, f'confusion_matrix_{file_part_text}.png'))

    plt.close()

  def _build_all_classification_reports(self, splits_y_and_y_hat_validation, splits_y_and_y_hat_train, labels, output_directory, y_and_y_hat_test=None):
    truncate_precision = 3

    self._build_classification_reports(splits_y_and_y_hat_validation, output_directory, self.validation_text, labels, truncate_precision)

    if self.export_also_for_train_folds: self._build_classification_reports(splits_y_and_y_hat_train, output_directory, self.train_text, labels, truncate_precision)

    if y_and_y_hat_test: self._build_classification_report(y_and_y_hat_test[0], y_and_y_hat_test[1], output_directory, f'for {self.test_text} data', f'{self.test_text}_data', labels, truncate_precision)

  def _build_classification_reports(self, splits_y_and_y_hat, output_directory, data_split, labels, truncate_precision):
    for split_index, (y, y_hat, _) in enumerate(splits_y_and_y_hat):
      self._build_classification_report(y, y_hat, output_directory, f'for CV split {split_index} {data_split} data', f'cv_split_{split_index}_{data_split}_data', labels, truncate_precision)

    y = reduce(lambda x, y: np.concatenate((x, y)), map(lambda x: x[0], splits_y_and_y_hat))
    y_hat = reduce(lambda x, y: np.concatenate((x, y)), map(lambda x: x[1], splits_y_and_y_hat))

    self._build_classification_report(y, y_hat, output_directory, f'for CV {data_split} data', f'cv_{data_split}_data', labels, truncate_precision)

  def _build_classification_report(self, y, y_hat, output_directory, title_part_text, file_part_text, labels, truncate_precision):
    from functools import partial

    from sklearn.metrics import precision_recall_fscore_support

    summary_metrics = list(precision_recall_fscore_support(y_true=y, y_pred=y_hat, labels=labels, beta=self.fscore_beta))

    average_metrics = list(precision_recall_fscore_support(y_true=y, y_pred=y_hat, labels=labels, beta=self.fscore_beta, average='weighted'))

    classification_report = pd.DataFrame(summary_metrics, index=['precision', 'recall', self.fscore_beta_text, 'support'])

    support = classification_report.loc['support']

    total = support.sum()

    average_metrics[-1] = total

    classification_report['avg / total'] = average_metrics

    classification_report = classification_report.T

    classification_report = classification_report.applymap(partial(self._truncate_number, truncate_precision=truncate_precision))

    classification_report['support'] = classification_report['support'].map(int)

    with open(os.path.join(output_directory, f'classification_report_{file_part_text}.txt'), "w") as f: f.write(f"Classification report {title_part_text} (Threshold = {self.threshold})" + '\n'*2 + f"{classification_report}")

  def _truncate_number(self, number, truncate_precision):
    return floor(number * 10 ** truncate_precision) / 10 ** truncate_precision

  def _build_all_precision_recall_curves(self, splits_threshold_metrics, splits_threshold_metrics_summary, output_directory, test_threshold_metrics=None):
    self._build_cv_precision_recall_curves(splits_threshold_metrics[['threshold', 'cv_split', 'validation_precision', 'validation_recall', f'validation_{self.fscore_beta_text}']], output_directory, self.validation_text)
    if self.export_also_for_train_folds: self._build_cv_precision_recall_curves(splits_threshold_metrics[['threshold', 'cv_split', 'train_precision', 'train_recall', f'train_{self.fscore_beta_text}']], output_directory, self.train_text)

    self._build_average_precision_recall_curve(splits_threshold_metrics_summary[['threshold', 'validation_precision_mean', 'validation_recall_mean', f'validation_{self.fscore_beta_text}_mean','validation_precision_std', 'validation_recall_std', f'validation_{self.fscore_beta_text}_std']], output_directory, self.validation_text)
    if self.export_also_for_train_folds: self._build_average_precision_recall_curve(splits_threshold_metrics_summary[['threshold', 'train_precision_mean', 'train_recall_mean', f'train_{self.fscore_beta_text}_mean', 'train_precision_std', 'train_recall_std', f'train_{self.fscore_beta_text}_std']], output_directory, self.train_text)

    if test_threshold_metrics is not None: self._build_precision_recall_curve(test_threshold_metrics[['threshold', 'test_precision', 'test_recall', f'test_{self.fscore_beta_text}']], output_directory, self.test_text)

  def _build_all_roc_curves(self, splits_threshold_metrics, splits_threshold_metrics_summary, output_directory, test_threshold_metrics=None):
    self._build_cv_roc_curves(splits_threshold_metrics[['threshold', 'cv_split', 'validation_true_positive_rate', 'validation_false_positive_rate']], output_directory, self.validation_text)
    if self.export_also_for_train_folds: self._build_cv_roc_curves(splits_threshold_metrics[['threshold', 'cv_split', 'train_true_positive_rate', 'train_false_positive_rate']], output_directory, self.train_text)

    self._build_average_roc_curve(splits_threshold_metrics_summary[['threshold', 'validation_true_positive_rate_mean', 'validation_false_positive_rate_mean', 'validation_true_positive_rate_std', 'validation_false_positive_rate_std']], output_directory, self.validation_text)
    if self.export_also_for_train_folds: self._build_average_roc_curve(splits_threshold_metrics_summary[['threshold', 'train_true_positive_rate_mean', 'train_false_positive_rate_mean', 'train_true_positive_rate_std', 'train_false_positive_rate_std']], output_directory, self.train_text)

    if test_threshold_metrics is not None: self._build_roc_curve(test_threshold_metrics[['threshold', 'test_true_positive_rate', 'test_false_positive_rate']], output_directory, self.test_text)

  def _build_average_precision_recall_curve(self, data_frame, output_directory, data_split):
    data_frame.columns = data_frame.columns.str.replace(f'^{data_split}_', '')

    fig = express.line(data_frame, x="recall_mean", y="precision_mean", hover_data=['threshold', f'{self.fscore_beta_text}_mean', 'precision_std', 'recall_std', f'{self.fscore_beta_text}_std'], title=f"Mean Precision-Recall curve from cross validation (CV) splits in {data_split} data")

    self._change_styles_in_figure(fig, x_title="Recall (mean)", y_title="Precision (mean)")

    plot(fig, filename=os.path.join(output_directory, f'cv_mean_pr_curve_{data_split}_data.html'), auto_open=False)

  def _build_average_roc_curve(self, data_frame, output_directory, data_split):
    data_frame.columns = data_frame.columns.str.replace(f'^{data_split}_', '')

    fig = express.line(data_frame, x="false_positive_rate_mean", y="true_positive_rate_mean", hover_data=['threshold', 'false_positive_rate_std', 'true_positive_rate_std'], title=f"Mean ROC curve from cross validation (CV) splits in {data_split} data")

    self._change_styles_in_figure(fig, x_title="False Positive Rate (mean)", y_title="True Positive Rate (mean)")

    plot(fig, filename=os.path.join(output_directory, f'cv_mean_roc_curve_{data_split}_data.html'), auto_open=False)

  def _build_cv_precision_recall_curves(self, data_frame, output_directory, data_split):
    data_frame.columns = data_frame.columns.str.replace(f'^{data_split}_', '')

    fig = express.line(data_frame, x="recall", y="precision", hover_data=['threshold', self.fscore_beta_text], color="cv_split", title=f"Precision-Recall curves from cross validation (CV) splits in {data_split} data")

    self._change_styles_in_figure(fig, x_title="Recall", y_title="Precision")

    plot(fig, filename=os.path.join(output_directory, f'cv_pr_curves_{data_split}_data.html'), auto_open=False)

  def _build_cv_roc_curves(self, data_frame, output_directory, data_split):
    data_frame.columns = data_frame.columns.str.replace(f'^{data_split}_', '')

    fig = express.line(data_frame, x="false_positive_rate", y="true_positive_rate", hover_data=['threshold'], color="cv_split", title=f"ROC curves from cross validation (CV) splits in {data_split} data")

    self._change_styles_in_figure(fig, x_title="False Positive Rate", y_title="True Positive Rate")

    plot(fig, filename=os.path.join(output_directory, f'cv_roc_curves_{data_split}_data.html'), auto_open=False)

  def _build_roc_curve(self, data_frame, output_directory, data_split):
    data_frame.columns = data_frame.columns.str.replace(f'^{data_split}_', '')

    fig = express.line(data_frame, x="false_positive_rate", y="true_positive_rate", hover_data=['threshold'], title=f"ROC curve for {data_split} data")

    self._change_styles_in_figure(fig, x_title="False Positive Rate", y_title="True Positive Rate")

    plot(fig, filename=os.path.join(output_directory, f'roc_curve_{data_split}_data.html'), auto_open=False)

  def _build_precision_recall_curve(self, data_frame, output_directory, data_split):
    data_frame.columns = data_frame.columns.str.replace(f'^{data_split}_', '')

    fig = express.line(data_frame, x="recall", y="precision", hover_data=['threshold', self.fscore_beta_text], title=f"Precision-Recall curve for {data_split} data")

    self._change_styles_in_figure(fig, x_title="Recall", y_title="Precision")

    plot(fig, filename=os.path.join(output_directory, f'pr_curve_{data_split}_data.html'), auto_open=False)

  def _change_styles_in_figure (self, figure, x_title, y_title):
    figure.update_traces(mode='lines+markers')

    figure.update_layout(
      plot_bgcolor = 'rgba(0,0,0,0)',
      xaxis_title = x_title,
      yaxis_title = y_title,
      xaxis_gridcolor = 'rgb(159, 197, 232)',
      yaxis_gridcolor = 'rgb(159, 197, 232)'
    )

  def _create_directory_path(self, output_directory, directory_name):
    directory_path = os.path.join(output_directory, directory_name)

    os.makedirs(directory_path, exist_ok=True)

    return directory_path

  def _calculate_y_hat_for_threshold (self, estimator, X, threshold):
    y_proba = estimator.predict_proba(X)

    y_hat = y_proba[:, 1] >= threshold

    return y_hat.astype(int)

  def _build_estimator (self):
    estimator = copy.deepcopy(self.estimator)

    if self.estimator_params: estimator.set_params(**self.estimator_params)

    return estimator

  def _calculate_diagnostic_performance (self, y, y_hat, data_split):
    true_positives = np.sum(np.logical_and(y_hat == 1, y == 1))
    true_negatives = np.sum(np.logical_and(y_hat == 0, y == 0))

    false_positives = np.sum(np.logical_and(y_hat == 1, y == 0))
    false_negatives = np.sum(np.logical_and(y_hat == 0, y == 1))

    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)

    true_positive_rate = true_positives / (true_positives + false_negatives)
    false_positive_rate = false_positives / (false_positives + true_negatives)

    sensitivity = true_positive_rate
    specificity = 1 - false_positive_rate

    geometric_mean = np.sqrt(sensitivity * specificity)

    precision = true_positives / (true_positives + false_positives)
    recall = sensitivity

    fbeta_score = self._calculate_fbeta_score(precision, recall, self.fscore_beta)

    performance = {}

    performance[f'{data_split}_sensitivity'] = sensitivity
    performance[f'{data_split}_specificity'] = specificity

    performance[f'{data_split}_accuracy'] = accuracy

    performance[f'{data_split}_true_positive_rate'] = true_positive_rate
    performance[f'{data_split}_false_positive_rate'] = false_positive_rate
    performance[f'{data_split}_geometric_mean'] = geometric_mean

    performance[f'{data_split}_precision'] = precision
    performance[f'{data_split}_recall'] = recall
    performance[f'{data_split}_{self.fscore_beta_text}'] = fbeta_score

    return performance

  def _calculate_fbeta_score (self, precision, recall, beta):
    bb = beta**2.0

    return (1 + bb) * (precision * recall) / (bb * precision + recall)

  def _get_splits_threshold_metrics_summary(self, dataframe):
    rows_with_nulls = dataframe[dataframe.isnull().any(axis=1)]

    threshold_values_to_remove = rows_with_nulls['threshold'].unique()

    rows_without_nulls = dataframe.loc[~dataframe['threshold'].isin(threshold_values_to_remove)]

    if rows_without_nulls.empty: raise Exception("The summary of threshold metrics cannot be generated.")

    rows_without_nulls = rows_without_nulls.drop(columns='cv_split')

    summary = rows_without_nulls.groupby('threshold').mean().add_suffix('_mean').merge(rows_without_nulls.groupby('threshold').std().add_suffix('_std'), left_index=True, right_index=True)

    return summary.reset_index()

  def _get_threshold_metrics_data_frame_columns(self, data_split):
    return [
      'threshold',

      f'{data_split}_sensitivity',
      f'{data_split}_specificity',
      f'{data_split}_accuracy',
      f'{data_split}_true_positive_rate',
      f'{data_split}_false_positive_rate',
      f'{data_split}_geometric_mean',
      f'{data_split}_precision',
      f'{data_split}_recall',
      f'{data_split}_{self.fscore_beta_text}'
    ]

  def _get_splits_threshold_metrics_data_frame_columns(self):
    columns = [
      'threshold',
      'cv_split',

      f'{self.validation_text}_sensitivity',
      f'{self.validation_text}_specificity',
      f'{self.validation_text}_accuracy',
      f'{self.validation_text}_true_positive_rate',
      f'{self.validation_text}_false_positive_rate',
      f'{self.validation_text}_geometric_mean',
      f'{self.validation_text}_precision',
      f'{self.validation_text}_recall',
      f'{self.validation_text}_{self.fscore_beta_text}'
    ]

    if self.export_also_for_train_folds:
      columns.extend([
        f'{self.train_text}_sensitivity',
        f'{self.train_text}_specificity',
        f'{self.train_text}_accuracy',
        f'{self.train_text}_true_positive_rate',
        f'{self.train_text}_false_positive_rate',
        f'{self.train_text}_geometric_mean',
        f'{self.train_text}_precision',
        f'{self.train_text}_recall',
        f'{self.train_text}_{self.fscore_beta_text}'
      ])

    return columns

  def _get_split_threshold_metrics (self, estimator, split_id, X_validation, y_validation, X_train, y_train):
    threshold_rows = []

    for threshold in np.arange (self.threshold_tuning_range[0], self.threshold_tuning_range[1], self.threshold_tuning_range[2]):
      row = [ threshold, split_id ]

      row.extend(self._calculate_diagnostic_performance(y_validation, self._calculate_y_hat_for_threshold(estimator, X_validation, threshold), self.validation_text).values())
      if self.export_also_for_train_folds: row.extend(self._calculate_diagnostic_performance(y_train, self._calculate_y_hat_for_threshold(estimator, X_train, threshold), self.train_text).values())

      threshold_rows.append(row)

    return pd.DataFrame(threshold_rows, columns=self._get_splits_threshold_metrics_data_frame_columns())

  def _get_splits_threshold_metrics (self, X, y, cv):
    np.random.seed(self.random_seed)

    splits_threshold_metrics = []

    for split_id, (train_indexes, validation_indexes) in enumerate(cv):
      X_train, y_train = X.loc[train_indexes], y.loc[train_indexes]
      X_validation, y_validation = X.loc[validation_indexes], y.loc[validation_indexes]

      estimator = self._build_estimator()
      estimator.fit(X_train, y_train)

      splits_threshold_metrics.append(self._get_split_threshold_metrics(estimator, split_id, X_validation, y_validation, X_train, y_train))

    splits_threshold_metrics = pd.concat(splits_threshold_metrics).reset_index(drop=True)

    return splits_threshold_metrics

  def _get_splits_y_and_y_hat_results (self, X, y, ids, cv):
    np.random.seed(self.random_seed)

    splits_y_and_y_hat_validation = []
    splits_y_and_y_hat_train = []

    for split_id, (train_indexes, validation_indexes) in enumerate(cv):
      X_train, y_train = X.loc[train_indexes], y.loc[train_indexes]
      X_validation, y_validation = X.loc[validation_indexes], y.loc[validation_indexes]

      ids_validation = ids.loc[validation_indexes]
      ids_train = ids.loc[train_indexes]

      estimator = self._build_estimator()
      estimator.fit(X_train, y_train)

      splits_y_and_y_hat_validation.append(self._get_y_and_y_hat_results(estimator, X_validation, y_validation, ids_validation))
      if self.export_also_for_train_folds: splits_y_and_y_hat_train.append(self._get_y_and_y_hat_results(estimator, X_train, y_train, ids_train))

    return splits_y_and_y_hat_validation, splits_y_and_y_hat_train

  def _get_y_and_y_hat_results (self, estimator, X, y, ids):
    return (y, self._calculate_y_hat_for_threshold(estimator, X, self.threshold), ids)

  def _get_threshold_metrics (self, estimator, X, y, data_split):
    threshold_rows = []

    for threshold in np.arange (self.threshold_tuning_range[0], self.threshold_tuning_range[1], self.threshold_tuning_range[2]):
      row = [ threshold ]

      row.extend(self._calculate_diagnostic_performance(y, self._calculate_y_hat_for_threshold(estimator, X, threshold), data_split).values())

      threshold_rows.append(row)

    return pd.DataFrame(threshold_rows, columns=self._get_threshold_metrics_data_frame_columns(data_split))