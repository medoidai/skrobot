import os, copy

import pandas as pd

import numpy as np

from functools import reduce

from plotly.offline import plot

from plotly import express

from sand.base_cross_validation_ml_task import BaseCrossValidationMlTask

class EvaluateCrossValidationMlTask(BaseCrossValidationMlTask):
  def __init__ (self, estimator, data_set_file_path, estimator_params=None, field_delimiter=',', feature_columns='all', id_column='id', label_column='label', random_seed=123456789, threshold='best', threshold_tuning_range=(0.01, 1.0, 0.01), export_classification_reports=False, export_confusion_matrixes=False, export_roc_curves=False, export_pr_curves=False, export_false_positives_reports=False, export_false_negatives_reports=False, export_also_for_train_folds=False, fscore_beta=1):
    super(EvaluateCrossValidationMlTask, self).__init__(EvaluateCrossValidationMlTask.__name__, locals())

    pd.set_option('display.max_colwidth', -1)

    self.fscore_beta_text = f'f{str(self.fscore_beta)}'

    self.train_text = 'train'

    self.test_text = 'test'

  def run(self, output_directory):
    np.random.seed(self.random_seed)

    self.data_set_data_frame = pd.read_csv(self.data_set_file_path, delimiter=self.field_delimiter)

    ids = self.data_set_data_frame[self.id_column]

    y = self.data_set_data_frame[self.label_column]

    X = self.data_set_data_frame.drop(columns=[self.label_column, self.id_column])

    if self.feature_columns != 'all':
      X = X[self.feature_columns]

    labels = sorted(y.unique())

    splits_threshold_metrics = self.get_splits_threshold_metrics(X, y, self._build_cv_splits(X, y))

    splits_threshold_metrics.to_html(os.path.join(output_directory, 'splits_threshold_metrics.html'), index=False)

    splits_threshold_metrics_summary = self.get_splits_threshold_metrics_summary(splits_threshold_metrics)

    splits_threshold_metrics_summary.to_html(os.path.join(output_directory, 'splits_threshold_metrics_summary.html'), index=False)

    self.threshold, threshold_metrics = self.get_threshold_and_its_metrics(splits_threshold_metrics_summary)

    splits_y_and_y_hat_test, splits_y_and_y_hat_train = self.get_splits_y_and_y_hat_results(X, y, ids, self._build_cv_splits(X, y))

    if self.export_pr_curves: self.build_all_precision_recall_curves(splits_threshold_metrics, splits_threshold_metrics_summary, self.create_directory_path(output_directory, 'pr_curves'))
    if self.export_roc_curves: self.build_all_roc_curves(splits_threshold_metrics, splits_threshold_metrics_summary, self.create_directory_path(output_directory, 'roc_curves'))
    if self.export_confusion_matrixes: self.build_all_confusion_matrixes(splits_y_and_y_hat_test, splits_y_and_y_hat_train, labels, self.create_directory_path(output_directory, 'confusion_matrixes'))
    if self.export_classification_reports: self.build_all_classification_reports(splits_y_and_y_hat_test, splits_y_and_y_hat_train, labels, self.create_directory_path(output_directory, 'classification_reports'))
    if self.export_false_positives_reports: self.build_all_false_positives_reports(splits_y_and_y_hat_test, splits_y_and_y_hat_train, self.create_directory_path(output_directory, 'false_positives_reports'))
    if self.export_false_negatives_reports: self.build_all_false_negatives_reports(splits_y_and_y_hat_test, splits_y_and_y_hat_train, self.create_directory_path(output_directory, 'false_negatives_reports'))

    return { 'threshold' : self.threshold, 'threshold_metrics' : threshold_metrics, 'splits_threshold_metrics' : splits_threshold_metrics, 'splits_threshold_metrics_summary' : splits_threshold_metrics_summary }

  def get_threshold_and_its_metrics (self, splits_threshold_metrics_summary):
    if self.threshold == 'best':
      row = splits_threshold_metrics_summary.loc[splits_threshold_metrics_summary[f'test_{self.fscore_beta_text}_mean'].idxmax()]
    else:
      row = splits_threshold_metrics_summary.loc[splits_threshold_metrics_summary['threshold'] == self.threshold].squeeze()

    if row.empty: raise Exception(f"The specified threshold [{self.threshold}] cannot be found!")

    return row['threshold'], row.drop('threshold')

  def build_all_false_negatives_reports(self, splits_y_and_y_hat_test, splits_y_and_y_hat_train, output_directory):
    self.build_false_negatives_reports(splits_y_and_y_hat_test, output_directory, self.test_text)
    if self.export_also_for_train_folds: self.build_false_negatives_reports(splits_y_and_y_hat_train, output_directory, self.train_text)

  def build_false_negatives_reports(self, splits_y_and_y_hat, output_directory, data_split):
    for split_index, (y, y_hat, ids) in enumerate(splits_y_and_y_hat):
      self.build_false_negatives_report(y, y_hat, ids, output_directory, f'for split {split_index} {data_split} data', f'split_{split_index}_{data_split}_data')

  def build_false_negatives_report(self, y, y_hat, ids, output_directory, title_part_text, file_part_text):
    data = np.vstack((y, y_hat, ids)).T

    false_negatives = data[(data[:,0] == 1) & (data[:,1] == 0)][:,2]

    if false_negatives.size != 0:
      with open(os.path.join(output_directory, f'false_negatives_report_{file_part_text}.txt'), "w") as f: f.write(f"False negatives report {title_part_text} (Threshold = {self.threshold})" + '\n'*2 + '\n'.join(map(str, false_negatives)))

  def build_all_false_positives_reports(self, splits_y_and_y_hat_test, splits_y_and_y_hat_train, output_directory):
    self.build_false_positives_reports(splits_y_and_y_hat_test, output_directory, self.test_text)
    if self.export_also_for_train_folds: self.build_false_positives_reports(splits_y_and_y_hat_train, output_directory, self.train_text)

  def build_false_positives_reports(self, splits_y_and_y_hat, output_directory, data_split):
    for split_index, (y, y_hat, ids) in enumerate(splits_y_and_y_hat):
      self.build_false_positives_report(y, y_hat, ids, output_directory, f'for split {split_index} {data_split} data', f'split_{split_index}_{data_split}_data')

  def build_false_positives_report(self, y, y_hat, ids, output_directory, title_part_text, file_part_text):
    data = np.vstack((y, y_hat, ids)).T

    false_positives = data[(data[:,0] == 0) & (data[:,1] == 1)][:,2]

    if false_positives.size != 0:
      with open(os.path.join(output_directory, f'false_positives_report_{file_part_text}.txt'), "w") as f: f.write(f"False positives report {title_part_text} (Threshold = {self.threshold})" + '\n'*2 + '\n'.join(map(str, false_positives)))

  def build_all_confusion_matrixes(self, splits_y_and_y_hat_test, splits_y_and_y_hat_train, labels, output_directory):
    self.build_confusion_matrixes(splits_y_and_y_hat_test, output_directory, self.test_text, labels)
    if self.export_also_for_train_folds: self.build_confusion_matrixes(splits_y_and_y_hat_train, output_directory, self.train_text, labels)

  def build_confusion_matrixes(self, splits_y_and_y_hat, output_directory, data_split, labels):
    for split_index, (y, y_hat, _) in enumerate(splits_y_and_y_hat):
      self.build_confusion_matrix(y, y_hat, output_directory, f'for split {split_index} {data_split} data', f'split_{split_index}_{data_split}_data', labels)

    y = reduce(lambda x, y: np.concatenate((x, y)), map(lambda x: x[0], splits_y_and_y_hat))
    y_hat = reduce(lambda x, y: np.concatenate((x, y)), map(lambda x: x[1], splits_y_and_y_hat))

    self.build_confusion_matrix(y, y_hat, output_directory, f'for all {data_split} data', f'all_{data_split}_data', labels)

  def build_confusion_matrix(self, y, y_hat, output_directory, title_part_text, file_part_text, labels):
    import matplotlib.pyplot as plt

    from scikitplot.metrics import plot_confusion_matrix

    plot_confusion_matrix(y, y_hat, labels=labels, title=f'Confusion matrix {title_part_text} (Threshold = {self.threshold})')

    plt.savefig(os.path.join(output_directory, f'confusion_matrix_{file_part_text}.png'))

    plt.close()

  def build_all_classification_reports(self, splits_y_and_y_hat_test, splits_y_and_y_hat_train, labels, output_directory):
    self.build_classification_reports(splits_y_and_y_hat_test, output_directory, self.test_text, labels)
    if self.export_also_for_train_folds: self.build_classification_reports(splits_y_and_y_hat_train, output_directory, self.train_text, labels)

  def build_classification_reports(self, splits_y_and_y_hat, output_directory, data_split, labels):
    truncate_precision = 3

    for split_index, (y, y_hat, _) in enumerate(splits_y_and_y_hat):
      self.build_classification_report(y, y_hat, output_directory, f'for split {split_index} {data_split} data', f'split_{split_index}_{data_split}_data', labels, truncate_precision)

    y = reduce(lambda x, y: np.concatenate((x, y)), map(lambda x: x[0], splits_y_and_y_hat))
    y_hat = reduce(lambda x, y: np.concatenate((x, y)), map(lambda x: x[1], splits_y_and_y_hat))

    self.build_classification_report(y, y_hat, output_directory, f'for all {data_split} data', f'all_{data_split}_data', labels, truncate_precision)

  def build_classification_report(self, y, y_hat, output_directory, title_part_text, file_part_text, labels, truncate_precision):
    from math import floor

    from sklearn.metrics import precision_recall_fscore_support

    summary_metrics = list(precision_recall_fscore_support(y_true=y, y_pred=y_hat, labels=labels, beta=self.fscore_beta))

    average_metrics = list(precision_recall_fscore_support(y_true=y, y_pred=y_hat, labels=labels, beta=self.fscore_beta, average='weighted'))

    classification_report = pd.DataFrame(summary_metrics, index=['precision', 'recall', self.fscore_beta_text, 'support'])

    support = classification_report.loc['support']

    total = support.sum()

    average_metrics[-1] = total

    classification_report['avg / total'] = average_metrics

    classification_report = classification_report.T

    classification_report = classification_report.applymap(lambda o: floor(o * 10 ** truncate_precision) / 10 ** truncate_precision)

    classification_report['support'] = classification_report['support'].map(int)

    with open(os.path.join(output_directory, f'classification_report_{file_part_text}.txt'), "w") as f: f.write(f"Classification report {title_part_text} (Threshold = {self.threshold})" + '\n'*2 + f"{classification_report}")

  def build_all_precision_recall_curves(self, splits_threshold_metrics, splits_threshold_metrics_summary, output_directory):
    self.build_cv_precision_recall_curves(splits_threshold_metrics[['threshold', 'cv_split', 'test_precision', 'test_recall', f'test_{self.fscore_beta_text}']], output_directory, self.test_text)
    if self.export_also_for_train_folds: self.build_cv_precision_recall_curves(splits_threshold_metrics[['threshold', 'cv_split', 'train_precision', 'train_recall', f'train_{self.fscore_beta_text}']], output_directory, self.train_text)

    self.build_average_precision_recall_curve(splits_threshold_metrics_summary[['threshold', 'test_precision_mean', 'test_recall_mean', f'test_{self.fscore_beta_text}_mean','test_precision_std', 'test_recall_std', f'test_{self.fscore_beta_text}_std']], output_directory, self.test_text)
    if self.export_also_for_train_folds: self.build_average_precision_recall_curve(splits_threshold_metrics_summary[['threshold', 'train_precision_mean', 'train_recall_mean', f'train_{self.fscore_beta_text}_mean', 'train_precision_std', 'train_recall_std', f'train_{self.fscore_beta_text}_std']], output_directory, self.train_text)

  def build_all_roc_curves(self, splits_threshold_metrics, splits_threshold_metrics_summary, output_directory):
    self.build_cv_roc_curves(splits_threshold_metrics[['threshold', 'cv_split', 'test_true_positive_rate', 'test_false_positive_rate']], output_directory, self.test_text)
    if self.export_also_for_train_folds: self.build_cv_roc_curves(splits_threshold_metrics[['threshold', 'cv_split', 'train_true_positive_rate', 'train_false_positive_rate']], output_directory, self.train_text)

    self.build_average_roc_curve(splits_threshold_metrics_summary[['threshold', 'test_true_positive_rate_mean', 'test_false_positive_rate_mean', 'test_true_positive_rate_std', 'test_false_positive_rate_std']], output_directory, self.test_text)
    if self.export_also_for_train_folds: self.build_average_roc_curve(splits_threshold_metrics_summary[['threshold', 'train_true_positive_rate_mean', 'train_false_positive_rate_mean', 'train_true_positive_rate_std', 'train_false_positive_rate_std']], output_directory, self.train_text)

  def build_average_precision_recall_curve(self, data_frame, output_directory, data_split):
    data_frame.columns = data_frame.columns.str.replace(f'^{data_split}_', '')

    fig = express.line(data_frame, x="recall_mean", y="precision_mean", hover_data=['threshold', f'{self.fscore_beta_text}_mean', 'precision_std', 'recall_std', f'{self.fscore_beta_text}_std'], title=f"Mean Precision-Recall curve from cross validation (CV) splits in {data_split} data")

    self.change_styles_in_figure(fig, x_title="Recall (mean)", y_title="Precision (mean)")

    plot(fig, filename=os.path.join(output_directory, f'mean_pr_curve_{data_split}_data.html'), auto_open=False)

  def build_average_roc_curve(self, data_frame, output_directory, data_split):
    data_frame.columns = data_frame.columns.str.replace(f'^{data_split}_', '')

    fig = express.line(data_frame, x="false_positive_rate_mean", y="true_positive_rate_mean", hover_data=['threshold', 'false_positive_rate_std', 'true_positive_rate_std'], title=f"Mean ROC curve from cross validation (CV) splits in {data_split} data")

    self.change_styles_in_figure(fig, x_title="False Positive Rate (mean)", y_title="True Positive Rate (mean)")

    plot(fig, filename=os.path.join(output_directory, f'mean_roc_curve_{data_split}_data.html'), auto_open=False)

  def build_cv_precision_recall_curves(self, data_frame, output_directory, data_split):
    data_frame.columns = data_frame.columns.str.replace(f'^{data_split}_', '')

    fig = express.line(data_frame, x="recall", y="precision", hover_data=['threshold', self.fscore_beta_text], color="cv_split", title=f"Precision-Recall curves from cross validation (CV) splits in {data_split} data")

    self.change_styles_in_figure(fig, x_title="Recall", y_title="Precision")

    plot(fig, filename=os.path.join(output_directory, f'pr_curves_{data_split}_data.html'), auto_open=False)

  def build_cv_roc_curves(self, data_frame, output_directory, data_split):
    data_frame.columns = data_frame.columns.str.replace(f'^{data_split}_', '')

    fig = express.line(data_frame, x="false_positive_rate", y="true_positive_rate", hover_data=['threshold'], color="cv_split", title=f"ROC curves from cross validation (CV) splits in {data_split} data")

    self.change_styles_in_figure(fig, x_title="False Positive Rate", y_title="True Positive Rate")

    plot(fig, filename=os.path.join(output_directory, f'roc_curves_{data_split}_data.html'), auto_open=False)

  def change_styles_in_figure (self, figure, x_title, y_title):
    figure.update_traces(mode='lines+markers')

    figure.update_layout(
      plot_bgcolor = 'rgba(0,0,0,0)',
      xaxis_title = x_title,
      yaxis_title = y_title,
      xaxis_gridcolor = 'rgb(159, 197, 232)',
      yaxis_gridcolor = 'rgb(159, 197, 232)'
    )

  def create_directory_path(self, output_directory, directory_name):
    directory_path = os.path.join(output_directory, directory_name)

    os.makedirs(directory_path, exist_ok=True)

    return directory_path

  def calculate_y_hat_for_threshold (self, estimator, X, threshold):
    y_proba = estimator.predict_proba(X)

    y_hat = y_proba[:, 1] >= threshold

    return y_hat.astype(int)

  def build_estimator (self):
    estimator = copy.deepcopy(self.estimator)

    if self.estimator_params: estimator.set_params(**self.estimator_params)

    return estimator

  def calculate_diagnostic_performance (self, y, y_hat, data_split):
    y_and_y_hat = np.vstack((y, y_hat)).T

    y_positives = y_and_y_hat[:, 0] == 1
    y_negatives = y_and_y_hat[:, 0] == 0

    y_hat_positives = y_and_y_hat[:, 1] == 1
    y_hat_negatives = y_and_y_hat[:, 1] == 0

    true_positives = y_positives & y_hat_positives
    true_negatives = y_negatives & y_hat_negatives

    sensitivity = np.sum(true_positives) / np.sum(y_positives)
    specificity = np.sum(true_negatives) / np.sum(y_negatives)

    true_positive_rate = sensitivity
    false_positive_rate = 1 - specificity

    precision = np.sum(true_positives) / np.sum(y_hat_positives)
    recall = sensitivity
    fbeta_score = self.calculate_fbeta_score(precision, recall, self.fscore_beta)

    performance = {}

    performance[f'{data_split}_sensitivity'] = sensitivity
    performance[f'{data_split}_specificity'] = specificity

    performance[f'{data_split}_true_positive_rate'] = true_positive_rate
    performance[f'{data_split}_false_positive_rate'] = false_positive_rate

    performance[f'{data_split}_precision'] = precision
    performance[f'{data_split}_recall'] = recall
    performance[f'{data_split}_{self.fscore_beta_text}'] = fbeta_score

    return performance

  def calculate_fbeta_score (self, precision, recall, beta):
    bb = beta**2.0

    return (1 + bb) * (precision * recall) / (bb * precision + recall)

  def get_splits_threshold_metrics_summary(self, dataframe):
    rows_with_nulls = dataframe[dataframe.isnull().any(axis=1)]

    threshold_values_to_remove = rows_with_nulls['threshold'].unique()

    rows_without_nulls = dataframe.loc[~dataframe['threshold'].isin(threshold_values_to_remove)]

    if rows_without_nulls.empty: raise Exception("The summary of threshold metrics cannot be generated!")

    rows_without_nulls = rows_without_nulls.drop(columns='cv_split')

    summary = rows_without_nulls.groupby('threshold').mean().add_suffix('_mean').merge(rows_without_nulls.groupby('threshold').std().add_suffix('_std'), left_index=True, right_index=True)

    return summary.reset_index()

  def get_splits_threshold_metrics_data_frame_columns(self):
    columns = [
      'threshold',
      'cv_split',

      f'{self.test_text}_sensitivity',
      f'{self.test_text}_specificity',
      f'{self.test_text}_true_positive_rate',
      f'{self.test_text}_false_positive_rate',
      f'{self.test_text}_precision',
      f'{self.test_text}_recall',
      f'{self.test_text}_{self.fscore_beta_text}'
    ]

    if self.export_also_for_train_folds:
      columns.extend([
        f'{self.train_text}_sensitivity',
        f'{self.train_text}_specificity',
        f'{self.train_text}_true_positive_rate',
        f'{self.train_text}_false_positive_rate',
        f'{self.train_text}_precision',
        f'{self.train_text}_recall',
        f'{self.train_text}_{self.fscore_beta_text}'
      ])

    return columns

  def get_split_threshold_metrics (self, estimator, split_id, X_test, y_test, X_train, y_train):
    threshold_rows = []

    for threshold in np.arange (self.threshold_tuning_range[0], self.threshold_tuning_range[1], self.threshold_tuning_range[2]):
      row = [ threshold, split_id ]

      row.extend(self.calculate_diagnostic_performance(y_test, self.calculate_y_hat_for_threshold(estimator, X_test, threshold), self.test_text).values())
      if self.export_also_for_train_folds: row.extend(self.calculate_diagnostic_performance(y_train, self.calculate_y_hat_for_threshold(estimator, X_train, threshold), self.train_text).values())

      threshold_rows.append(row)

    return pd.DataFrame(threshold_rows, columns=self.get_splits_threshold_metrics_data_frame_columns())

  def get_splits_threshold_metrics (self, X, y, cv):
    splits_threshold_metrics = []

    for split_id, (train_indexes, test_indexes) in enumerate(cv):
      X_train, y_train = X.loc[train_indexes], y.loc[train_indexes]
      X_test, y_test = X.loc[test_indexes], y.loc[test_indexes]

      estimator = self.build_estimator()
      estimator.fit(X_train, y_train)

      splits_threshold_metrics.append(self.get_split_threshold_metrics(estimator, split_id, X_test, y_test, X_train, y_train))

    splits_threshold_metrics = pd.concat(splits_threshold_metrics).reset_index(drop=True)

    return splits_threshold_metrics

  def get_splits_y_and_y_hat_results (self, X, y, ids, cv):
    splits_y_and_y_hat_test = []
    splits_y_and_y_hat_train = []

    for split_id, (train_indexes, test_indexes) in enumerate(cv):
      X_train, y_train = X.loc[train_indexes], y.loc[train_indexes]
      X_test, y_test = X.loc[test_indexes], y.loc[test_indexes]

      ids_test = ids.loc[test_indexes]
      ids_train = ids.loc[train_indexes]

      estimator = self.build_estimator()
      estimator.fit(X_train, y_train)

      splits_y_and_y_hat_test.append((y_test, self.calculate_y_hat_for_threshold(estimator, X_test, self.threshold), ids_test))
      if self.export_also_for_train_folds: splits_y_and_y_hat_train.append((y_train, self.calculate_y_hat_for_threshold(estimator, X_train, self.threshold), ids_train))

    return splits_y_and_y_hat_test, splits_y_and_y_hat_train