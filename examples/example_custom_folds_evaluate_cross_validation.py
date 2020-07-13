from os import path

from sklearn.linear_model import LogisticRegression

from sand.core import Experiment
from sand.tasks import EvaluateCrossValidationTask

######### Scikit-learn Code

data_set_file_path = path.join('data','dataset-1.csv')

random_seed = 42

lr_estimator = LogisticRegression(solver='liblinear', random_state=random_seed)

######### Sand Code

# Build an Experiment
experiment = Experiment('output', __file__).set_experimenter('echatzikyriakidis').build()

# Run Evaluation Task
results = experiment.run(EvaluateCrossValidationTask(estimator=lr_estimator,
                                                     train_data_set_file_path=data_set_file_path,
                                                     test_data_set_file_path=data_set_file_path,
                                                     export_classification_reports=True,
                                                     export_confusion_matrixes=True,
                                                     export_pr_curves=True,
                                                     export_roc_curves=True,
                                                     export_false_positives_reports=True,
                                                     export_false_negatives_reports=True,
                                                     export_also_for_train_folds=True,
                                                     random_seed=random_seed).custom_folds(folds_file_path=path.join('data','folds-1.csv')))

# Print in-memory results
print(results['threshold'])
print(results['cv_threshold_metrics'])
print(results['cv_splits_threshold_metrics'])
print(results['cv_splits_threshold_metrics_summary'])
print(results['test_threshold_metrics'])