from os import path

from sklearn.linear_model import LogisticRegression

from skrobot.core import Experiment
from skrobot.tasks import EvaluationCrossValidationTask

######### Initialization Code

random_seed = 42

lr_estimator = LogisticRegression(solver='liblinear', random_state=random_seed)

######### skrobot Code

# Build an Experiment
experiment = Experiment('experiments-output').set_source_code_file_path(__file__).set_experimenter('echatzikyriakidis').build()

# Run Evaluation Task
results = experiment.run(EvaluationCrossValidationTask(estimator=lr_estimator,
                                                       train_data_set_file_path=path.join('data','money-laundering-data-train.csv'),
                                                       test_data_set_file_path=path.join('data','money-laundering-data-test.csv'),
                                                       export_classification_reports=True,
                                                       export_confusion_matrixes=True,
                                                       export_pr_curves=True,
                                                       export_roc_curves=True,
                                                       export_false_positives_reports=True,
                                                       export_false_negatives_reports=True,
                                                       export_also_for_train_folds=True,
                                                       random_seed=random_seed).custom_folds(folds_file_path=path.join('data','money-laundering-folds.csv')))

# Print in-memory results
print(results['threshold'])
print(results['cv_threshold_metrics'])
print(results['cv_splits_threshold_metrics'])
print(results['cv_splits_threshold_metrics_summary'])
print(results['test_threshold_metrics'])