from os import path

from sand.experiment import Experiment
from sand.evaluate_cross_validation_ml_task import EvaluateCrossValidationMlTask

experiment = Experiment('output', __file__).set_experimenter('echatzikyriakidis').build()

######### Experiment

from sklearn.linear_model import LogisticRegression

random_seed = 42

lr_estimator = LogisticRegression(solver='liblinear', random_state=random_seed)

results = experiment.run(EvaluateCrossValidationMlTask(estimator_template=lr_estimator,
                                                       data_set_file_path=path.join('data','dataset-1.csv'),
                                                       export_classification_reports=True,
                                                       export_confusion_matrixes=True,
                                                       export_pr_curves=True,
                                                       export_roc_curves=True,
                                                       export_false_positives_reports=True,
                                                       export_false_negatives_reports=True,
                                                       export_also_for_train_folds=True,
                                                       random_seed=random_seed).custom_folds(folds_file_path=path.join('data','folds-1.csv')))

print(results['threshold'])
print(results['threshold_metrics'])
print(results['splits_threshold_metrics'])
print(results['splits_threshold_metrics_summary'])
