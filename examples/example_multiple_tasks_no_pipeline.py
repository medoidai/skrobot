from os import path

from sklearn.linear_model import LogisticRegression

from sand.core import Experiment
from sand.tasks import TrainTask
from sand.tasks import EvaluateCrossValidationTask
from sand.tasks import FeatureSelectionCrossValidationTask
from sand.tasks import HyperParametersSearchCrossValidationTask

######### Scikit-learn Code

data_set_file_path = path.join('data', 'dataset-1.csv')

folds_file_path = path.join('data', 'folds-1.csv')

random_seed = 42

lr_estimator = LogisticRegression(solver='liblinear', random_state=random_seed)

search_params = { "C" : [1.e-01, 1.e+00, 1.e+01], "penalty" : [ "l1", "l2" ] }

######### Sand Code

# Build an Experiment
experiment = Experiment('output', __file__).set_experimenter('echatzikyriakidis').build()


# Run Feature Selection Task
features_columns = experiment.run(FeatureSelectionCrossValidationTask (estimator=lr_estimator,
                                                                       train_data_set_file_path=data_set_file_path,
                                                                       random_seed=random_seed).custom_folds(folds_file_path=folds_file_path))

# Run Hyperparameters Search Task
hyperparameters_search_results = experiment.run(HyperParametersSearchCrossValidationTask (estimator=lr_estimator,
                                                                                          search_params=search_params,
                                                                                          train_data_set_file_path=data_set_file_path,
                                                                                          feature_columns=features_columns,
                                                                                          random_seed=random_seed).random_search().custom_folds(folds_file_path=folds_file_path))

# Run Evaluation Task	
evaluation_results = experiment.run(EvaluateCrossValidationTask(estimator=lr_estimator,
                                                                estimator_params=hyperparameters_search_results['best_params'],
                                                                train_data_set_file_path=data_set_file_path,
                                                                test_data_set_file_path=data_set_file_path,
                                                                export_classification_reports=True,
                                                                export_confusion_matrixes=True,
                                                                export_pr_curves=True,
                                                                export_roc_curves=True,
                                                                export_false_positives_reports=True,
                                                                export_false_negatives_reports=True,
                                                                export_also_for_train_folds=True,
                                                                feature_columns=features_columns,
                                                                random_seed=random_seed).custom_folds(folds_file_path=folds_file_path))

# Run Train Task
train_results = experiment.run(TrainTask(estimator=lr_estimator,
                                         estimator_params=hyperparameters_search_results['best_params'],
                                         train_data_set_file_path=data_set_file_path,
                                         feature_columns=features_columns,
                                         random_seed=random_seed))

# Print in-memory results
print(features_columns)

print(hyperparameters_search_results['best_params'])
print(hyperparameters_search_results['best_estimator'])
print(hyperparameters_search_results['best_score'])
print(hyperparameters_search_results['search_results'])

print(evaluation_results['threshold'])
print(evaluation_results['cv_threshold_metrics'])
print(evaluation_results['cv_splits_threshold_metrics'])
print(evaluation_results['cv_splits_threshold_metrics_summary'])
print(evaluation_results['test_threshold_metrics'])

print(train_results['estimator'])