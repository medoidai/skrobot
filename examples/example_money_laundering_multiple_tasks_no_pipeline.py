from os import path

from sklearn.linear_model import LogisticRegression

from skrobot.core import Experiment
from skrobot.tasks import TrainTask
from skrobot.tasks import PredictionTask
from skrobot.tasks import EvaluationCrossValidationTask
from skrobot.tasks import FeatureSelectionCrossValidationTask
from skrobot.tasks import HyperParametersSearchCrossValidationTask

######### Initialization Code

train_data_set = path.join('data','money-laundering-data-train.csv')

test_data_set = path.join('data','money-laundering-data-test.csv')

new_data_set = path.join('data','money-laundering-data-new.csv')

folds_data = path.join('data', 'money-laundering-folds.csv')

random_seed = 42

lr_estimator = LogisticRegression(solver='liblinear', random_state=random_seed)

search_params = { "C" : [1.e-01, 1.e+00, 1.e+01], "penalty" : [ "l1", "l2" ] }

######### skrobot Code

# Build an Experiment
experiment = Experiment('experiments-output').set_source_code_file_path(__file__).set_experimenter('echatzikyriakidis').build()


# Run Feature Selection Task
features_columns = experiment.run(FeatureSelectionCrossValidationTask (estimator=lr_estimator,
                                                                       train_data_set=train_data_set,
                                                                       random_seed=random_seed).custom_folds(folds_data=folds_data))

# Run Hyperparameters Search Task
hyperparameters_search_results = experiment.run(HyperParametersSearchCrossValidationTask (estimator=lr_estimator,
                                                                                          search_params=search_params,
                                                                                          train_data_set=train_data_set,
                                                                                          feature_columns=features_columns,
                                                                                          random_seed=random_seed).random_search().custom_folds(folds_data=folds_data))

# Run Evaluation Task	
evaluation_results = experiment.run(EvaluationCrossValidationTask(estimator=lr_estimator,
                                                                  estimator_params=hyperparameters_search_results['best_params'],
                                                                  train_data_set=train_data_set,
                                                                  test_data_set=test_data_set,
                                                                  export_classification_reports=True,
                                                                  export_confusion_matrixes=True,
                                                                  export_pr_curves=True,
                                                                  export_roc_curves=True,
                                                                  export_false_positives_reports=True,
                                                                  export_false_negatives_reports=True,
                                                                  export_also_for_train_folds=True,
                                                                  feature_columns=features_columns,
                                                                  random_seed=random_seed).custom_folds(folds_data=folds_data))

# Run Train Task
train_results = experiment.run(TrainTask(estimator=lr_estimator,
                                         estimator_params=hyperparameters_search_results['best_params'],
                                         train_data_set=train_data_set,
                                         feature_columns=features_columns,
                                         random_seed=random_seed))

# Run Prediction Task
predictions = experiment.run(PredictionTask(estimator=train_results['estimator'],
                                            data_set=new_data_set,
                                            feature_columns=features_columns,
                                            threshold=evaluation_results['threshold']))

# Print in-memory results
print(features_columns)

print(hyperparameters_search_results['best_params'])
print(hyperparameters_search_results['best_index'])
print(hyperparameters_search_results['best_estimator'])
print(hyperparameters_search_results['best_score'])
print(hyperparameters_search_results['search_results'])

print(evaluation_results['threshold'])
print(evaluation_results['cv_threshold_metrics'])
print(evaluation_results['cv_splits_threshold_metrics'])
print(evaluation_results['cv_splits_threshold_metrics_summary'])
print(evaluation_results['test_threshold_metrics'])

print(train_results['estimator'])

print(predictions)