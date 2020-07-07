from os import path

from sand.experiment import Experiment
from sand.train_ml_task import TrainMlTask
from sand.evaluate_cross_validation_ml_task import EvaluateCrossValidationMlTask
from sand.feature_selection_cross_validation_ml_task import FeatureSelectionCrossValidationMlTask
from sand.hyperparameters_search_cross_validation_ml_task import HyperParametersSearchCrossValidationMlTask

experiment = Experiment('output', __file__).set_experimenter('echatzikyriakidis').build()

######### Experiment

from sklearn.linear_model import LogisticRegression

data_set_file_path = path.join('data', 'dataset-1.csv')
folds_file_path = path.join('data', 'folds-1.csv')

random_seed = 42

lr_estimator = LogisticRegression(solver='liblinear', random_state=random_seed)

search_params = { "C" : [1.e-01, 1.e+00, 1.e+01], "penalty" : [ "l1", "l2" ] }

features_columns = experiment.run(FeatureSelectionCrossValidationMlTask (estimator=lr_estimator,
                                                                         data_set_file_path=data_set_file_path,
                                                                         random_seed=random_seed).custom_folds(folds_file_path=folds_file_path))

hyperparameters_search_results = experiment.run(HyperParametersSearchCrossValidationMlTask (estimator=lr_estimator,
                                                                                            search_params=search_params,
                                                                                            data_set_file_path=data_set_file_path,
                                                                                            feature_columns=features_columns,
                                                                                            random_seed=random_seed).random_search().custom_folds(folds_file_path=folds_file_path))

evaluation_results = experiment.run(EvaluateCrossValidationMlTask(estimator=lr_estimator,
                                                                  estimator_params=hyperparameters_search_results['best_params'],
                                                                  data_set_file_path=data_set_file_path,
                                                                  export_classification_reports=True,
                                                                  export_confusion_matrixes=True,
                                                                  export_pr_curves=True,
                                                                  export_roc_curves=True,
                                                                  export_false_positives_reports=True,
                                                                  export_false_negatives_reports=True,
                                                                  export_also_for_train_folds=True,
                                                                  feature_columns=features_columns,
                                                                  random_seed=random_seed).custom_folds(folds_file_path=folds_file_path))

train_results = experiment.run(TrainMlTask(estimator=lr_estimator,
                                           estimator_params=hyperparameters_search_results['best_params'],
                                           data_set_file_path=data_set_file_path,
                                           feature_columns=features_columns,
                                           random_seed=random_seed))

print(features_columns)

print(hyperparameters_search_results['best_params'])
print(hyperparameters_search_results['best_estimator'])
print(hyperparameters_search_results['best_score'])
print(hyperparameters_search_results['search_results'])

print(evaluation_results['threshold'])
print(evaluation_results['threshold_metrics'])
print(evaluation_results['splits_threshold_metrics'])
print(evaluation_results['splits_threshold_metrics_summary'])

print(train_results['estimator'])
