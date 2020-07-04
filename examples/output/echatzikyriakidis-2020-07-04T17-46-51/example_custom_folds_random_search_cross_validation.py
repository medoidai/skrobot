from os import path

from sand.experiment import Experiment
from sand.hyperparameters_search_cross_validation_ml_task import HyperParametersSearchCrossValidationMlTask

experiment = Experiment('output', __file__).set_experimenter('echatzikyriakidis').build()

######### Experiment

from sklearn.linear_model import LogisticRegression

random_seed = 42

estimator = LogisticRegression(solver='liblinear', random_state=random_seed)

results = experiment.run(HyperParametersSearchCrossValidationMlTask (estimator_template=estimator,
                                                                     search_params={ "C" : [1.e-01, 1.e+00, 1.e+01], "penalty" : [ "l1", "l2" ] },
                                                                     data_set_file_path=path.join('data','dataset-1.csv'),
                                                                     random_seed=random_seed).random_search().custom_folds(folds_file_path=path.join('data','folds-1.csv')))

print(results['best_estimator'])
print(results['best_params'])
print(results['best_score'])
print(results['search_results'])
