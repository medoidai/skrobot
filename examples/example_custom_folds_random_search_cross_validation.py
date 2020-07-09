from os import path

from sklearn.linear_model import LogisticRegression

from sand.experiment import Experiment
from sand.tasks import HyperParametersSearchCrossValidationTask

######### Scikit-learn Code

random_seed = 42

lr_estimator = LogisticRegression(solver='liblinear', random_state=random_seed)

######### Sand Code

# Build an Experiment
experiment = Experiment('output', __file__).set_experimenter('echatzikyriakidis').build()

# Run Hyperparameters Search Task
results = experiment.run(HyperParametersSearchCrossValidationTask (estimator=lr_estimator,
                                                                   search_params={ "C" : [1.e-01, 1.e+00, 1.e+01], "penalty" : [ "l1", "l2" ] },
                                                                   data_set_file_path=path.join('data','dataset-1.csv'),
                                                                   random_seed=random_seed).random_search().custom_folds(folds_file_path=path.join('data','folds-1.csv')))

# Print in-memory results
print(results['best_estimator'])
print(results['best_params'])
print(results['best_score'])
print(results['search_results'])
