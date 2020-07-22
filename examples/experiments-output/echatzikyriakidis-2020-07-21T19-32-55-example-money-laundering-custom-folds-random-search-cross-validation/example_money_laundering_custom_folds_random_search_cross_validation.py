from os import path

from sklearn.linear_model import LogisticRegression

from skrobot.core import Experiment
from skrobot.tasks import HyperParametersSearchCrossValidationTask

######### Initialization Code

random_seed = 42

lr_estimator = LogisticRegression(solver='liblinear', random_state=random_seed)

######### skrobot Code

# Build an Experiment
experiment = Experiment('experiments-output').set_source_code_file_path(__file__).set_experimenter('echatzikyriakidis').build()

# Run Hyperparameters Search Task
results = experiment.run(HyperParametersSearchCrossValidationTask (estimator=lr_estimator,
                                                                   search_params={ "C" : [1.e-01, 1.e+00, 1.e+01], "penalty" : [ "l1", "l2" ] },
                                                                   train_data_set_file_path=path.join('data','money-laundering-data-train.csv'),
                                                                   random_seed=random_seed).random_search().custom_folds(folds_file_path=path.join('data','money-laundering-folds.csv')))

# Print in-memory results
print(results['best_estimator'])
print(results['best_params'])
print(results['best_index'])
print(results['best_score'])
print(results['search_results'])
