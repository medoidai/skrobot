from os import path

from sklearn.linear_model import LogisticRegression

from skrobot.core import Experiment
from skrobot.tasks import FeatureSelectionCrossValidationTask

######### Initialization Code

random_seed = 42

lr_estimator = LogisticRegression(solver='liblinear', random_state=random_seed)

######### skrobot Code

# Build an Experiment
experiment = Experiment('experiments-output').set_source_code_file_path(__file__).set_experimenter('echatzikyriakidis').build()

# Run Feature Selection Task
features_columns = experiment.run(FeatureSelectionCrossValidationTask (estimator=lr_estimator,
                                                                       train_data_set=path.join('data','money-laundering-data-train.csv'),
                                                                       random_seed=random_seed).custom_folds(folds_data=path.join('data','money-laundering-folds.csv')))

# Print in-memory results
print(features_columns)