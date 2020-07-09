from os import path

from sklearn.linear_model import LogisticRegression

from sand.core import Experiment
from sand.tasks import FeatureSelectionCrossValidationTask

######### Scikit-learn Code

random_seed = 42

lr_estimator = LogisticRegression(solver='liblinear', random_state=random_seed)

######### Sand Code

# Build an Experiment
experiment = Experiment('output', __file__).set_experimenter('echatzikyriakidis').build()

# Run Feature Selection Task
features_columns = experiment.run(FeatureSelectionCrossValidationTask (estimator=lr_estimator,
                                                                       data_set_file_path=path.join('data','dataset-1.csv'),
                                                                       random_seed=random_seed).custom_folds(folds_file_path=path.join('data','folds-1.csv')))

# Print in-memory results
print(features_columns)