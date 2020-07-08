from os import path

######### Scikit-learn Code

from sklearn.linear_model import LogisticRegression

random_seed = 42

lr_estimator = LogisticRegression(solver='liblinear', random_state=random_seed)

######### Sand Code

from sand.experiment import Experiment
from sand.feature_selection_cross_validation_ml_task import FeatureSelectionCrossValidationMlTask

# Build an Experiment
experiment = Experiment('output', __file__).set_experimenter('echatzikyriakidis').build()

# Run Feature Selection ML Task
features_columns = experiment.run(FeatureSelectionCrossValidationMlTask (estimator=lr_estimator,
                                                                         data_set_file_path=path.join('data','dataset-1.csv'),
                                                                         random_seed=random_seed).custom_folds(folds_file_path=path.join('data','folds-1.csv')))

# Print in-memory results
print(features_columns)