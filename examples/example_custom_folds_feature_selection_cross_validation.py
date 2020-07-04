from os import path

from sand.experiment import Experiment
from sand.feature_selection_cross_validation_ml_task import FeatureSelectionCrossValidationMlTask

experiment = Experiment('output', __file__).set_experimenter('echatzikyriakidis').build()

######### Experiment

from sklearn.linear_model import LogisticRegression

random_seed = 42

estimator = LogisticRegression(solver='liblinear', random_state=random_seed)

features_columns = experiment.run(FeatureSelectionCrossValidationMlTask (estimator_template=estimator,
                                                                         data_set_file_path=path.join('data','dataset-1.csv'),
                                                                         random_seed=random_seed).custom_folds(folds_file_path=path.join('data','folds-1.csv')))

print(features_columns)
