from os import path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

from sand.experiment import Experiment
from sand.tasks import TrainTask
from sand.tasks import EvaluateCrossValidationTask
from sand.tasks import HyperParametersSearchCrossValidationTask

######### Scikit-learn Code

data_set_file_path = path.join('data', 'dataset-2.csv')

random_seed = 42

id_column = 'PassengerId'

label_column = 'Survived'

numeric_features = ['Age', 'Fare']

categorical_features = ['Embarked', 'Sex']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer()),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(transformers=[
    ('numerical', numeric_transformer, numeric_features),
    ('categorical', categorical_transformer, categorical_features)])

classifier = LogisticRegression(solver='liblinear', random_state=random_seed)

pipe = Pipeline(steps=[('preprocessor', preprocessor),
                       ('selection', SelectKBest()),
                       ('classifier', classifier)])

search_params = {
    "classifier__C" : [ 1.e-01, 1.e+00, 1.e+01 ],
    "classifier__penalty" : [ "l1", "l2" ],
    "preprocessor__numerical__imputer__strategy" : [ "mean", "median" ],
    "selection__k" : [ 5, 6, 7 ]
}

######### Sand Code

# Build an Experiment
experiment = Experiment('output', __file__).set_experimenter('echatzikyriakidis').build()

# Run Hyperparameters Search Task
hyperparameters_search_results = experiment.run(HyperParametersSearchCrossValidationTask (estimator=pipe,
                                                                                          search_params=search_params,
                                                                                          data_set_file_path=data_set_file_path,
                                                                                          id_column=id_column,
                                                                                          label_column=label_column,
                                                                                          random_seed=random_seed).random_search().stratified_folds(total_folds=5, shuffle=True))

# Run Evaluation Task
evaluation_results = experiment.run(EvaluateCrossValidationTask(estimator=pipe,
                                                                estimator_params=hyperparameters_search_results['best_params'],
                                                                data_set_file_path=data_set_file_path,
                                                                id_column=id_column,
                                                                label_column=label_column,
                                                                random_seed=random_seed,
                                                                export_classification_reports=True,
                                                                export_confusion_matrixes=True,
                                                                export_pr_curves=True,
                                                                export_roc_curves=True,
                                                                export_false_positives_reports=True,
                                                                export_false_negatives_reports=True,
                                                                export_also_for_train_folds=True).stratified_folds(total_folds=5, shuffle=True))

# Run Train Task
train_results = experiment.run(TrainTask(estimator=pipe,
                                         estimator_params=hyperparameters_search_results['best_params'],
                                         data_set_file_path=data_set_file_path,
                                         id_column=id_column,
                                         label_column=label_column,
                                         random_seed=random_seed))

# Print in-memory results
print(hyperparameters_search_results['best_params'])
print(hyperparameters_search_results['best_estimator'])
print(hyperparameters_search_results['best_score'])
print(hyperparameters_search_results['search_results'])

print(evaluation_results['threshold'])
print(evaluation_results['threshold_metrics'])
print(evaluation_results['splits_threshold_metrics'])
print(evaluation_results['splits_threshold_metrics_summary'])

print(train_results['estimator'])