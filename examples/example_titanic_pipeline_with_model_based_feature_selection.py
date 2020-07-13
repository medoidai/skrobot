from os import path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

from sand.core import Experiment
from sand.tasks import TrainTask
from sand.tasks import FeatureSelectionCrossValidationTask
from sand.tasks import EvaluateCrossValidationTask
from sand.tasks import HyperParametersSearchCrossValidationTask
from sand.feature_selection import ColumnSelector
from sand.notification import BaseNotifier

######### Scikit-learn Code

train_data_set_file_path = path.join('data', 'titanic-train.csv')
test_data_set_file_path = path.join('data', 'titanic-test.csv')

random_seed = 42

id_column = 'PassengerId'

label_column = 'Survived'

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer()),
    ('scaler', StandardScaler())])

sex_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore'))])

embarked_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(transformers=[
    ('numerical_transfomer', numeric_transformer, ['Age', 'Fare']),
    ('sex_transfomer', sex_transformer, ['Sex']),
    ('embarked_transfomer', embarked_transformer, ['Embarked'])])

classifier = LogisticRegression(solver='liblinear', random_state=random_seed)

search_params = {
    "classifier__C" : [ 1.e-01, 1.e+00, 1.e+01 ],
    "classifier__penalty" : [ "l1", "l2" ],
    "preprocessor__numerical_transfomer__imputer__strategy" : [ "mean", "median" ]
}

######### Sand Code

# Define a Notifier (This is optional and you can implement any notifier you want, e.g. for Slack / Jira / Discord)
class ConsoleNotifier(BaseNotifier):
    def notify (self, message):
        print(message)

# Build an Experiment
experiment = Experiment('output', __file__).set_experimenter('echatzikyriakidis').set_notifier(ConsoleNotifier()).build()

# Run Feature Selection Task
features_columns = experiment.run(FeatureSelectionCrossValidationTask (estimator=classifier,
                                                                       train_data_set_file_path=train_data_set_file_path,
                                                                       preprocessor=preprocessor,
                                                                       min_features_to_select=3,
                                                                       id_column=id_column,
                                                                       label_column=label_column,
                                                                       random_seed=random_seed).stratified_folds(total_folds=5, shuffle=True))

pipe = Pipeline(steps=[('preprocessor', preprocessor),
                       ('selector', ColumnSelector(cols=features_columns)),
                       ('classifier', classifier)])

# Run Hyperparameters Search Task
hyperparameters_search_results = experiment.run(HyperParametersSearchCrossValidationTask (estimator=pipe,
                                                                                          search_params=search_params,
                                                                                          train_data_set_file_path=train_data_set_file_path,
                                                                                          id_column=id_column,
                                                                                          label_column=label_column,
                                                                                          random_seed=random_seed).random_search(n_iters=100).stratified_folds(total_folds=5, shuffle=True))

# Run Evaluation Task
evaluation_results = experiment.run(EvaluateCrossValidationTask(estimator=pipe,
                                                                estimator_params=hyperparameters_search_results['best_params'],
                                                                train_data_set_file_path=train_data_set_file_path,
                                                                test_data_set_file_path=test_data_set_file_path,
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
                                         train_data_set_file_path=train_data_set_file_path,
                                         id_column=id_column,
                                         label_column=label_column,
                                         random_seed=random_seed))

# Print in-memory results
print(features_columns)

print(hyperparameters_search_results['best_params'])
print(hyperparameters_search_results['best_estimator'])
print(hyperparameters_search_results['best_score'])
print(hyperparameters_search_results['search_results'])

print(evaluation_results['threshold'])
print(evaluation_results['cv_threshold_metrics'])
print(evaluation_results['cv_splits_threshold_metrics'])
print(evaluation_results['cv_splits_threshold_metrics_summary'])
print(evaluation_results['test_threshold_metrics'])

print(train_results['estimator'])