from os import path

from sand.experiment import Experiment
from sand.train_ml_task import TrainMlTask
from sand.feature_selection_cross_validation_ml_task import FeatureSelectionCrossValidationMlTask
from sand.evaluate_cross_validation_ml_task import EvaluateCrossValidationMlTask
from sand.hyperparameters_search_cross_validation_ml_task import HyperParametersSearchCrossValidationMlTask
from sand.feature_selection.column_selector import ColumnSelector
from sand.notification.base_notifier import BaseNotifier

class ConsoleNotifier(BaseNotifier):
    def notify (self, message):
        print(message)

experiment = Experiment('output', __file__).set_experimenter('echatzikyriakidis').set_notifier(ConsoleNotifier()).build()

######### Experiment

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

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

search_params = {
    "classifier__C" : [ 1.e-01, 1.e+00, 1.e+01 ],
    "classifier__penalty" : [ "l1", "l2" ],
    "preprocessor__numerical__imputer__strategy" : [ "mean", "median" ]
}

features_columns = experiment.run(FeatureSelectionCrossValidationMlTask (estimator_template=classifier,
                                                                         data_set_file_path=data_set_file_path,
                                                                         preprocessor_template=preprocessor,
                                                                         min_features_to_select=3,
                                                                         id_column=id_column,
                                                                         label_column=label_column,
                                                                         random_seed=random_seed).stratified_folds(total_folds=5, shuffle=True))

pipe = Pipeline(steps=[('preprocessor', preprocessor),
                       ('selector', ColumnSelector(cols=features_columns)),
                       ('classifier', classifier)])

hyperparameters_search_results = experiment.run(HyperParametersSearchCrossValidationMlTask (estimator_template=pipe,
                                                                                            search_params=search_params,
                                                                                            data_set_file_path=data_set_file_path,
                                                                                            id_column=id_column,
                                                                                            label_column=label_column,
                                                                                            random_seed=random_seed).random_search(n_iters=100).stratified_folds(total_folds=5, shuffle=True))

evaluation_results = experiment.run(EvaluateCrossValidationMlTask(estimator_template=pipe,
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

train_results = experiment.run(TrainMlTask(estimator_template=pipe,
                                           estimator_params=hyperparameters_search_results['best_params'],
                                           data_set_file_path=data_set_file_path,
                                           id_column=id_column,
                                           label_column=label_column,
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