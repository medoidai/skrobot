from os import path

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.linear_model import SGDClassifier

from sand.core import Experiment
from sand.tasks import TrainTask
from sand.tasks import EvaluateCrossValidationTask
from sand.tasks import HyperParametersSearchCrossValidationTask
from sand.feature_selection import ColumnSelector

######### Scikit-learn Code

random_seed = 42

field_delimiter = '\t'

data_set_file_path = path.join('data', 'dataset-3.tsv')

pipe = Pipeline(steps=[
    ('column_selection', ColumnSelector(cols=['message'], drop_axis=True)),
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('feature_selection', SelectPercentile(chi2)),
    ('classifier', SGDClassifier(loss='log'))])

search_params = {
    'classifier__max_iter': [ 20, 50, 80 ],
    'classifier__alpha': [ 0.00001, 0.000001 ],
    'classifier__penalty': [ 'l2', 'elasticnet' ],
    "vectorizer__stop_words" : [ "english", None ],
    "vectorizer__ngram_range" : [ (1, 1), (1, 2) ],
    "vectorizer__max_df": [ 0.5, 0.75, 1.0 ],
    "tfidf__use_idf" : [ True, False ],
    "tfidf__norm" : [ 'l1', 'l2' ],
    "feature_selection__percentile" : [ 70, 60, 50 ]
}

######### Sand Code

# Build an Experiment
experiment = Experiment('output', __file__).set_experimenter('echatzikyriakidis').build()

# Run Hyperparameters Search Task
hyperparameters_search_results = experiment.run(HyperParametersSearchCrossValidationTask (estimator=pipe,
                                                                                          search_params=search_params,
                                                                                          data_set_file_path=data_set_file_path,
                                                                                          field_delimiter=field_delimiter,
                                                                                          random_seed=random_seed).random_search().stratified_folds(total_folds=5, shuffle=True))

# Run Evaluation Task
evaluation_results = experiment.run(EvaluateCrossValidationTask(estimator=pipe,
                                                                estimator_params=hyperparameters_search_results['best_params'],
                                                                data_set_file_path=data_set_file_path,
                                                                field_delimiter=field_delimiter,
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
                                         field_delimiter=field_delimiter,
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