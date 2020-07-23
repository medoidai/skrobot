from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.linear_model import SGDClassifier

from skrobot.core import Experiment
from skrobot.tasks import TrainTask
from skrobot.tasks import PredictionTask
from skrobot.tasks import EvaluationCrossValidationTask
from skrobot.tasks import HyperParametersSearchCrossValidationTask
from skrobot.feature_selection import ColumnSelector

######### Initialization Code

train_data_set_file_path = 'https://bit.ly/sms-spam-ham-data-train'

test_data_set_file_path = 'https://bit.ly/sms-spam-ham-data-test'

new_data_set_file_path = 'https://bit.ly/sms-spam-ham-data-new'

field_delimiter = '\t'

random_seed = 42

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

######### skrobot Code

# Build an Experiment
experiment = Experiment('experiments-output').set_source_code_file_path(__file__).set_experimenter('echatzikyriakidis').build()

# Run Hyperparameters Search Task
hyperparameters_search_results = experiment.run(HyperParametersSearchCrossValidationTask (estimator=pipe,
                                                                                          search_params=search_params,
                                                                                          train_data_set_file_path=train_data_set_file_path,
                                                                                          field_delimiter=field_delimiter,
                                                                                          random_seed=random_seed).random_search().stratified_folds(total_folds=5, shuffle=True))

# Run Evaluation Task
evaluation_results = experiment.run(EvaluationCrossValidationTask(estimator=pipe,
                                                                  estimator_params=hyperparameters_search_results['best_params'],
                                                                  train_data_set_file_path=train_data_set_file_path,
                                                                  test_data_set_file_path=test_data_set_file_path,
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
                                         train_data_set_file_path=train_data_set_file_path,
                                         field_delimiter=field_delimiter,
                                         random_seed=random_seed))

# Run Prediction Task
predictions = experiment.run(PredictionTask(estimator=train_results['estimator'],
                                            data_set_file_path=new_data_set_file_path,
                                            field_delimiter=field_delimiter,
                                            threshold=evaluation_results['threshold']))

# Print in-memory results
print(hyperparameters_search_results['best_params'])
print(hyperparameters_search_results['best_index'])
print(hyperparameters_search_results['best_estimator'])
print(hyperparameters_search_results['best_score'])
print(hyperparameters_search_results['search_results'])

print(evaluation_results['threshold'])
print(evaluation_results['cv_threshold_metrics'])
print(evaluation_results['cv_splits_threshold_metrics'])
print(evaluation_results['cv_splits_threshold_metrics_summary'])
print(evaluation_results['test_threshold_metrics'])

print(train_results['estimator'])

print(predictions)