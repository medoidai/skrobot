![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

<br /><p align="center"><a href="https://www.medoid.ai/" target="_blank"><img src="https://www.medoid.ai/wp-content/uploads/2020/05/medoid-ai-logo-2.png" width="300px;" /></a></p>

## Sand

### What is it about?

Sand is a Python module for designing, running and tracking Machine Learning experiments / tasks. It is built on top of [scikit-learn](https://scikit-learn.org/) framework.

### How do I install it?

#### PyPI

```sh
$ pip install sand
```

#### Development Version

The Sand version on PyPI may always be one step behind; you can install the latest development version from the GitHub repository by executing

```sh
$ pip install git+git://github.com/medoidai/sand.git
```

Or, you can clone the GitHub repository and install Sand from your local drive via

```sh
$ python setup.py install
```

### Which are the components?

**NOTE** : Currently, Sand can be used only for binary classification problems.

| Component                      | What is this?                    |
|--------------------------------|----------------------------------|
| Base Task | A task is a configurable and reproducible piece of code built on top of scikit-learn that can be used in machine learning pipelines |
| Evaluation Task | This task can be used to evaluate a scikit-learn estimator on some data |
| Feature Selection Task | This task can be used to perform feature selection with Recursive Feature Elimination using a scikit-learn estimator on some data |
| Train Task | This task can be used to fit a scikit-learn estimator on some data |
| Hyperparameters Search Task | This task can be used to search the best hyperparameters of a scikit-learn estimator on some data |
| Experiments Runner | The experiments runner runs tasks in the context of an experiment |
| Tasks Runner | The tasks runner is like the experiments runner but without the "experiment" stuff, thus it can be used in the production world |

#### Evaluation Task

* Cross validation runs by default and can be configured to use either stratified k-folds or custom folds

* The provided estimator is not affected and is used only as a template

* The provided estimator can be either a scikit-learn machine learning model (eg: LogisticRegression) or a pipeline ending with an estimator

* The provided estimator needs to be able to predict probabilities through a ``predict_proba`` method

* The following evaluation results can be generated on-demand for hold-out test set as well as train / validation CV folds:

  * PR / ROC Curves (as interactive HTML plots)
  * Confusion Matrixes (as PNG images)
  * Classification Reports (as text)
  * Performance Metrics (as static HTML tables)
  * False Positives (as text)
  * False Negatives (as text)

* The evaluation results can be generated either for a specifc provided threshold or for the best one found from threshold tuning

* The threshold used along with its related performance metrics and summary metrics from all CV splits as well as hold-out test set are returned as a result

#### Feature Selection Task

* Cross validation runs by default and can be configured to use either stratified k-folds or custom folds

* The provided estimator can be either a scikit-learn machine learning model (eg: LogisticRegression) or a pipeline ending with an estimator

* The provided estimator needs to provide feature importances through either a ``coef_`` or a ``feature_importances_`` attribute

* Along with the provided estimator a preprocessor can also be provided to preprocess the data before feature selection runs

* The provided estimator and preprocessor are not affected and are used only as templates

* The selected features can be either column names (from the original data) or column indexes (from the preprocessed data) depending on whether a preprocessor was used or not

* The selected features are stored in a text file and also returned as a result

#### Train Task

* The provided estimator is not affected and is used only as a template

* The provided estimator can be either a scikit-learn machine learning model (eg: LogisticRegression) or a pipeline ending with an estimator

* The fitted estimator is stored as a pickle file and also returned as a result

#### Hyperparameters Search Task

* Cross validation runs by default and can be configured to use either stratified k-folds or custom folds

* The provided estimator is not affected and is used only as a template

* The provided estimator can be either a scikit-learn machine learning model (eg: LogisticRegression) or a pipeline ending with an estimator

* The search can be either randomized or grid-based

* The search results as well as the best estimator found with its related hyperparameters and score are returned as a result

* The search results are stored in a file as a static HTML table

#### Experiments Runner

* Each experiment when it runs it leaves in a unique directory a footprint of metadata files (experiment source code, experiment ID, experiment date/time, experimenter name, experiment default / overloaded parameters in JSON format)

* Notifications can be send after running a task, through an easy to implement API (it can be useful for teams who need to get notified for the progress of the experiment, eg: in Slack)

* In case of error when running a task, a text file will be generated with the related error

#### Tasks Runner

* It runs the provided tasks and saves in a file the default / overloaded parameters in JSON format

* In case of error when running a task, a text file will be generated with the related error

### Why does exists?

It can help Data Scientists and Machine Learning Engineers:

* to keep track of modelling experiments / tasks

* to automate the repetitive (and boring) stuff when designing modelling pipelines

* to spend more time on the things that truly matter when solving a problem

### The people behind it

The following members of our team were involved in developing the initial release of this app:

* [Efstathios Chatzikyriakidis](https://github.com/echatzikyriakidis)

### Can I contribute?

Of course, the project is [Free Software](https://www.gnu.org/philosophy/free-sw.en.html) and you can contribute to it!

### What license do you use?

See our license ([LICENSE.txt](LICENSE.txt)) for more details.

### How do I use it?

Many examples can be found in the [examples](examples) directory.

Below, are some examples that use many of Sand's components to built a machine learning modelling pipeline.

#### Example on Titanic Dataset

```python
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

######### Initialization Code

train_data_set_file_path = path.join('data', 'titanic-train.csv')

test_data_set_file_path = path.join('data', 'titanic-test.csv')

random_seed = 42

id_column = 'PassengerId'

label_column = 'Survived'

numerical_features = ['Age', 'Fare', 'SibSp', 'Parch']

categorical_features = ['Embarked', 'Sex', 'Pclass']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer()),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(transformers=[
    ('numerical_transfomer', numeric_transformer, numerical_features),
    ('categorical_transfomer', categorical_transformer, categorical_features)])

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
                                                                       min_features_to_select=4,
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
```

#### Example on SMS Spam Collection Dataset

```python
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

######### Initialization Code

train_data_set_file_path = path.join('data', 'sms-spam-ham-train.tsv')

test_data_set_file_path = path.join('data', 'sms-spam-ham-test.tsv')

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

######### Sand Code

# Build an Experiment
experiment = Experiment('output', __file__).set_experimenter('echatzikyriakidis').build()

# Run Hyperparameters Search Task
hyperparameters_search_results = experiment.run(HyperParametersSearchCrossValidationTask (estimator=pipe,
                                                                                          search_params=search_params,
                                                                                          train_data_set_file_path=train_data_set_file_path,
                                                                                          field_delimiter=field_delimiter,
                                                                                          random_seed=random_seed).random_search().stratified_folds(total_folds=5, shuffle=True))

# Run Evaluation Task
evaluation_results = experiment.run(EvaluateCrossValidationTask(estimator=pipe,
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
```

#### Example on Money Laundering Dataset

```python
from os import path

from sklearn.linear_model import LogisticRegression

from sand.core import Experiment
from sand.tasks import TrainTask
from sand.tasks import EvaluateCrossValidationTask
from sand.tasks import FeatureSelectionCrossValidationTask
from sand.tasks import HyperParametersSearchCrossValidationTask

######### Initialization Code

train_data_set_file_path = path.join('data','money-laundering-data-train.csv')

test_data_set_file_path = path.join('data','money-laundering-data-test.csv')

folds_file_path = path.join('data', 'money-laundering-folds.csv')

random_seed = 42

lr_estimator = LogisticRegression(solver='liblinear', random_state=random_seed)

search_params = { "C" : [1.e-01, 1.e+00, 1.e+01], "penalty" : [ "l1", "l2" ] }

######### Sand Code

# Build an Experiment
experiment = Experiment('output', __file__).set_experimenter('echatzikyriakidis').build()


# Run Feature Selection Task
features_columns = experiment.run(FeatureSelectionCrossValidationTask (estimator=lr_estimator,
                                                                       train_data_set_file_path=train_data_set_file_path,
                                                                       random_seed=random_seed).custom_folds(folds_file_path=folds_file_path))

# Run Hyperparameters Search Task
hyperparameters_search_results = experiment.run(HyperParametersSearchCrossValidationTask (estimator=lr_estimator,
                                                                                          search_params=search_params,
                                                                                          train_data_set_file_path=train_data_set_file_path,
                                                                                          feature_columns=features_columns,
                                                                                          random_seed=random_seed).random_search().custom_folds(folds_file_path=folds_file_path))

# Run Evaluation Task	
evaluation_results = experiment.run(EvaluateCrossValidationTask(estimator=lr_estimator,
                                                                estimator_params=hyperparameters_search_results['best_params'],
                                                                train_data_set_file_path=train_data_set_file_path,
                                                                test_data_set_file_path=test_data_set_file_path,
                                                                export_classification_reports=True,
                                                                export_confusion_matrixes=True,
                                                                export_pr_curves=True,
                                                                export_roc_curves=True,
                                                                export_false_positives_reports=True,
                                                                export_false_negatives_reports=True,
                                                                export_also_for_train_folds=True,
                                                                feature_columns=features_columns,
                                                                random_seed=random_seed).custom_folds(folds_file_path=folds_file_path))

# Run Train Task
train_results = experiment.run(TrainTask(estimator=lr_estimator,
                                         estimator_params=hyperparameters_search_results['best_params'],
                                         train_data_set_file_path=train_data_set_file_path,
                                         feature_columns=features_columns,
                                         random_seed=random_seed))

# Print in-memory results
print(features_columns)

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
```

### Auto-generated example results

![Classification Report 1](static/classification-report-1.png)
![Classification Report 2](static/classification-report-2.png)
![Classification Report 3](static/classification-report-3.png)
![Classification Report 4](static/classification-report-4.png)
![Confusion Matrix 1](static/confusion-matrix-1.png)
![Confusion Matrix 2](static/confusion-matrix-2.png)
![Confusion Matrix 3](static/confusion-matrix-3.png)
![Confusion Matrix 4](static/confusion-matrix-4.png)
![False Negatives 1](static/false-negatives-1.png)
![False Negatives 2](static/false-negatives-2.png)
![False Positives 1](static/false-positives-1.png)
![False Positives 2](static/false-positives-2.png)
![Mean PR Curve 1](static/mean-pr-curve-1.png)
![Mean PR Curve 2](static/mean-pr-curve-2.png)
![Mean ROC Curve 1](static/mean-roc-curve-1.png)
![Mean ROC Curve 2](static/mean-roc-curve-2.png)
![PR Curve 1](static/pr-curve-1.png)
![PR Curve 2](static/pr-curve-2.png)
![ROC Curve 1](static/roc-curve-1.png)
![ROC Curve 2](static/roc-curve-2.png)
![Hyperparameters Search Results](static/search-results-optimized-for-f1.png)
![Threshold Metrics 1](static/splits-threshold-metrics.png)
![Threshold Metrics 2](static/splits-threshold-metrics-summary.png)
![Experiment Log](static/experiment-log.png)
![Features Selected](static/features-selected.png)
![Feature Selection Task Params](static/feature-selection-cross-validation-task-params.png)
![Hyperparameters Search Task Params](static/hyper-parameters-search-cross-validation-task-params.png)
![Evaluate Task Params](static/evaluate-cross-validation-task-params.png)
![Train Task Params](static/train-task-params.png)

Thank you!
