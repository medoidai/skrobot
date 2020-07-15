[![Python](https://img.shields.io/badge/python-3.6%20%7C%203.7-blue?style=plastic)](https://pypi.org/project/sand/)
[![PyPI](https://img.shields.io/badge/pypi_package-1.0.0-blue?style=plastic)](https://pypi.org/project/sand/)
[![License](https://img.shields.io/badge/license-MIT-blue?style=plastic)](https://github.com/medoidai/sand/blob/master/LICENSE.txt)
[![Status](https://img.shields.io/badge/status-stable-green?style=plastic)](https://pypi.org/project/sand/)

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
| Base Task | All tasks inherit from this component. A task is a configurable and reproducible piece of code built on top of scikit-learn that can be used in machine learning pipelines. |
| Base Notifier | All notifiers inherit from this component. A notifier can be used to send success / failure notifications for tasks execution. |
| Base Cross Validation Task | All tasks that use cross validation functionality inherit from this component. |
| Train Task | This task can be used to fit a scikit-learn estimator on some data. |
| Evaluation Cross Validation Task | This task can be used to evaluate a scikit-learn estimator on some data. |
| Feature Selection Cross Validation Task | This task can be used to perform feature selection with Recursive Feature Elimination using a scikit-learn estimator on some data. |
| Hyperparameters Search Cross Validation Task | This task can be used to search the best hyperparameters of a scikit-learn estimator on some data. |
| Experiment | This is used to build and run experiments. It can run tasks in the context of an experiment and glue everything together to complete a modelling pipeline. |
| Task Runner | This is like the Experiment component but without the "experiment" stuff. It can be used to run various tasks and glue everything together to complete a modelling pipeline. |

#### Evaluation Cross Validation Task

* Cross validation runs by default and can be configured to use either stratified k-folds or custom folds

* The provided estimator is not affected and is used only as a template

* The provided estimator can be either a scikit-learn machine learning model (e.g., LogisticRegression) or a pipeline ending with an estimator

* The provided estimator needs to be able to predict probabilities through a ``predict_proba`` method

* The following evaluation results can be generated on-demand for hold-out test set as well as train / validation CV folds:

  * PR / ROC Curves
  * Confusion Matrixes
  * Classification Reports
  * Performance Metrics
  * False Positives
  * False Negatives

* The evaluation results can be generated either for a specifc provided threshold or for the best one found from threshold tuning

* The threshold used along with its related performance metrics and summary metrics from all CV splits as well as hold-out test set are returned as a result

#### Feature Selection Cross Validation Task

* Cross validation runs by default and can be configured to use either stratified k-folds or custom folds

* The provided estimator can be either a scikit-learn machine learning model (e.g., LogisticRegression) or a pipeline ending with an estimator

* The provided estimator needs to provide feature importances through either a ``coef_`` or a ``feature_importances_`` attribute

* Along with the provided estimator a preprocessor can also be provided to preprocess the data before feature selection runs

* The provided estimator and preprocessor are not affected and are used only as templates

* The selected features can be either column names (from the original data) or column indexes (from the preprocessed data) depending on whether a preprocessor was used or not

* The selected features are stored in a text file and also returned as a result

#### Train Task

* The provided estimator is not affected and is used only as a template

* The provided estimator can be either a scikit-learn machine learning model (e.g., LogisticRegression) or a pipeline ending with an estimator

* The fitted estimator is stored as a pickle file and also returned as a result

#### Hyperparameters Search Cross Validation Task

* Cross validation runs by default and can be configured to use either stratified k-folds or custom folds

* The provided estimator is not affected and is used only as a template

* The provided estimator can be either a scikit-learn machine learning model (e.g., LogisticRegression) or a pipeline ending with an estimator

* The search can be either randomized or grid-based

* The search results as well as the best estimator found with its related hyperparameters and score are returned as a result

* The search results are stored in a file as a static HTML table

#### Experiment

* Each experiment when it runs it leaves in a unique directory a footprint of metadata files (experiment source code, experiment ID, experiment date/time, experimenter name, experiment default / overwritten parameters in JSON format)

* Notifications can be send after running a task, through an easy to implement API (it can be useful for teams who need to get notified for the progress of the experiment)

* In case of error when running a task, a text file will be generated with the related error

#### Task Runner

* It runs the provided tasks and saves in a file the default / overwritten parameters in JSON format

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

See our license ([LICENSE.txt](https://github.com/medoidai/sand/blob/master/LICENSE.txt)) for more details.

### How do I use it?

Many examples can be found in the [examples](https://github.com/medoidai/sand/tree/master/examples) directory.

Below, are some examples that use many of Sand's components to built a machine learning modelling pipeline.

#### Example on Titanic Dataset ([auto-generated results](https://github.com/medoidai/sand/tree/master/examples/experiments-output/echatzikyriakidis-2020-07-14T14-10-24-example-titanic-pipeline-with-model-based-feature-selection))

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
from sand.tasks import EvaluationCrossValidationTask
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
experiment = Experiment('experiments-output', __file__).set_experimenter('echatzikyriakidis').set_notifier(ConsoleNotifier()).build()

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
evaluation_results = experiment.run(EvaluationCrossValidationTask(estimator=pipe,
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

#### Example on SMS Spam Collection Dataset ([auto-generated results](https://github.com/medoidai/sand/tree/master/examples/experiments-output/echatzikyriakidis-2020-07-14T13-02-29-example-sms-spam-ham-pipeline-with-filtering-feature-selection))

```python
from os import path

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.linear_model import SGDClassifier

from sand.core import Experiment
from sand.tasks import TrainTask
from sand.tasks import EvaluationCrossValidationTask
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
experiment = Experiment('experiments-output', __file__).set_experimenter('echatzikyriakidis').build()

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

### Sample of auto-generated results

#### Classification Reports

![Image 1](https://github.com/medoidai/sand/raw/master/static/image-1.png)
![Image 2](https://github.com/medoidai/sand/raw/master/static/image-2.png)
![Image 3](https://github.com/medoidai/sand/raw/master/static/image-3.png)
![Image 4](https://github.com/medoidai/sand/raw/master/static/image-4.png)
![Image 5](https://github.com/medoidai/sand/raw/master/static/image-5.png)

#### Confusion Matrixes

![Image 6](https://github.com/medoidai/sand/raw/master/static/image-6.png)
![Image 7](https://github.com/medoidai/sand/raw/master/static/image-7.png)
![Image 8](https://github.com/medoidai/sand/raw/master/static/image-8.png)
![Image 9](https://github.com/medoidai/sand/raw/master/static/image-9.png)
![Image 10](https://github.com/medoidai/sand/raw/master/static/image-10.png)

#### False Negatives

![Image 11](https://github.com/medoidai/sand/raw/master/static/image-11.png)
![Image 12](https://github.com/medoidai/sand/raw/master/static/image-12.png)
![Image 13](https://github.com/medoidai/sand/raw/master/static/image-13.png)

#### False Positives

![Image 14](https://github.com/medoidai/sand/raw/master/static/image-14.png)
![Image 15](https://github.com/medoidai/sand/raw/master/static/image-15.png)
![Image 16](https://github.com/medoidai/sand/raw/master/static/image-16.png)

#### PR / ROC Curves

![Image 17](https://github.com/medoidai/sand/raw/master/static/image-17.png)
![Image 18](https://github.com/medoidai/sand/raw/master/static/image-18.png)
![Image 19](https://github.com/medoidai/sand/raw/master/static/image-19.png)
![Image 20](https://github.com/medoidai/sand/raw/master/static/image-20.png)
![Image 21](https://github.com/medoidai/sand/raw/master/static/image-21.png)
![Image 22](https://github.com/medoidai/sand/raw/master/static/image-22.png)
![Image 23](https://github.com/medoidai/sand/raw/master/static/image-23.png)
![Image 24](https://github.com/medoidai/sand/raw/master/static/image-24.png)
![Image 25](https://github.com/medoidai/sand/raw/master/static/image-25.png)
![Image 26](https://github.com/medoidai/sand/raw/master/static/image-26.png)

#### Performance Metrics

*On train / validation CV folds:*

![Image 27](https://github.com/medoidai/sand/raw/master/static/image-27.png)
![Image 28](https://github.com/medoidai/sand/raw/master/static/image-28.png)

*On hold-out test set:*

![Image 38](https://github.com/medoidai/sand/raw/master/static/image-38.png)

#### Hyperparameters Search Results

![Image 29](https://github.com/medoidai/sand/raw/master/static/image-29.png)

#### Task Parameters Logging

![Image 30](https://github.com/medoidai/sand/raw/master/static/image-30.png)
![Image 31](https://github.com/medoidai/sand/raw/master/static/image-31.png)
![Image 32](https://github.com/medoidai/sand/raw/master/static/image-32.png)
![Image 33](https://github.com/medoidai/sand/raw/master/static/image-33.png)

#### Experiment Logging

![Image 34](https://github.com/medoidai/sand/raw/master/static/image-34.png)

#### Features Selected

*The selected column indexes from the transformed features (this is generated when a preprocessor is used):*

![Image 35](https://github.com/medoidai/sand/raw/master/static/image-35.png)

*The selected column names from the original features (this is generated when no preprocessor is used):*

![Image 36](https://github.com/medoidai/sand/raw/master/static/image-36.png)

#### Expreriment Source Code

![Image 37](https://github.com/medoidai/sand/raw/master/static/image-37.png)

**Thank you!**
