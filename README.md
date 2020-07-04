<br /><p align="center"><a href="https://www.medoid.ai/" target="_blank"><img src="https://www.medoid.ai/wp-content/uploads/2020/05/medoid-ai-logo-2.png" width="300px;" /></a></p>

## Sand

### What is it about?

Sand is a Python module for designing, running and tracking Machine Learning experiments / tasks. It is built on top of [scikit-learn](https://scikit-learn.org/) framework.

### How do I install it?

```sh
$ pip install sand
```

### Which are the components?

**NOTE** : Currently, binary classification is supported only.

| Component                      | What is this?                    |
|--------------------------------|----------------------------------|
| Base ML Task | An ML task is a configurable and reproducible piece of code built on top of scikit-learn that implements a repetitive Machine Learning task |
| Evaluation ML Task | This task can be used to evaluate a scikit-learn estimator on some data |
| Feature Selection ML Task | This task can be used to perform feature selection with Recursive Feature Elimination using a scikit-learn estimator on some data |
| Train ML Task | This task can be used to fit a scikit-learn estimator on some data |
| Hyperparameters Search ML Task | This task can be used to search the best hyperparameters of a scikit-learn estimator on some data |
| Experiments Runner | The experiments runner runs ML tasks in the context of an experiment |
| ML Tasks Runner | The ML tasks runner is like the experiments runner but without the "experiment" stuff, thus it can be used in the production world |

#### Evaluation ML Task

* Cross validation runs by default and can be configured to use either stratified k-folds or custom folds

* The provided estimator is not affected and is used only as a template

* The provided estimator can be either a scikit-learn ML model (eg: LogisticRegression) or a pipeline ending with an estimator

* The provided estimator needs to be able to predict probabilities

* The following evaluation results can be generated on-demand for train / test CV folds:

  * PR / ROC Curves (as interactive HTML plots)
  * Confusion Matrixes (as PNG images)
  * Classification Reports (as text)
  * Performance Metrics (as static HTML tables)
  * False Positives (as text)
  * False Negatives (as text)

* The evaluation results can be generated either for a specifc provided threshold or for the best one found from threshold tuning

* The threshold used along with its related performance metrics and summary metrics from all CV splits are returned as a result

#### Feature Selection ML Task

* Cross validation runs by default and can be configured to use either stratified k-folds or custom folds

* The provided estimator can be either a scikit-learn ML model (eg: LogisticRegression) or a pipeline ending with an estimator

* Along with the provided estimator a preprocessor can also be provided to preprocess the data before feature selection runs

* The provided estimator and preprocessor are not affected and are used only as templates

* The selected features can be either column names (from the original data) or column indexes (from the preprocessed data) depending on whether a preprocessor was used or not

* The selected features are stored in a text file and also returned as a result

#### Train ML Task

* The provided estimator is not affected and is used only as a template

* The provided estimator can be either a scikit-learn ML model (eg: LogisticRegression) or a pipeline ending with an estimator

* The fitted estimator is stored as a pickle file and also returned as a result

#### Hyperparameters Search ML Task

* Cross validation runs by default and can be configured to use either stratified k-folds or custom folds

* The provided estimator is not affected and is used only as a template

* The provided estimator can be either a scikit-learn ML model (eg: LogisticRegression) or a pipeline ending with an estimator

* The search can be either randomized or grid-based

* The search results as well as the best estimator found with its related hyperparameters and score are returned as a result

* The search results are stored in a file as a static HTML table

#### Experiments Runner

* Each experiment when it runs it leaves in a unique directory a footprint of metadata (experiment source code, experiment ID, experiment date/time, experimenter name, experiment default / overloaded parameters in JSON format)

* Notifications can be send after running an ML task, through an easy to implement API (it can be useful for teams who need to get notified for the progress of the experiment, eg: in Slack)

* In case of error when running an ML task, a text file will be generated with the related error

#### ML Tasks Runner

* It is used just to run ML tasks as well as track the default / overloaded parameters in JSON format

* In case of error when running an ML task, a text file will be generated with the related error

### How do I use it?

Many examples can be found in the [examples](examples) directory.

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

See our [LICENSE](LICENSE) for more details.
