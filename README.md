<br /><p align="center"><a href="https://www.medoid.ai/" target="_blank"><img src="https://www.medoid.ai/wp-content/uploads/2020/05/medoid-ai-logo-2.png" width="300px;" /></a></p>

## Sand

### What is it about?

Sand is a Python module for designing, running and tracking Machine Learning experiments / tasks. It is built on top of [scikit-learn](https://scikit-learn.org/) framework.

### How do I install it?

```sh
$ pip install sand
```

### Which are the components?

| Component                      | What is this?                    |
|--------------------------------|----------------------------------|
| Base ML Task | An ML task is a configurable and reproducible piece of code built on top of scikit-learn that implements a repetitive Machine Learning task |
| Evaluation ML Task | This task can be used to evaluate a scikit-learn estimator on some data |
| Feature Selection ML Task | This task can be used to perform feature selection with Recursive Feature Elimination using a scikit-learn estimator on some data |
| Train ML Task | This task can be used to fit a scikit-learn estimator on some data |
| Hyperparameters Search ML Task | This task can be used to search the best hyperparameters of a scikit-learn estimator on some data |
| Experiments Runner | The experiments runner runs ML tasks in the context of an experiment |
| ML Tasks Runner | The ML tasks runner is like the experiments runner but without the "experiment" stuff, thus it can be used in the production world |

**NOTE** : Currently, binary classification is supported only.

#### Evaluation ML Task

* Cross validation is used with either stratified K folds or custom folds

* The provided estimator is not affected and is used only as a template

* The estimator needs to be able to predict probabilities

* The following evaluation information can be generated on demand for train / test folds of CV splits: interactive PR / ROC curve plots, Confusion Matrixes / Classification Reports, False Positives / Negatives and various threshold performance metrics measured

* The task can either perform threshold tuning or can run for a specific threshold

* All the evaluation results are calculated either for the best found threshold on test mean score or for the provided one

* The best found or provided threshold along with its performance metrics is returned as well as summary metrics from all CV splits

* The provided estimator can be either a scikit-learn Machine Learning model (eg: LogisticRegression) or a pipeline ending with an estimator

#### Feature Selection ML Task

* Cross validation is used with either stratified K folds or custom folds

* The provided estimator is not affected and is used only as a template

* Along with the provided estimator a preprocessor can be provided to preprocess the data before feature selection runs

* This is useful when in our pipeline we have some feature transformation

* The selected features are stored in a file in the disk

* In case no preprocessor is provided and feature selection happens on the provided data the features are returned and stored from the column names

* However, in case a preprocessor is provided then the column indexes from the transformed features are returned / stored

#### Train ML Task

* The provided estimator is not affected and is used only as a template

* The fitted estimator is returned as well as it is stored in the disk as a pickle file

* The provided estimator can be either a scikit-learn Machine Learning model (eg: LogisticRegression) or a pipeline ending with an estimator

#### Hyperparameters Search ML Task

* The search can be either randomized or grid-based

* Cross validation is used with either stratified K folds or custom folds

* The provided estimator is not affected and is used only as a template

* The best estimator is returned (fitted on all data) along with its hyperparameters / score and all the results from the search

* The search results are stored also in the disk as an HTML table where the rows are sorted by the on mean test objective score performance across all CV splits

* The provided estimator can be either a scikit-learn Machine Learning model (eg: LogisticRegression) or a pipeline ending with an estimator

#### Experiments Runner

* Each experiment when it runs it leaves a footprint of metadata in a directory

* For each experiment the source code is kept as well as metadata information such as the experiment ID, the experiment date/time, the experimenter (the user / system that run the experiment), the default / overloaded parameters of the ML tasks in JSON format

* In case of error when running an ML task a file will be generated with the error

* The experiment runner can be configured to send notifications after running a task either succesfully or not

#### ML Tasks Runner

* It is used just to run tasks as well as keep the default / overloaded parameters in JSON format

* In case of error when running a task a file will be generated with the error

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
