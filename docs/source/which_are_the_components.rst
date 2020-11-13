Which are the components?
=========================

For the module's users
----------------------

+---------------------------------+------------------------------------+
| Component                       | What is this?                      |
+=================================+====================================+
| Train Task                      | This task can be used to fit a     |
|                                 | scikit-learn estimator on some     |
|                                 | data.                              |
+---------------------------------+------------------------------------+
| Prediction Task                 | This task can be used to predict   |
|                                 | new data using a scikit-learn      |
|                                 | estimator.                         |
+---------------------------------+------------------------------------+
| Evaluation Cross Validation     | This task can be used to evaluate  |
| Task                            | a scikit-learn estimator on some   |
|                                 | data.                              |
+---------------------------------+------------------------------------+
| Feature Selection Cross         | This task can be used to perform   |
| Validation Task                 | feature selection with Recursive   |
|                                 | Feature Elimination using a        |
|                                 | scikit-learn estimator on some     |
|                                 | data.                              |
+---------------------------------+------------------------------------+
| Hyperparameters Search Cross    | This task can be used to search    |
| Validation Task                 | the best hyperparameters of a      |
|                                 | scikit-learn estimator on some     |
|                                 | data.                              |
+---------------------------------+------------------------------------+
| Experiment                      | This is used to build, track and   |
|                                 | run an experiment. It can run      |
|                                 | tasks in the context of an         |
|                                 | experiment.                        |
+---------------------------------+------------------------------------+
| Task Runner                     | This is a simplified version (in   |
|                                 | functionality) of the Experiment   |
|                                 | component. It leaves out all the   |
|                                 | “experiment” stuff and is focused  |
|                                 | mostly in the execution and        |
|                                 | tracking of tasks.                 |
+---------------------------------+------------------------------------+

For the module's developers
---------------------------

+---------------------------------+------------------------------------+
| Component                       | What is this?                      |
+=================================+====================================+
| Base Task                       | All tasks inherit from this        |
|                                 | component. A task is a             |
|                                 | configurable and reproducible      |
|                                 | piece of code built on top of      |
|                                 | scikit-learn that can be used in   |
|                                 | machine learning pipelines.        |
+---------------------------------+------------------------------------+
| Base Cross Validation Task      | All tasks that use cross           |
|                                 | validation functionality inherit   |
|                                 | from this component.               |
+---------------------------------+------------------------------------+
| Base Notifier                   | All notifiers inherit from this    |
|                                 | component. A notifier can be used  |
|                                 | to send success / failure          |
|                                 | notifications for tasks execution. |
+---------------------------------+------------------------------------+