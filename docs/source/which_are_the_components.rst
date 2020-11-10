Which are the components?
=========================

For the module users
--------------------

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
| Experiment                      | This is used to build and run      |
|                                 | experiments. It can run tasks in   |
|                                 | the context of an experiment and   |
|                                 | glue everything together to        |
|                                 | complete a modelling pipeline.     |
+---------------------------------+------------------------------------+
| Task Runner                     | This is like the Experiment        |
|                                 | component but without the          |
|                                 | “experiment” stuff. It can be used |
|                                 | to run various tasks and glue      |
|                                 | everything together to complete a  |
|                                 | modelling pipeline.                |
+---------------------------------+------------------------------------+

For the module developers
-------------------------

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