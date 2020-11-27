import json, os, uuid, datetime, shutil

from numpyencoder import NumpyEncoder

from ..notification import BaseNotifier

class Experiment:
  """
  The :class:`.Experiment` class can be used to build, track and run an experiment.

  It can run :class:`.BaseTask` tasks in the context of an experiment.

  When building an experiment and/or running tasks, various metadata as well as task-related files are stored for tracking experiments.

  Lastly, an experiment can be configured to send notifications when running a task, which can be useful for teams who need to get notified for the progress of the experiment.
  """

  def __init__ (self, experiments_repository):
    """
    This is the constructor method and can be used to create a new object instance of :class:`.Experiment` class.

    :param experiments_repository: The root directory path under which a unique directory is created for the experiment.
    :type experiments_repository: str
    """

    self._experiments_repository = experiments_repository

    self._experimenter = 'anonymous'

    self._source_code_file_path = None

    self._notifier = None

  def set_notifier(self, notifier : BaseNotifier):
    """
    Optional method.
    
    Set the experiment's notifier.

    :param notifier: The experiment's notifier.
    :type notifier: :class:`.BaseNotifier`
    
    :return: The object instance itself.
    :rtype: :class:`.Experiment`
    """

    self._notifier = notifier

    return self

  def set_source_code_file_path(self, source_code_file_path):
    """
    Optional method.
    
    Set the experiment's source code file path.

    :param source_code_file_path: The experiment's source code file path.
    :type source_code_file_path: str
    
    :return: The object instance itself.
    :rtype: :class:`.Experiment`
    """

    self._source_code_file_path = source_code_file_path

    return self

  def set_experimenter(self, experimenter):
    """
    Optional method.
    
    Set the experimenter's name.

    By default the experimenter's name is *anonymous*. However, if you want to override it you can pass a new name.

    :param experimenter: The experimenter's name.
    :type experimenter: str
    
    :return: The object instance itself.
    :rtype: :class:`.Experiment`
    """

    self._experimenter = experimenter

    return self

  def build(self):
    """
    Build the :class:`.Experiment`.

    When an experiment is built, it creates a unique directory under which it stores various experiment-related metadata and files for tracking reasons.

    Specifically, under the experiment's directory an *experiment.log* JSON file is created, which contains a unique auto-generated experiment ID, the current date & time, and the experimenter's name.

    Also, the experiment's directory name contains the experimenter's name as well as current date & time.

    Lastly, in case :meth:`.set_source_code_file_path` is used, the experiment's source code file is copied also under the experiment's directory.

    :return: The object instance itself.
    :rtype: :class:`.Experiment`
    """
    self._create_experiment_log()

    self._create_experiment_directory()

    self._save_experiment_log_file()

    self._save_source_code_file()

    return self

  def run(self, task):
    """
    Run a :class:`.BaseTask` task.

    When running a task, its recorded parameters (e.g., *train_task.params*) and any other task-related generated files are stored under experiment's directory for tracking reasons.

    The task's recorded parameters are in JSON format.

    Also, in case :meth:`.set_notifier` is used to set a notifier, a notification is sent for the success or failure (including the error message) of the task's execution.

    Lastly, in case an exception occurs, a text file (e.g., *train_task.errors*) is generated under experiment's directory containing the error message.

    :param task: The task to run.
    :type task: :class:`.BaseTask`

    :return: The task's result.
    :rtype: Depends on the ``task`` parameter.
    """

    task_type = task.get_type()

    try:
      self._save_configuration_file(task.get_configuration(), task_type)

      results = task.run(self._experiment_directory_path)
      
      self._send_success_notification(task_type)

      return results
    except Exception as exception:
      self._save_errors_file(exception, task_type)

      self._send_failure_notification(exception, task_type)

      raise

  def _create_experiment_log(self):
    self._experiment_log = {
      'datetime' : datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S"),

      'experimenter' : self._experimenter,

      'experiment_id' : uuid.uuid4().hex
    }

  def _send_success_notification(self, task_type):
    self._send_notification(f'The task [{task_type}] under experiment [{self._experiment_log["experiment_id"]}] is completed successfully.')

  def _send_failure_notification(self, exception, task_type):
    self._send_notification(f'The task [{task_type}] under experiment [{self._experiment_log["experiment_id"]}] has failed with error:' + '\n'*2 + f'{repr(exception)}')

  def _send_notification(self, message):
    if self._notifier: self._notifier.notify(message)

  def _create_experiment_directory(self):
    self._experiment_directory_path = os.path.join(self._experiments_repository, f"{self._experimenter}-{self._experiment_log['datetime']}")

    if not os.path.exists(self._experiment_directory_path): os.makedirs(self._experiment_directory_path, exist_ok=True)

  def _save_experiment_log_file(self):
    self._save_dictionary_as_json_file(self._experiment_log, os.path.join(self._experiment_directory_path, 'experiment.log'))

  def _save_source_code_file(self):
    if self._source_code_file_path: shutil.copy(self._source_code_file_path, self._experiment_directory_path)

  def _save_errors_file(self, exception, task_type):
    with open(os.path.join(self._experiment_directory_path, f'{task_type}.errors'), 'w') as f: f.write(repr(exception))

  def _save_configuration_file(self, configuration, task_type):
    self._save_dictionary_as_json_file(configuration, os.path.join(self._experiment_directory_path, f'{task_type}.params'))

  def _save_dictionary_as_json_file(self, dictionary, file_path):
    with open(file_path, 'w') as f: f.write(json.dumps(dictionary, default=lambda o: repr(o), indent=True, cls=NumpyEncoder))