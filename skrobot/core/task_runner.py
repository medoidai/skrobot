import json, os

from numpyencoder import NumpyEncoder

class TaskRunner:
  """
  The :class:`.TaskRunner` class is a simplified version (in functionality) of the :class:`.Experiment` class.

  It leaves out all the "experiment" stuff and is focused mostly in the execution and tracking of :class:`.BaseTask` tasks.
  """

  def __init__ (self, output_directory_path):
    """
    This is the constructor method and can be used to create a new object instance of :class:`.TaskRunner` class.

    :param output_directory_path: The output directory path under which task-related generated files are stored.
    :type output_directory_path: str
    """

    self._output_directory_path = output_directory_path

    os.makedirs(self._output_directory_path)

  def run(self, task):
    """
    Run a :class:`.BaseTask` task.

    When running a task, its recorded parameters (e.g., *train_task.params*) and any other task-related generated files are stored under output directory for tracking reasons.

    The task's recorded parameters are in JSON format.

    Lastly, in case an exception occurs, a text file (e.g., *train_task.errors*) is generated under output directory containing the error message.

    :param task: The task to run.
    :type task: :class:`.BaseTask`

    :return: The task's result.
    :rtype: Depends on the ``task`` parameter.
    """

    task_type = task.get_type()

    try:
      self._save_configuration_file(task.get_configuration(), task_type)

      return task.run(self._output_directory_path)
    except Exception as exception:
      self._save_errors_file(exception, task_type)

      raise

  def _save_errors_file(self, exception, task_type):
    with open(os.path.join(self._output_directory_path, f'{task_type}.errors'), 'w') as f: f.write(repr(exception))

  def _save_configuration_file(self, configuration, task_type):
    self._save_dictionary_as_json_file(configuration, os.path.join(self._output_directory_path, f'{task_type}.params'))

  def _save_dictionary_as_json_file(self, dictionary, file_path):
    with open(file_path, 'w') as f: f.write(json.dumps(dictionary, default=lambda o: repr(o), indent=True, cls=NumpyEncoder))