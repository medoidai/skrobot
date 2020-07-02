import json, os

from numpyencoder import NumpyEncoder

class MlTaskRunner:
  def __init__ (self, output_directory_path):
    self._output_directory_path = output_directory_path

  def run(self, ml_task):
    task_type = ml_task.get_type()

    try:
      self._save_configuration_file(ml_task.get_configuration(), task_type)

      return ml_task.run(self._output_directory_path)
    except Exception as exception:
      self._save_errors_file(exception, task_type)

      raise

  def _save_errors_file(self, exception, task_type):
    with open(os.path.join(self._output_directory_path, f'{task_type}.errors'), 'w') as f: f.write(repr(exception))

  def _save_configuration_file(self, configuration, task_type):
    self._save_dictionary_as_json_file(configuration, os.path.join(self._output_directory_path, f'{task_type}.params'))

  def _save_dictionary_as_json_file(self, dictionary, file_path):
    with open(file_path, 'w') as f: f.write(json.dumps(dictionary, default=lambda o: repr(o), indent=True, cls=NumpyEncoder))