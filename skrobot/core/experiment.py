import json, os, uuid, datetime, shutil

from numpyencoder import NumpyEncoder

from ..notification import BaseNotifier

class Experiment:
  def __init__ (self, experiments_repository):
    self._experiments_repository = experiments_repository

    self._experimenter = 'anonymous'

    self._source_code_file_path = None

    self._notifier = None

  def set_notifier(self, notifier : BaseNotifier):
    self._notifier = notifier

    return self

  def set_source_code_file_path(self, source_code_file_path):
    self._source_code_file_path = source_code_file_path

    return self

  def set_experimenter(self, experimenter):
    self._experimenter = experimenter

    return self

  def build(self):
    self._create_experiment_log()

    self._create_experiment_directory()

    self._save_experiment_log_file()

    self._save_source_code_file()

    return self

  def run(self, task):
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