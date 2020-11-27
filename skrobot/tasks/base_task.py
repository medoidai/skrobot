import copy

from stringcase import snakecase

from abc import ABC, abstractmethod

class BaseTask(ABC):
  """
  The :class:`.BaseTask` is an abstract base class for implementing tasks.

  A task is a configurable and reproducible piece of code built on top of scikit-learn that can be used in machine learning pipelines.
  """
  def __init__ (self, type_name, args):
    """
    This is the constructor method and can be used from child :class:`.BaseTask` implementations.

    :param type_name: The task's type name. A common practice is to pass the name of the task's class.
    :type type_name: str

    :param args: The task's parameters. A common practice is to pass the parameters at the time of task's object creation. It is a dictionary of key-value pairs, where the key is the parameter name and the value is the parameter value.
    :type args: dict
    """

    self.arguments = {}

    self._update_arguments({'type' : snakecase(type_name) })

    self._update_arguments(args)

    super(BaseTask, self).__init__()

  def get_type(self):
    """
    Get the task's type name.

    :return: The task's type name.
    :rtype: str
    """
    return self.type

  def get_configuration(self):
    """
    Get the task's parameters.

    :return: The task's parameters as a dictionary of key-value pairs, where the key is the parameter name and the value is the parameter value.
    :rtype: dict
    """
    return copy.deepcopy(self.arguments)

  @abstractmethod
  def run(self, output_directory):
    """
    An abstract method for running the task.

    :param output_directory: The output directory path under which task-related generated files are stored.
    :type output_directory: str
    """
    pass

  def _update_arguments (self, args):
    arguments = self._filter_arguments(args)

    self.arguments.update(arguments)

    self.__dict__.update(arguments)

  def _filter_arguments(self, args):
    return copy.deepcopy({ k:v for k,v in args.items() if not k.startswith("__") and not k == "self" and not k.endswith("__") })