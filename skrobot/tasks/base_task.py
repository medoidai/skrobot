import copy

from stringcase import snakecase

from abc import ABC, abstractmethod

class BaseTask(ABC):
  def __init__ (self, type_name, args):
    self.arguments = {}

    self.update_arguments({'type' : snakecase(type_name) })

    self.update_arguments(args)

    super(BaseTask, self).__init__()

  def get_type(self):
    return self.type

  def update_arguments (self, args):
    arguments = self.filter_arguments(args)

    self.arguments.update(arguments)

    self.__dict__.update(arguments)

  def filter_arguments(self, args):
    return copy.deepcopy({ k:v for k,v in args.items() if not k.startswith("__") and not k == "self" and not k.endswith("__") })

  def get_configuration(self):
    return copy.deepcopy(self.arguments)

  @abstractmethod
  def run(self, output_directory):
    pass