from abc import ABC, abstractmethod

class BaseNotifier(ABC):
    """
    All notifiers inherit from the :class:`.BaseNotifier`. A notifier can be used to send success / failure notifications for tasks execution.
    """
    @abstractmethod
    def notify(self, message):
        pass