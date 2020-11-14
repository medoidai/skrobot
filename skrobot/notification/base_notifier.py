from abc import ABC, abstractmethod

class BaseNotifier(ABC):
    """
    All notifiers inherit from the :class:`.BaseNotifier`. A notifier can be used to send success / failure notifications for tasks execution.
    """
    @abstractmethod
    def notify(self, message):
        """
        The method that must be implemented by the child notifiers.
	
        :param message: The message of success or failure the notifier will send to observers.
        :type message: str
        """
        pass