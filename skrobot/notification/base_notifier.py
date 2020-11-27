from abc import ABC, abstractmethod

class BaseNotifier(ABC):
    """
    The :class:`.BaseNotifier` is an abstract base class for implementing notifiers.

    A notifier can be used to send notifications.
    """
    @abstractmethod
    def notify(self, message):
        """
        An abstract method for sending the notification.

        :param message: The notification's message.
        :type message: str
        """
        pass