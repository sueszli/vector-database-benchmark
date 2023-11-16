import abc

class UserInputRequester(abc.ABC):
    """
    Base Class / Interface for requesting user input

    e.g. from the console
    """

    @abc.abstractmethod
    def ask(self, prompt: str) -> str:
        if False:
            while True:
                i = 10
        '\n        Ask the user for a text input, the input is not sensitive\n        and can be echoed to the user\n\n        :param prompt: message to display when asking for the input\n        :return: the value of the user input\n        '
        raise NotImplementedError

    @abc.abstractmethod
    def ask_password(self, prompt: str) -> str:
        if False:
            while True:
                i = 10
        '\n        Ask the user for a text input, the input _is_ sensitive\n        and should be masked as the user gives the input\n\n        :param prompt: message to display when asking for the input\n        :return: the value of the user input\n        '
        raise NotImplementedError