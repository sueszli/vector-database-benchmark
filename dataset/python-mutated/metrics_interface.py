"""Python TFLite metrics helper interface."""
import abc

class TFLiteMetricsInterface(metaclass=abc.ABCMeta):
    """Abstract class for TFLiteMetrics."""

    @abc.abstractmethod
    def increase_counter_debugger_creation(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    @abc.abstractmethod
    def increase_counter_interpreter_creation(self):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    @abc.abstractmethod
    def increase_counter_converter_attempt(self):
        if False:
            print('Hello World!')
        raise NotImplementedError

    @abc.abstractmethod
    def increase_counter_converter_success(self):
        if False:
            return 10
        raise NotImplementedError

    @abc.abstractmethod
    def set_converter_param(self, name, value):
        if False:
            print('Hello World!')
        raise NotImplementedError

    @abc.abstractmethod
    def set_converter_error(self, error_data):
        if False:
            print('Hello World!')
        raise NotImplementedError

    @abc.abstractmethod
    def set_converter_latency(self, value):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError