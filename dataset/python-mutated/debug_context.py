"""
Information and debug options for a specific runtime.
"""

class DebugContext:

    def __init__(self, debug_ports=None, debugger_path=None, debug_args=None, debug_function=None, container_env_vars=None):
        if False:
            return 10
        '\n        Initialize the Debug Context with Lambda debugger options\n\n        :param tuple(int) debug_ports: Collection of debugger ports to be exposed from a docker container\n        :param Path debugger_path: Path to a debugger to be launched\n        :param string debug_args: Additional arguments to be passed to the debugger\n        :param string debug_function: The Lambda function logicalId that will have the debugging options enabled in case\n        of warm containers option is enabled\n        :param dict container_env_vars: Additional environmental variables to be set.\n        '
        self.debug_ports = debug_ports
        self.debugger_path = debugger_path
        self.debug_args = debug_args
        self.debug_function = debug_function
        self.container_env_vars = container_env_vars

    def __bool__(self):
        if False:
            for i in range(10):
                print('nop')
        return bool(self.debug_ports)

    def __nonzero__(self):
        if False:
            print('Hello World!')
        return self.__bool__()