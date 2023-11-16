from robot.libraries.BuiltIn import BuiltIn

class LogLevels:
    ROBOT_LISTENER_API_VERSION = 2

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.ROBOT_LIBRARY_LISTENER = self
        self.messages = []

    def _log_message(self, msg):
        if False:
            i = 10
            return i + 15
        self.messages.append('%s: %s' % (msg['level'], msg['message']))

    def logged_messages_should_be(self, *expected):
        if False:
            return 10
        BuiltIn().should_be_equal('\n'.join(self.messages), '\n'.join(expected))