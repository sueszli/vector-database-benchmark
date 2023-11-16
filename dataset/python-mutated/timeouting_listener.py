from robot.errors import TimeoutError

class timeouting_listener:
    ROBOT_LISTENER_API_VERSION = 2
    timeout = False

    def start_keyword(self, name, info):
        if False:
            for i in range(10):
                print('nop')
        self.timeout = name == 'BuiltIn.Log'

    def end_keyword(self, name, info):
        if False:
            print('Hello World!')
        self.timeout = False

    def log_message(self, message):
        if False:
            i = 10
            return i + 15
        if self.timeout:
            self.timeout = False
            raise TimeoutError('Emulated timeout inside log_message')