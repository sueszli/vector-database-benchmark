from azureml.core.run import Run
from .base_channel import BaseChannel
from .log_utils import LogType, nni_log

class AMLChannel(BaseChannel):

    def __init__(self, args):
        if False:
            while True:
                i = 10
        self.args = args
        self.run = Run.get_context()
        super(AMLChannel, self).__init__(args)
        self.current_message_index = -1

    def _inner_open(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def _inner_close(self):
        if False:
            i = 10
            return i + 15
        pass

    def _inner_send(self, message):
        if False:
            print('Hello World!')
        try:
            self.run.log('trial_runner', message.decode('utf8'))
        except Exception as exception:
            nni_log(LogType.Error, 'meet unhandled exception when send message: %s' % exception)

    def _inner_receive(self):
        if False:
            return 10
        messages = []
        message_dict = self.run.get_metrics()
        if 'nni_manager' not in message_dict:
            return []
        message_list = message_dict['nni_manager']
        if not message_list:
            return messages
        if type(message_list) is list:
            if self.current_message_index < len(message_list) - 1:
                messages = message_list[self.current_message_index + 1:len(message_list)]
                self.current_message_index = len(message_list) - 1
        elif self.current_message_index == -1:
            messages = [message_list]
            self.current_message_index += 1
        newMessage = []
        for message in messages:
            newMessage.append(message.encode('utf8'))
        return newMessage