from __future__ import print_function
import six
from trashcli.restore.output import Output
from trashcli.restore.output_event import Println, Die, Quit, Exiting, OutputEvent

class RealOutput(Output):

    def __init__(self, stdout, stderr, exit):
        if False:
            print('Hello World!')
        self.stdout = stdout
        self.stderr = stderr
        self.exit = exit

    def quit(self):
        if False:
            while True:
                i = 10
        self.die('')

    def printerr(self, msg):
        if False:
            print('Hello World!')
        print(six.text_type(msg), file=self.stderr)

    def println(self, line):
        if False:
            return 10
        print(six.text_type(line), file=self.stdout)

    def die(self, error):
        if False:
            while True:
                i = 10
        self.printerr(error)
        self.exit(1)

    def append_event(self, event):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(event, Println):
            self.println(event.msg)
        elif isinstance(event, Die):
            self.die(event.msg)
        elif isinstance(event, Quit):
            self.quit()
        elif isinstance(event, Exiting):
            self.println('Exiting')
        else:
            raise Exception('Unknown call %s' % event)