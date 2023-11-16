from typing import List
from trashcli.restore.output import Output
from trashcli.restore.output_event import Quit, Die, Println, OutputEvent

class OutputRecorder(Output):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.events = []

    def quit(self):
        if False:
            return 10
        self.append_event(Quit())

    def die(self, msg):
        if False:
            return 10
        self.append_event(Die(msg))

    def println(self, msg):
        if False:
            return 10
        self.append_event(Println(msg))

    def append_event(self, event):
        if False:
            for i in range(10):
                print('nop')
        self.events.append(event)

    def apply_to(self, output):
        if False:
            while True:
                i = 10
        for event in self.events:
            output.append_event(event)