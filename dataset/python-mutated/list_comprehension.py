from typing import List
from integration_test.taint import source, sink

class Sink:

    def run(self, command: str) -> str:
        if False:
            while True:
                i = 10
        sink(command)
        return ''

def take_input() -> None:
    if False:
        return 10
    sinks: List[Sink] = [Sink()]
    result = [s.run(source()) for s in sinks]

def inductive_comprehension_sink(arguments: List[str]) -> None:
    if False:
        while True:
            i = 10
    command = '  '.join((argument.lower() for argument in arguments))
    sink(command)

def eval_via_comprehension_sink() -> None:
    if False:
        return 10
    inductive_comprehension_sink(source())