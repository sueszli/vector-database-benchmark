"""

Provider node failing the subtask

"""
import mock
from golem_messages.message.tasks import WantToComputeTask, TaskToCompute
from golem.task.tasksession import TaskSession
from golemapp import main
original_send = TaskSession.send
original_interpret = TaskSession.interpret
received_ttc = False

def send(self, msg, *args, **kwargs):
    if False:
        i = 10
        return i + 15
    if received_ttc and isinstance(msg, WantToComputeTask):
        return
    original_send(self, msg, *args, **kwargs)

def interpret(self, msg, *args, **kwargs):
    if False:
        while True:
            i = 10
    global received_ttc
    if isinstance(msg, TaskToCompute):
        received_ttc = True
    original_interpret(self, msg, *args, **kwargs)

@mock.patch('golem.task.tasksession.TaskSession.interpret', interpret)
@mock.patch('golem.task.tasksession.TaskSession.send', send)
def start_node(*_):
    if False:
        for i in range(10):
            print('nop')
    main()
start_node()