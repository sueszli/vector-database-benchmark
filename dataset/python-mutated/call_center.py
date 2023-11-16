from abc import ABCMeta, abstractmethod
from collections import deque
from enum import Enum

class Rank(Enum):
    OPERATOR = 0
    SUPERVISOR = 1
    DIRECTOR = 2

class Employee(metaclass=ABCMeta):

    def __init__(self, employee_id, name, rank, call_center):
        if False:
            print('Hello World!')
        self.employee_id = employee_id
        self.name = name
        self.rank = rank
        self.call = None
        self.call_center = call_center

    def take_call(self, call):
        if False:
            print('Hello World!')
        'Assume the employee will always successfully take the call.'
        self.call = call
        self.call.employee = self
        self.call.state = CallState.IN_PROGRESS

    def complete_call(self):
        if False:
            return 10
        self.call.state = CallState.COMPLETE
        self.call_center.notify_call_completed(self.call)

    @abstractmethod
    def escalate_call(self):
        if False:
            while True:
                i = 10
        pass

    def _escalate_call(self):
        if False:
            print('Hello World!')
        self.call.state = CallState.READY
        call = self.call
        self.call = None
        self.call_center.notify_call_escalated(call)

class Operator(Employee):

    def __init__(self, employee_id, name):
        if False:
            i = 10
            return i + 15
        super(Operator, self).__init__(employee_id, name, Rank.OPERATOR)

    def escalate_call(self):
        if False:
            print('Hello World!')
        self.call.level = Rank.SUPERVISOR
        self._escalate_call()

class Supervisor(Employee):

    def __init__(self, employee_id, name):
        if False:
            while True:
                i = 10
        super(Operator, self).__init__(employee_id, name, Rank.SUPERVISOR)

    def escalate_call(self):
        if False:
            print('Hello World!')
        self.call.level = Rank.DIRECTOR
        self._escalate_call()

class Director(Employee):

    def __init__(self, employee_id, name):
        if False:
            i = 10
            return i + 15
        super(Operator, self).__init__(employee_id, name, Rank.DIRECTOR)

    def escalate_call(self):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError('Directors must be able to handle any call')

class CallState(Enum):
    READY = 0
    IN_PROGRESS = 1
    COMPLETE = 2

class Call(object):

    def __init__(self, rank):
        if False:
            return 10
        self.state = CallState.READY
        self.rank = rank
        self.employee = None

class CallCenter(object):

    def __init__(self, operators, supervisors, directors):
        if False:
            for i in range(10):
                print('nop')
        self.operators = operators
        self.supervisors = supervisors
        self.directors = directors
        self.queued_calls = deque()

    def dispatch_call(self, call):
        if False:
            return 10
        if call.rank not in (Rank.OPERATOR, Rank.SUPERVISOR, Rank.DIRECTOR):
            raise ValueError('Invalid call rank: {}'.format(call.rank))
        employee = None
        if call.rank == Rank.OPERATOR:
            employee = self._dispatch_call(call, self.operators)
        if call.rank == Rank.SUPERVISOR or employee is None:
            employee = self._dispatch_call(call, self.supervisors)
        if call.rank == Rank.DIRECTOR or employee is None:
            employee = self._dispatch_call(call, self.directors)
        if employee is None:
            self.queued_calls.append(call)

    def _dispatch_call(self, call, employees):
        if False:
            return 10
        for employee in employees:
            if employee.call is None:
                employee.take_call(call)
                return employee
        return None

    def notify_call_escalated(self, call):
        if False:
            for i in range(10):
                print('nop')
        pass

    def notify_call_completed(self, call):
        if False:
            return 10
        pass

    def dispatch_queued_call_to_newly_freed_employee(self, call, employee):
        if False:
            while True:
                i = 10
        pass