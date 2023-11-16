"""
Implementation of the HSM (hierarchical state machine) or
NFSM (nested finite state machine) C++ example from
http://www.eventhelix.com/RealtimeMantra/HierarchicalStateMachine.htm#.VwqLVEL950w
in Python

- single source 'message type' for state transition changes
- message type considered, messages (comment) not considered to avoid complexity
"""

class UnsupportedMessageType(BaseException):
    pass

class UnsupportedState(BaseException):
    pass

class UnsupportedTransition(BaseException):
    pass

class HierachicalStateMachine:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._active_state = Active(self)
        self._standby_state = Standby(self)
        self._suspect_state = Suspect(self)
        self._failed_state = Failed(self)
        self._current_state = self._standby_state
        self.states = {'active': self._active_state, 'standby': self._standby_state, 'suspect': self._suspect_state, 'failed': self._failed_state}
        self.message_types = {'fault trigger': self._current_state.on_fault_trigger, 'switchover': self._current_state.on_switchover, 'diagnostics passed': self._current_state.on_diagnostics_passed, 'diagnostics failed': self._current_state.on_diagnostics_failed, 'operator inservice': self._current_state.on_operator_inservice}

    def _next_state(self, state):
        if False:
            for i in range(10):
                print('nop')
        try:
            self._current_state = self.states[state]
        except KeyError:
            raise UnsupportedState

    def _send_diagnostics_request(self):
        if False:
            print('Hello World!')
        return 'send diagnostic request'

    def _raise_alarm(self):
        if False:
            while True:
                i = 10
        return 'raise alarm'

    def _clear_alarm(self):
        if False:
            while True:
                i = 10
        return 'clear alarm'

    def _perform_switchover(self):
        if False:
            return 10
        return 'perform switchover'

    def _send_switchover_response(self):
        if False:
            while True:
                i = 10
        return 'send switchover response'

    def _send_operator_inservice_response(self):
        if False:
            return 10
        return 'send operator inservice response'

    def _send_diagnostics_failure_report(self):
        if False:
            print('Hello World!')
        return 'send diagnostics failure report'

    def _send_diagnostics_pass_report(self):
        if False:
            print('Hello World!')
        return 'send diagnostics pass report'

    def _abort_diagnostics(self):
        if False:
            print('Hello World!')
        return 'abort diagnostics'

    def _check_mate_status(self):
        if False:
            while True:
                i = 10
        return 'check mate status'

    def on_message(self, message_type):
        if False:
            print('Hello World!')
        if message_type in self.message_types.keys():
            self.message_types[message_type]()
        else:
            raise UnsupportedMessageType

class Unit:

    def __init__(self, HierachicalStateMachine):
        if False:
            for i in range(10):
                print('nop')
        self.hsm = HierachicalStateMachine

    def on_switchover(self):
        if False:
            i = 10
            return i + 15
        raise UnsupportedTransition

    def on_fault_trigger(self):
        if False:
            i = 10
            return i + 15
        raise UnsupportedTransition

    def on_diagnostics_failed(self):
        if False:
            return 10
        raise UnsupportedTransition

    def on_diagnostics_passed(self):
        if False:
            return 10
        raise UnsupportedTransition

    def on_operator_inservice(self):
        if False:
            print('Hello World!')
        raise UnsupportedTransition

class Inservice(Unit):

    def __init__(self, HierachicalStateMachine):
        if False:
            i = 10
            return i + 15
        self._hsm = HierachicalStateMachine

    def on_fault_trigger(self):
        if False:
            for i in range(10):
                print('nop')
        self._hsm._next_state('suspect')
        self._hsm._send_diagnostics_request()
        self._hsm._raise_alarm()

    def on_switchover(self):
        if False:
            i = 10
            return i + 15
        self._hsm._perform_switchover()
        self._hsm._check_mate_status()
        self._hsm._send_switchover_response()

class Active(Inservice):

    def __init__(self, HierachicalStateMachine):
        if False:
            while True:
                i = 10
        self._hsm = HierachicalStateMachine

    def on_fault_trigger(self):
        if False:
            i = 10
            return i + 15
        super().perform_switchover()
        super().on_fault_trigger()

    def on_switchover(self):
        if False:
            print('Hello World!')
        self._hsm.on_switchover()
        self._hsm.next_state('standby')

class Standby(Inservice):

    def __init__(self, HierachicalStateMachine):
        if False:
            for i in range(10):
                print('nop')
        self._hsm = HierachicalStateMachine

    def on_switchover(self):
        if False:
            i = 10
            return i + 15
        super().on_switchover()
        self._hsm._next_state('active')

class OutOfService(Unit):

    def __init__(self, HierachicalStateMachine):
        if False:
            for i in range(10):
                print('nop')
        self._hsm = HierachicalStateMachine

    def on_operator_inservice(self):
        if False:
            while True:
                i = 10
        self._hsm.on_switchover()
        self._hsm.send_operator_inservice_response()
        self._hsm.next_state('suspect')

class Suspect(OutOfService):

    def __init__(self, HierachicalStateMachine):
        if False:
            for i in range(10):
                print('nop')
        self._hsm = HierachicalStateMachine

    def on_diagnostics_failed(self):
        if False:
            while True:
                i = 10
        super().send_diagnostics_failure_report()
        super().next_state('failed')

    def on_diagnostics_passed(self):
        if False:
            return 10
        super().send_diagnostics_pass_report()
        super().clear_alarm()
        super().next_state('standby')

    def on_operator_inservice(self):
        if False:
            return 10
        super().abort_diagnostics()
        super().on_operator_inservice()

class Failed(OutOfService):
    """No need to override any method."""

    def __init__(self, HierachicalStateMachine):
        if False:
            while True:
                i = 10
        self._hsm = HierachicalStateMachine