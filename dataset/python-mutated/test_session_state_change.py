from sqlalchemy import exc as sa_exc
from sqlalchemy.orm import state_changes
from sqlalchemy.testing import eq_
from sqlalchemy.testing import expect_raises_message
from sqlalchemy.testing import fixtures

class StateTestChange(state_changes._StateChangeState):
    a = 1
    b = 2
    c = 3

class StateMachineTest(fixtures.TestBase):

    def test_single_change(self):
        if False:
            return 10
        'test single method that declares and invokes a state change'
        _NO_CHANGE = state_changes._StateChangeStates.NO_CHANGE

        class Machine(state_changes._StateChange):

            @state_changes._StateChange.declare_states((StateTestChange.a, _NO_CHANGE), StateTestChange.b)
            def move_to_b(self):
                if False:
                    return 10
                self._state = StateTestChange.b
        m = Machine()
        eq_(m._state, _NO_CHANGE)
        m.move_to_b()
        eq_(m._state, StateTestChange.b)

    def test_single_incorrect_change(self):
        if False:
            i = 10
            return i + 15
        'test single method that declares a state change but changes to the\n        wrong state.'
        _NO_CHANGE = state_changes._StateChangeStates.NO_CHANGE

        class Machine(state_changes._StateChange):

            @state_changes._StateChange.declare_states((StateTestChange.a, _NO_CHANGE), StateTestChange.b)
            def move_to_b(self):
                if False:
                    for i in range(10):
                        print('nop')
                self._state = StateTestChange.c
        m = Machine()
        eq_(m._state, _NO_CHANGE)
        with expect_raises_message(sa_exc.IllegalStateChangeError, "Method 'move_to_b\\(\\)' caused an unexpected state change to <StateTestChange.c: 3>"):
            m.move_to_b()

    def test_single_failed_to_change(self):
        if False:
            return 10
        "test single method that declares a state change but didn't do\n        the change."
        _NO_CHANGE = state_changes._StateChangeStates.NO_CHANGE

        class Machine(state_changes._StateChange):

            @state_changes._StateChange.declare_states((StateTestChange.a, _NO_CHANGE), StateTestChange.b)
            def move_to_b(self):
                if False:
                    while True:
                        i = 10
                pass
        m = Machine()
        eq_(m._state, _NO_CHANGE)
        with expect_raises_message(sa_exc.IllegalStateChangeError, "Method 'move_to_b\\(\\)' failed to change state to <StateTestChange.b: 2> as expected"):
            m.move_to_b()

    def test_change_from_sub_method_with_declaration(self):
        if False:
            print('Hello World!')
        'test successful state change by one method calling another that\n        does the change.\n\n        '
        _NO_CHANGE = state_changes._StateChangeStates.NO_CHANGE

        class Machine(state_changes._StateChange):

            @state_changes._StateChange.declare_states((StateTestChange.a, _NO_CHANGE), StateTestChange.b)
            def _inner_move_to_b(self):
                if False:
                    while True:
                        i = 10
                self._state = StateTestChange.b

            @state_changes._StateChange.declare_states((StateTestChange.a, _NO_CHANGE), StateTestChange.b)
            def move_to_b(self):
                if False:
                    for i in range(10):
                        print('nop')
                with self._expect_state(StateTestChange.b):
                    self._inner_move_to_b()
        m = Machine()
        eq_(m._state, _NO_CHANGE)
        m.move_to_b()
        eq_(m._state, StateTestChange.b)

    def test_method_and_sub_method_no_change(self):
        if False:
            while True:
                i = 10
        'test methods that declare the state should not change'
        _NO_CHANGE = state_changes._StateChangeStates.NO_CHANGE

        class Machine(state_changes._StateChange):

            @state_changes._StateChange.declare_states((StateTestChange.a,), _NO_CHANGE)
            def _inner_do_nothing(self):
                if False:
                    return 10
                pass

            @state_changes._StateChange.declare_states((StateTestChange.a,), _NO_CHANGE)
            def do_nothing(self):
                if False:
                    return 10
                self._inner_do_nothing()
        m = Machine()
        eq_(m._state, _NO_CHANGE)
        m._state = StateTestChange.a
        m.do_nothing()
        eq_(m._state, StateTestChange.a)

    def test_method_w_no_change_illegal_inner_change(self):
        if False:
            return 10
        _NO_CHANGE = state_changes._StateChangeStates.NO_CHANGE

        class Machine(state_changes._StateChange):

            @state_changes._StateChange.declare_states((StateTestChange.a, _NO_CHANGE), StateTestChange.c)
            def _inner_move_to_c(self):
                if False:
                    i = 10
                    return i + 15
                self._state = StateTestChange.c

            @state_changes._StateChange.declare_states((StateTestChange.a,), _NO_CHANGE)
            def do_nothing(self):
                if False:
                    i = 10
                    return i + 15
                self._inner_move_to_c()
        m = Machine()
        eq_(m._state, _NO_CHANGE)
        m._state = StateTestChange.a
        with expect_raises_message(sa_exc.IllegalStateChangeError, "Method '_inner_move_to_c\\(\\)' can't be called here; method 'do_nothing\\(\\)' is already in progress and this would cause an unexpected state change to <StateTestChange.c: 3>"):
            m.do_nothing()
        eq_(m._state, StateTestChange.a)

    def test_change_from_method_sub_w_no_change(self):
        if False:
            return 10
        'test methods that declare the state should not change'
        _NO_CHANGE = state_changes._StateChangeStates.NO_CHANGE

        class Machine(state_changes._StateChange):

            @state_changes._StateChange.declare_states((StateTestChange.a,), _NO_CHANGE)
            def _inner_do_nothing(self):
                if False:
                    print('Hello World!')
                pass

            @state_changes._StateChange.declare_states((StateTestChange.a,), StateTestChange.b)
            def move_to_b(self):
                if False:
                    i = 10
                    return i + 15
                self._inner_do_nothing()
                self._state = StateTestChange.b
        m = Machine()
        eq_(m._state, _NO_CHANGE)
        m._state = StateTestChange.a
        m.move_to_b()
        eq_(m._state, StateTestChange.b)

    def test_invalid_change_from_declared_sub_method_with_declaration(self):
        if False:
            for i in range(10):
                print('nop')
        'A method uses _expect_state() to call a sub-method, which must\n        declare that state as its destination if no exceptions are raised.\n\n        '
        _NO_CHANGE = state_changes._StateChangeStates.NO_CHANGE

        class Machine(state_changes._StateChange):

            @state_changes._StateChange.declare_states((StateTestChange.a, _NO_CHANGE), StateTestChange.c)
            def _inner_move_to_c(self):
                if False:
                    while True:
                        i = 10
                self._state = StateTestChange.c

            @state_changes._StateChange.declare_states((StateTestChange.a, _NO_CHANGE), StateTestChange.b)
            def move_to_b(self):
                if False:
                    for i in range(10):
                        print('nop')
                with self._expect_state(StateTestChange.b):
                    self._inner_move_to_c()
        m = Machine()
        eq_(m._state, _NO_CHANGE)
        with expect_raises_message(sa_exc.IllegalStateChangeError, "Cant run operation '_inner_move_to_c\\(\\)' here; will move to state <StateTestChange.c: 3> where we are expecting <StateTestChange.b: 2>"):
            m.move_to_b()

    def test_invalid_change_from_invalid_sub_method_with_declaration(self):
        if False:
            for i in range(10):
                print('nop')
        "A method uses _expect_state() to call a sub-method, which must\n        declare that state as its destination if no exceptions are raised.\n\n        Test an error is raised if the sub-method doesn't change to the\n        correct state.\n\n        "
        _NO_CHANGE = state_changes._StateChangeStates.NO_CHANGE

        class Machine(state_changes._StateChange):

            @state_changes._StateChange.declare_states((StateTestChange.a, _NO_CHANGE), StateTestChange.b)
            def _inner_move_to_c(self):
                if False:
                    for i in range(10):
                        print('nop')
                self._state = StateTestChange.c

            @state_changes._StateChange.declare_states((StateTestChange.a, _NO_CHANGE), StateTestChange.b)
            def move_to_b(self):
                if False:
                    print('Hello World!')
                with self._expect_state(StateTestChange.b):
                    self._inner_move_to_c()
        m = Machine()
        eq_(m._state, _NO_CHANGE)
        with expect_raises_message(sa_exc.IllegalStateChangeError, "While method 'move_to_b\\(\\)' was running, method '_inner_move_to_c\\(\\)' caused an unexpected state change to <StateTestChange.c: 3>"):
            m.move_to_b()

    def test_invalid_prereq_state(self):
        if False:
            for i in range(10):
                print('nop')
        _NO_CHANGE = state_changes._StateChangeStates.NO_CHANGE

        class Machine(state_changes._StateChange):

            @state_changes._StateChange.declare_states((StateTestChange.a, _NO_CHANGE), StateTestChange.b)
            def move_to_b(self):
                if False:
                    return 10
                self._state = StateTestChange.b

            @state_changes._StateChange.declare_states((StateTestChange.c,), 'd')
            def move_to_d(self):
                if False:
                    print('Hello World!')
                self._state = 'd'
        m = Machine()
        eq_(m._state, _NO_CHANGE)
        m.move_to_b()
        eq_(m._state, StateTestChange.b)
        with expect_raises_message(sa_exc.IllegalStateChangeError, "Can't run operation 'move_to_d\\(\\)' when Session is in state <StateTestChange.b: 2>"):
            m.move_to_d()

    def test_declare_only(self):
        if False:
            while True:
                i = 10
        _NO_CHANGE = state_changes._StateChangeStates.NO_CHANGE

        class Machine(state_changes._StateChange):

            @state_changes._StateChange.declare_states(state_changes._StateChangeStates.ANY, StateTestChange.b)
            def _inner_move_to_b(self):
                if False:
                    print('Hello World!')
                self._state = StateTestChange.b

            def move_to_b(self):
                if False:
                    return 10
                with self._expect_state(StateTestChange.b):
                    self._move_to_b()
        m = Machine()
        eq_(m._state, _NO_CHANGE)
        with expect_raises_message(AssertionError, 'Unexpected call to _expect_state outside of state-changing method'):
            m.move_to_b()

    def test_sibling_calls_maintain_correct_state(self):
        if False:
            return 10
        _NO_CHANGE = state_changes._StateChangeStates.NO_CHANGE

        class Machine(state_changes._StateChange):

            @state_changes._StateChange.declare_states(state_changes._StateChangeStates.ANY, StateTestChange.c)
            def move_to_c(self):
                if False:
                    while True:
                        i = 10
                self._state = StateTestChange.c

            @state_changes._StateChange.declare_states(state_changes._StateChangeStates.ANY, _NO_CHANGE)
            def do_nothing(self):
                if False:
                    for i in range(10):
                        print('nop')
                pass
        m = Machine()
        m.do_nothing()
        eq_(m._state, _NO_CHANGE)
        m.move_to_c()
        eq_(m._state, StateTestChange.c)

    def test_change_from_sub_method_requires_declaration(self):
        if False:
            i = 10
            return i + 15
        "A method can't call another state-changing method without using\n        _expect_state() to allow the state change to occur.\n\n        "
        _NO_CHANGE = state_changes._StateChangeStates.NO_CHANGE

        class Machine(state_changes._StateChange):

            @state_changes._StateChange.declare_states((StateTestChange.a, _NO_CHANGE), StateTestChange.b)
            def _inner_move_to_b(self):
                if False:
                    print('Hello World!')
                self._state = StateTestChange.b

            @state_changes._StateChange.declare_states((StateTestChange.a, _NO_CHANGE), StateTestChange.b)
            def move_to_b(self):
                if False:
                    print('Hello World!')
                self._inner_move_to_b()
        m = Machine()
        with expect_raises_message(sa_exc.IllegalStateChangeError, "Method '_inner_move_to_b\\(\\)' can't be called here; method 'move_to_b\\(\\)' is already in progress and this would cause an unexpected state change to <StateTestChange.b: 2>"):
            m.move_to_b()