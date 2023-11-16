import unittest
from manticore.wasm import ManticoreWASM
from manticore.wasm.types import Value_t
from pathlib import Path
test_if_file = str(Path(__file__).parent.joinpath('inputs', 'if.wasm'))
test_br_if_file = str(Path(__file__).parent.joinpath('inputs', 'br_if.wasm'))
test_br_table_file = str(Path(__file__).parent.joinpath('inputs', 'br_table.wasm'))
test_call_indirect_file = str(Path(__file__).parent.joinpath('inputs', 'call_indirect.wasm'))

class TestSymbolicBranch(unittest.TestCase):

    def can_terminate_with(self, val, state):
        if False:
            for i in range(10):
                print('nop')
        if str(state._terminated_by).startswith('Execution raised Trap'):
            return False
        stack = state.platform.stack
        assert stack.has_type_on_top(Value_t, 1)
        result = stack.peek()
        input_arg = state.context['arg']
        return state.can_be_true(result == val)

    def test_if(self):
        if False:
            print('Hello World!')
        m = ManticoreWASM(test_if_file)

        def arg_gen(state):
            if False:
                return 10
            arg = state.new_symbolic_value(32, 'arg')
            state.context['arg'] = arg
            return [arg]
        m.main(arg_gen)
        m.run()
        assert any((self.can_terminate_with(0, state) for state in m.terminated_states))

    def test_br_if(self):
        if False:
            while True:
                i = 10
        m = ManticoreWASM(test_br_if_file)

        def arg_gen(state):
            if False:
                while True:
                    i = 10
            arg = state.new_symbolic_value(32, 'arg')
            state.context['arg'] = arg
            return [arg]
        m.main(arg_gen)
        m.run()
        assert any((self.can_terminate_with(0, state) for state in m.terminated_states))

    def test_br_table(self):
        if False:
            i = 10
            return i + 15
        m = ManticoreWASM(test_br_table_file)

        def arg_gen(state):
            if False:
                print('Hello World!')
            arg = state.new_symbolic_value(32, 'arg')
            state.context['arg'] = arg
            return [arg]
        m.main(arg_gen)
        m.run()
        assert any((self.can_terminate_with(0, state) for state in m.terminated_states))
        assert any((self.can_terminate_with(1, state) for state in m.terminated_states))

    def test_call_indirect(self):
        if False:
            return 10
        m = ManticoreWASM(test_call_indirect_file)

        def arg_gen(state):
            if False:
                return 10
            arg = state.new_symbolic_value(32, 'arg')
            state.context['arg'] = arg
            return [arg]
        m.main(arg_gen)
        m.run()
        assert any((self.can_terminate_with(0, state) for state in m.terminated_states))
        assert any((self.can_terminate_with(1, state) for state in m.terminated_states))
if __name__ == '__main__':
    unittest.main()