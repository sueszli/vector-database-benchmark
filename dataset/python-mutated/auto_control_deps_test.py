import itertools
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import auto_control_deps as acd
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_switch_case
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import gen_sendrecv_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import test
from tensorflow.python.training import adam
from tensorflow.python.training import momentum

class AutomaticControlDependenciesTest(test.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.must_run_order_insensitive_stateful_ops = acd.MUST_RUN_ORDER_INSENSITIVE_STATEFUL_OPS

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        acd.MUST_RUN_ORDER_INSENSITIVE_STATEFUL_OPS = self.must_run_order_insensitive_stateful_ops
        super().tearDown()

    def testBasic(self):
        if False:
            print('Hello World!')
        with context.graph_mode(), self.cached_session():
            v = resource_variable_ops.ResourceVariable(1.0)
            self.evaluate(variables.global_variables_initializer())
            with acd.AutomaticControlDependencies() as c:
                v.assign(v + 1)
                v.assign(2 * v)
                val = v.read_value()
                val = c.mark_as_return(val)
            self.assertAllEqual(val, 4.0)

    def testUnorderedOpsRunInParallel(self):
        if False:
            for i in range(10):
                print('nop')
        acd.MUST_RUN_ORDER_INSENSITIVE_STATEFUL_OPS |= frozenset(('EagerPyFunc',))
        side_effects = []

        def side_effect_one(x):
            if False:
                return 10
            side_effects.append(1)
            return x

        def side_effect_two(x):
            if False:
                while True:
                    i = 10
            side_effects.append(2)
            return x

        @def_function.function
        def f():
            if False:
                while True:
                    i = 10
            script_ops.eager_py_func(side_effect_one, [1], [dtypes.int32])
            script_ops.eager_py_func(side_effect_two, [1], [dtypes.int32])
            return 1
        side_effects = []
        self.evaluate(f())
        self.assertSetEqual(set(side_effects), set((1, 2)))

    def testIndependentOpsRunInParallel(self):
        if False:
            return 10
        v = resource_variable_ops.ResourceVariable(1)
        self.evaluate(variables.global_variables_initializer())

        @def_function.function
        def f():
            if False:
                for i in range(10):
                    print('nop')
            gen_resource_variable_ops.assign_variable_op(v.handle, 1)
            ops.get_default_graph().experimental_acd_manager.run_independently(gen_resource_variable_ops.assign_variable_op(v.handle, 2))
        var_values = set()
        for _ in range(10000):
            self.evaluate(f())
            var_values.add(self.evaluate(resource_variable_ops.read_variable_op(v.handle, dtypes.int32)))
        self.assertSetEqual(var_values, set((1, 2)))

    def testIndependentOpsInLoop(self):
        if False:
            while True:
                i = 10
        v = resource_variable_ops.ResourceVariable(0)
        self.evaluate(variables.global_variables_initializer())

        @def_function.function
        def f():
            if False:
                i = 10
                return i + 15
            for i in math_ops.range(3):
                ops.get_default_graph().experimental_acd_manager.run_independently(gen_resource_variable_ops.assign_variable_op(v.handle, i))
        self.evaluate(f())
        self.assertEqual(self.evaluate(resource_variable_ops.read_variable_op(v.handle, dtypes.int32)), 2)

    def testNoControlDepsBetweenVariableReads(self):
        if False:
            return 10
        with context.graph_mode(), self.cached_session():
            v = resource_variable_ops.ResourceVariable(1.0)
            self.evaluate(variables.global_variables_initializer())
            with acd.AutomaticControlDependencies():
                read_op1 = gen_resource_variable_ops.read_variable_op(v.handle, v.dtype).op
                read_op2 = gen_resource_variable_ops.read_variable_op(v.handle, v.dtype).op
                gen_resource_variable_ops.assign_variable_op(v.handle, v + 1)
            self.assertNotIn(read_op1, read_op2.control_inputs)
            self.assertNotIn(read_op2, read_op1.control_inputs)

    def testVariableReadThenWrite(self):
        if False:
            for i in range(10):
                print('nop')
        with context.graph_mode(), self.cached_session():
            v = resource_variable_ops.ResourceVariable(1.0)
            self.evaluate(variables.global_variables_initializer())
            with acd.AutomaticControlDependencies():
                read_op1 = gen_resource_variable_ops.read_variable_op(v.handle, v.dtype).op
                read_op2 = gen_resource_variable_ops.read_variable_op(v.handle, v.dtype).op
                assign_op = gen_resource_variable_ops.assign_variable_op(v.handle, v + 1)
            self.assertIn(read_op1, assign_op.control_inputs)
            self.assertIn(read_op2, assign_op.control_inputs)
            self.assertNotIn(read_op1, read_op2.control_inputs)
            self.assertNotIn(read_op2, read_op1.control_inputs)

    def testVariableWriteThenRead(self):
        if False:
            i = 10
            return i + 15
        with context.graph_mode(), self.cached_session():
            v = resource_variable_ops.ResourceVariable(1.0)
            self.evaluate(variables.global_variables_initializer())
            with acd.AutomaticControlDependencies():
                assign_op = gen_resource_variable_ops.assign_variable_op(v.handle, v + 1)
                read_op1 = gen_resource_variable_ops.read_variable_op(v.handle, v.dtype).op
                read_op2 = gen_resource_variable_ops.read_variable_op(v.handle, v.dtype).op
            self.assertIn(assign_op, read_op1.control_inputs)
            self.assertIn(assign_op, read_op2.control_inputs)
            self.assertNotIn(read_op1, read_op2.control_inputs)
            self.assertNotIn(read_op2, read_op1.control_inputs)

    def testIdentityPassThrough(self):
        if False:
            for i in range(10):
                print('nop')
        with context.graph_mode(), self.cached_session():
            v = resource_variable_ops.ResourceVariable(1.0)
            self.evaluate(variables.global_variables_initializer())
            with acd.AutomaticControlDependencies():
                gen_resource_variable_ops.assign_variable_op(v.handle, v + 1)
                identity_handle = gen_array_ops.identity(v.handle)
                assign_op2 = gen_resource_variable_ops.assign_variable_op(v.handle, v + 1)
                read_op = gen_resource_variable_ops.read_variable_op(identity_handle, v.dtype).op
            self.assertIn(assign_op2, read_op.control_inputs)

    def testVariableReadsInOpsWithMustRun(self):
        if False:
            return 10
        with context.graph_mode(), self.cached_session():
            v = resource_variable_ops.ResourceVariable(1.0)
            self.evaluate(variables.global_variables_initializer())
            with acd.AutomaticControlDependencies() as c:
                read_op = gen_resource_variable_ops.read_variable_op(v.handle, v.dtype).op
                c.mark_as_return(read_op.outputs[0])
            self.assertIn(read_op, c.ops_which_must_run)

    def testVariableMultipleReadsAndWrites(self):
        if False:
            return 10
        with context.graph_mode(), self.cached_session():
            v = resource_variable_ops.ResourceVariable(1.0)
            self.evaluate(variables.global_variables_initializer())
            with acd.AutomaticControlDependencies() as c:
                read_op1 = gen_resource_variable_ops.read_variable_op(v.handle, v.dtype).op
                read_op2 = gen_resource_variable_ops.read_variable_op(v.handle, v.dtype).op
                assign_op1 = gen_resource_variable_ops.assign_variable_op(v.handle, v + 1)
                assign_op2 = gen_resource_variable_ops.assign_variable_op(v.handle, v + 1)
                read_op3 = gen_resource_variable_ops.read_variable_op(v.handle, v.dtype).op
                read_op4 = gen_resource_variable_ops.read_variable_op(v.handle, v.dtype).op
                assign_op3 = gen_resource_variable_ops.assign_variable_op(v.handle, v + 1)
                assign_op4 = gen_resource_variable_ops.assign_variable_op(v.handle, v + 1)
                c.mark_as_return(read_op1.outputs[0])
                c.mark_as_return(read_op2.outputs[0])
                c.mark_as_return(read_op3.outputs[0])
                c.mark_as_return(read_op4.outputs[0])
            self.assertIn(read_op1, assign_op1.control_inputs)
            self.assertIn(read_op2, assign_op1.control_inputs)
            self.assertIn(assign_op1, assign_op2.control_inputs)
            self.assertIn(assign_op2, read_op3.control_inputs)
            self.assertIn(assign_op2, read_op4.control_inputs)
            self.assertIn(read_op3, assign_op3.control_inputs)
            self.assertIn(read_op4, assign_op3.control_inputs)
            self.assertIn(assign_op3, assign_op4.control_inputs)
            read_ops = [read_op1, read_op2, read_op3, read_op4]
            for (src_op, tgt_op) in itertools.product(read_ops, read_ops):
                self.assertNotIn(src_op, tgt_op.control_inputs)
            self.assertIn(read_op1, c.ops_which_must_run)
            self.assertIn(read_op2, c.ops_which_must_run)
            self.assertIn(read_op3, c.ops_which_must_run)
            self.assertIn(read_op4, c.ops_which_must_run)
            self.assertIn(assign_op4, c.ops_which_must_run)

    def testSendInOpsWithMustRun(self):
        if False:
            return 10
        with context.graph_mode(), self.cached_session():
            v = resource_variable_ops.ResourceVariable(1.0)
            self.evaluate(variables.global_variables_initializer())
            with acd.AutomaticControlDependencies() as c:
                send_op = gen_sendrecv_ops.send(v, 'x', '/', 0, '/')
            self.assertIn(send_op, c.ops_which_must_run)

    def _testVariableReadInFunctionalOp(self, build_functional_op, op_type):
        if False:
            for i in range(10):
                print('nop')
        v = resource_variable_ops.ResourceVariable(1.0)
        self.evaluate(variables.global_variables_initializer())

        @def_function.function
        def read_var_in_while():
            if False:
                i = 10
                return i + 15
            gen_resource_variable_ops.read_variable_op(v.handle, v.dtype, name='read1')
            result = build_functional_op(v)
            gen_resource_variable_ops.read_variable_op(v.handle, v.dtype, name='read2')
            gen_resource_variable_ops.assign_variable_op(v.handle, v + 1)
            return result
        func_graph = read_var_in_while.get_concrete_function().graph
        assert len(func_graph.inputs) == 1

        def get_op(op_type, sub_name):
            if False:
                i = 10
                return i + 15
            operations = [op for op in func_graph.get_operations() if op.type == op_type and sub_name in op.name]
            assert len(operations) == 1
            return operations[0]
        read1 = get_op('ReadVariableOp', 'read1')
        functional_op = get_op(op_type, '')
        read2 = get_op('ReadVariableOp', 'read2')
        assign_op = get_op('AssignVariableOp', '')
        self.assertNotIn(read1, functional_op.control_inputs)
        self.assertNotIn(functional_op, read2.control_inputs)
        self.assertIn(read1, assign_op.control_inputs)
        self.assertIn(read2, assign_op.control_inputs)
        self.assertIn(functional_op, assign_op.control_inputs)

    def testVariableReadInWhileLoop(self):
        if False:
            print('Hello World!')

        def build_functional_op(v):
            if False:
                print('Hello World!')

            def body(_):
                if False:
                    return 10
                return gen_resource_variable_ops.read_variable_op(v.handle, v.dtype)
            return while_loop.while_loop(lambda i: True, body, [0.0], maximum_iterations=1)
        self._testVariableReadInFunctionalOp(build_functional_op, 'While')

    def testVariableReadInCondTrueBranch(self):
        if False:
            return 10

        def build_functional_op(v):
            if False:
                while True:
                    i = 10

            def then_branch():
                if False:
                    for i in range(10):
                        print('nop')
                return gen_resource_variable_ops.read_variable_op(v.handle, v.dtype)

            def else_branch():
                if False:
                    while True:
                        i = 10
                return array_ops.zeros([], v.dtype)
            return cond.cond(constant_op.constant(True), then_branch, else_branch)
        self._testVariableReadInFunctionalOp(build_functional_op, 'If')

    def testVariableReadInCondFalseBranch(self):
        if False:
            while True:
                i = 10

        def build_functional_op(v):
            if False:
                print('Hello World!')

            def then_branch():
                if False:
                    for i in range(10):
                        print('nop')
                return array_ops.zeros([], v.dtype)

            def else_branch():
                if False:
                    return 10
                return gen_resource_variable_ops.read_variable_op(v.handle, v.dtype)
            return cond.cond(constant_op.constant(False), then_branch, else_branch)
        self._testVariableReadInFunctionalOp(build_functional_op, 'If')

    def testVariableReadInCaseBranch0(self):
        if False:
            while True:
                i = 10

        def build_functional_op(v):
            if False:
                return 10

            def branch0():
                if False:
                    print('Hello World!')
                return gen_resource_variable_ops.read_variable_op(v.handle, v.dtype)

            def branch1():
                if False:
                    while True:
                        i = 10
                return array_ops.zeros([], v.dtype)
            return control_flow_switch_case.switch_case(constant_op.constant(0), [branch0, branch1])
        self._testVariableReadInFunctionalOp(build_functional_op, 'Case')

    def testVariableReadInCaseBranch1(self):
        if False:
            print('Hello World!')

        def build_functional_op(v):
            if False:
                i = 10
                return i + 15

            def branch0():
                if False:
                    return 10
                return array_ops.zeros([], v.dtype)

            def branch1():
                if False:
                    i = 10
                    return i + 15
                return gen_resource_variable_ops.read_variable_op(v.handle, v.dtype)
            return control_flow_switch_case.switch_case(constant_op.constant(0), [branch0, branch1])
        self._testVariableReadInFunctionalOp(build_functional_op, 'Case')

    def testVariableReadInFunction(self):
        if False:
            print('Hello World!')

        def build_functional_op(v):
            if False:
                print('Hello World!')

            @def_function.function
            def fn_with_read():
                if False:
                    for i in range(10):
                        print('nop')
                return gen_resource_variable_ops.read_variable_op(v.handle, v.dtype)
            return fn_with_read()
        self._testVariableReadInFunctionalOp(build_functional_op, 'StatefulPartitionedCall')

    def testVariableReadInNestedFunction(self):
        if False:
            for i in range(10):
                print('nop')

        def build_functional_op(v):
            if False:
                for i in range(10):
                    print('nop')

            @def_function.function
            def fn_with_read():
                if False:
                    print('Hello World!')

                @def_function.function
                def inner_fn():
                    if False:
                        i = 10
                        return i + 15
                    return gen_resource_variable_ops.read_variable_op(v.handle, v.dtype)
                return inner_fn()
            return fn_with_read()
        self._testVariableReadInFunctionalOp(build_functional_op, 'StatefulPartitionedCall')

    def testVariableReadInWhileInInnerFunc(self):
        if False:
            return 10

        def build_functional_op(v):
            if False:
                while True:
                    i = 10

            @def_function.function
            def fn_with_read():
                if False:
                    for i in range(10):
                        print('nop')

                @def_function.function
                def inner_fn():
                    if False:
                        return 10

                    def body(_):
                        if False:
                            print('Hello World!')
                        return gen_resource_variable_ops.read_variable_op(v.handle, v.dtype)
                    return while_loop.while_loop(lambda i: True, body, [0.0], maximum_iterations=1)
                return inner_fn()
            return fn_with_read()
        self._testVariableReadInFunctionalOp(build_functional_op, 'StatefulPartitionedCall')

    def testVariableReadInCondInInnerFunc(self):
        if False:
            for i in range(10):
                print('nop')

        def build_functional_op(v):
            if False:
                print('Hello World!')

            @def_function.function
            def fn_with_read():
                if False:
                    print('Hello World!')

                @def_function.function
                def inner_fn():
                    if False:
                        return 10

                    def then_branch():
                        if False:
                            while True:
                                i = 10
                        return gen_resource_variable_ops.read_variable_op(v.handle, v.dtype)

                    def else_branch():
                        if False:
                            for i in range(10):
                                print('nop')
                        return array_ops.zeros([], v.dtype)
                    return cond.cond(constant_op.constant(True), then_branch, else_branch)
                return inner_fn()
            return fn_with_read()
        self._testVariableReadInFunctionalOp(build_functional_op, 'StatefulPartitionedCall')

    def _testVariableWriteInFunctionalOp(self, build_functional_op, op_type):
        if False:
            print('Hello World!')
        v = resource_variable_ops.ResourceVariable(1.0)
        self.evaluate(variables.global_variables_initializer())

        @def_function.function
        def write_var_in_while():
            if False:
                print('Hello World!')
            gen_resource_variable_ops.read_variable_op(v.handle, v.dtype, name='read1')
            result = build_functional_op(v)
            gen_resource_variable_ops.read_variable_op(v.handle, v.dtype, name='read2')
            gen_resource_variable_ops.assign_variable_op(v.handle, v + 1)
            return result
        func_graph = write_var_in_while.get_concrete_function().graph
        assert len(func_graph.inputs) == 1

        def get_op(op_type, sub_name):
            if False:
                print('Hello World!')
            operations = [op for op in func_graph.get_operations() if op.type == op_type and sub_name in op.name]
            assert len(operations) == 1
            return operations[0]
        read1 = get_op('ReadVariableOp', 'read1')
        functional_op = get_op(op_type, '')
        read2 = get_op('ReadVariableOp', 'read2')
        assign_op = get_op('AssignVariableOp', '')
        self.assertIn(read1, functional_op.control_inputs)
        self.assertIn(functional_op, read2.control_inputs)
        self.assertIn(read2, assign_op.control_inputs)
        self.assertIn(functional_op, assign_op.control_inputs)

    def testVariableWriteInWhileLoop(self):
        if False:
            print('Hello World!')

        def build_functional_op(v):
            if False:
                print('Hello World!')

            def body(_):
                if False:
                    for i in range(10):
                        print('nop')
                gen_resource_variable_ops.assign_variable_op(v.handle, v + 1)
                return gen_resource_variable_ops.read_variable_op(v.handle, v.dtype)
            return while_loop.while_loop(lambda i: True, body, [0.0], maximum_iterations=1)
        self._testVariableWriteInFunctionalOp(build_functional_op, 'While')

    def testVariableWriteInCondTrueBranch(self):
        if False:
            i = 10
            return i + 15

        def build_functional_op(v):
            if False:
                for i in range(10):
                    print('nop')

            def then_branch():
                if False:
                    for i in range(10):
                        print('nop')
                gen_resource_variable_ops.assign_variable_op(v.handle, v + 1)
                return gen_resource_variable_ops.read_variable_op(v.handle, v.dtype)

            def else_branch():
                if False:
                    while True:
                        i = 10
                return array_ops.zeros([], v.dtype)
            return cond.cond(constant_op.constant(True), then_branch, else_branch)
        self._testVariableWriteInFunctionalOp(build_functional_op, 'If')

    def testVariableWriteInCondFalseBranch(self):
        if False:
            i = 10
            return i + 15

        def build_functional_op(v):
            if False:
                return 10

            def then_branch():
                if False:
                    while True:
                        i = 10
                return array_ops.zeros([], v.dtype)

            def else_branch():
                if False:
                    while True:
                        i = 10
                gen_resource_variable_ops.assign_variable_op(v.handle, v + 1)
                return gen_resource_variable_ops.read_variable_op(v.handle, v.dtype)
            return cond.cond(constant_op.constant(False), then_branch, else_branch)
        self._testVariableWriteInFunctionalOp(build_functional_op, 'If')

    def testVariableWriteInCaseBranch0(self):
        if False:
            print('Hello World!')

        def build_functional_op(v):
            if False:
                print('Hello World!')

            def branch0():
                if False:
                    i = 10
                    return i + 15
                gen_resource_variable_ops.assign_variable_op(v.handle, v + 1)
                return gen_resource_variable_ops.read_variable_op(v.handle, v.dtype)

            def branch1():
                if False:
                    return 10
                return array_ops.zeros([], v.dtype)
            return control_flow_switch_case.switch_case(constant_op.constant(0), [branch0, branch1])
        self._testVariableWriteInFunctionalOp(build_functional_op, 'Case')

    def testVariableWriteInCaseBranch1(self):
        if False:
            for i in range(10):
                print('nop')

        def build_functional_op(v):
            if False:
                print('Hello World!')

            def branch0():
                if False:
                    while True:
                        i = 10
                return array_ops.zeros([], v.dtype)

            def branch1():
                if False:
                    while True:
                        i = 10
                gen_resource_variable_ops.assign_variable_op(v.handle, v + 1)
                return gen_resource_variable_ops.read_variable_op(v.handle, v.dtype)
            return control_flow_switch_case.switch_case(constant_op.constant(0), [branch0, branch1])
        self._testVariableWriteInFunctionalOp(build_functional_op, 'Case')

    def testVariableWriteInFunction(self):
        if False:
            i = 10
            return i + 15

        def build_functional_op(v):
            if False:
                return 10

            @def_function.function
            def fn_with_write():
                if False:
                    for i in range(10):
                        print('nop')
                gen_resource_variable_ops.assign_variable_op(v.handle, v + 1)
                return gen_resource_variable_ops.read_variable_op(v.handle, v.dtype)
            return fn_with_write()
        self._testVariableWriteInFunctionalOp(build_functional_op, 'StatefulPartitionedCall')

    def testVariableWriteInNestedFunction(self):
        if False:
            return 10

        def build_functional_op(v):
            if False:
                while True:
                    i = 10

            @def_function.function
            def fn_with_write():
                if False:
                    return 10

                @def_function.function
                def inner_fn():
                    if False:
                        i = 10
                        return i + 15
                    gen_resource_variable_ops.assign_variable_op(v.handle, v + 1)
                    return gen_resource_variable_ops.read_variable_op(v.handle, v.dtype)
                return inner_fn()
            return fn_with_write()
        self._testVariableWriteInFunctionalOp(build_functional_op, 'StatefulPartitionedCall')

    def testVariableWriteInWhileInInnerFunc(self):
        if False:
            for i in range(10):
                print('nop')

        def build_functional_op(v):
            if False:
                print('Hello World!')

            @def_function.function
            def fn_with_write():
                if False:
                    print('Hello World!')

                @def_function.function
                def inner_fn():
                    if False:
                        while True:
                            i = 10

                    def body(_):
                        if False:
                            return 10
                        gen_resource_variable_ops.assign_variable_op(v.handle, v + 1)
                        return gen_resource_variable_ops.read_variable_op(v.handle, v.dtype)
                    return while_loop.while_loop(lambda i: True, body, [0.0], maximum_iterations=1)
                return inner_fn()
            return fn_with_write()
        self._testVariableWriteInFunctionalOp(build_functional_op, 'StatefulPartitionedCall')

    def testVariableWriteInCondInInnerFunc(self):
        if False:
            while True:
                i = 10

        def build_functional_op(v):
            if False:
                i = 10
                return i + 15

            @def_function.function
            def fn_with_write():
                if False:
                    return 10

                @def_function.function
                def inner_fn():
                    if False:
                        for i in range(10):
                            print('nop')

                    def then_branch():
                        if False:
                            i = 10
                            return i + 15
                        gen_resource_variable_ops.assign_variable_op(v.handle, v + 1)
                        return gen_resource_variable_ops.read_variable_op(v.handle, v.dtype)

                    def else_branch():
                        if False:
                            while True:
                                i = 10
                        return array_ops.zeros([], v.dtype)
                    return cond.cond(constant_op.constant(True), then_branch, else_branch)
                return inner_fn()
            return fn_with_write()
        self._testVariableWriteInFunctionalOp(build_functional_op, 'StatefulPartitionedCall')

    @test_util.run_v1_only('b/120545219')
    def testCondMustRun(self):
        if False:
            while True:
                i = 10
        with context.graph_mode(), self.cached_session():
            v = resource_variable_ops.ResourceVariable(1.0)
            self.evaluate(variables.global_variables_initializer())
            p = array_ops.placeholder(dtype=dtypes.bool)
            with acd.AutomaticControlDependencies() as c:

                def true_fn():
                    if False:
                        i = 10
                        return i + 15
                    v.assign(v + 1)
                    return 0.0

                def false_fn():
                    if False:
                        return 10
                    v.assign(v + 4)
                    return 1.0
                cond.cond(p, true_fn, false_fn)
                val = v.read_value()
                val = c.mark_as_return(val)
            self.assertAllEqual(val.eval(feed_dict={p: False}), 5.0)
            self.assertAllEqual(val.eval(feed_dict={p: True}), 6.0)

    @test_util.run_v1_only('b/120545219')
    def testCondMustRunSeparateRead(self):
        if False:
            while True:
                i = 10
        with context.graph_mode(), self.cached_session():
            v = resource_variable_ops.ResourceVariable(1.0)
            self.evaluate(variables.global_variables_initializer())
            p = array_ops.placeholder(dtype=dtypes.bool)
            with acd.AutomaticControlDependencies() as c:

                def true_fn():
                    if False:
                        i = 10
                        return i + 15
                    v.assign(v + 1)
                    return 0.0

                def false_fn():
                    if False:
                        for i in range(10):
                            print('nop')
                    v.assign(v + 4)
                    return 1.0
                cond.cond(p, true_fn, false_fn)
                one = constant_op.constant(1.0)
                one = c.mark_as_return(one)
            one.eval(feed_dict={p: False})
            self.assertAllEqual(v.read_value(), 5.0)
            one.eval(feed_dict={p: True})
            self.assertAllEqual(v.read_value(), 6.0)

    @test_util.run_v1_only('b/120545219')
    def testCondNested(self):
        if False:
            while True:
                i = 10
        with context.graph_mode(), self.cached_session():
            v = resource_variable_ops.ResourceVariable(1.0)
            self.evaluate(variables.global_variables_initializer())
            p = array_ops.placeholder(dtype=dtypes.bool)
            q = array_ops.placeholder(dtype=dtypes.bool)
            with acd.AutomaticControlDependencies() as c:

                def true_fn():
                    if False:
                        return 10
                    v.assign(v + 1, name='true')
                    return 1.0

                def false_fn():
                    if False:
                        return 10

                    def inner_true_fn():
                        if False:
                            print('Hello World!')
                        v.assign(v * 2, name='false_true')
                        return 2.0

                    def inner_false_fn():
                        if False:
                            for i in range(10):
                                print('nop')
                        v.assign(v * 3, name='false_false')
                        return 3.0
                    cond.cond(q, inner_true_fn, inner_false_fn)
                    return 1.0
                cond.cond(p, true_fn, false_fn)
                with ops.name_scope('final'):
                    val = v.read_value()
                val = c.mark_as_return(val)
            self.assertAllEqual(val.eval(feed_dict={p: False, q: False}), 3.0)
            self.assertAllEqual(val.eval(feed_dict={p: False, q: True}), 6.0)
            self.assertAllEqual(val.eval(feed_dict={p: True, q: True}), 7.0)
            self.assertAllEqual(val.eval(feed_dict={p: True, q: False}), 8.0)

    @test_util.run_v1_only('b/120545219')
    def testCondOneBranch(self):
        if False:
            for i in range(10):
                print('nop')
        with context.graph_mode(), self.cached_session():
            v = resource_variable_ops.ResourceVariable(1.0)
            self.evaluate(variables.global_variables_initializer())
            p = array_ops.placeholder(dtype=dtypes.bool)
            with acd.AutomaticControlDependencies() as c:

                def true_fn():
                    if False:
                        print('Hello World!')
                    return 0.0

                def false_fn():
                    if False:
                        print('Hello World!')
                    v.assign(v + 4)
                    return 1.0
                cond.cond(p, true_fn, false_fn)
                val = v.read_value()
                val = c.mark_as_return(val)
            self.assertAllEqual(val.eval(feed_dict={p: False}), 5.0)
            self.assertAllEqual(val.eval(feed_dict={p: True}), 5.0)

    @test_util.run_v1_only('b/120545219')
    def testCondOneBranchUpdateBefore(self):
        if False:
            print('Hello World!')
        with context.graph_mode(), self.cached_session():
            v = resource_variable_ops.ResourceVariable(1.0)
            self.evaluate(variables.global_variables_initializer())
            p = array_ops.placeholder(dtype=dtypes.bool)
            with acd.AutomaticControlDependencies() as c:
                v.assign(v * 2)

                def true_fn():
                    if False:
                        print('Hello World!')
                    return 0.0

                def false_fn():
                    if False:
                        i = 10
                        return i + 15
                    v.assign(v + 4)
                    return 1.0
                cond.cond(p, true_fn, false_fn)
                val = v.read_value()
                val = c.mark_as_return(val)
            self.assertAllEqual(val.eval(feed_dict={p: False}), 6.0)
            self.assertAllEqual(val.eval(feed_dict={p: True}), 12.0)

    @test_util.run_v1_only('b/120545219')
    def testCondOneBranchUpdateAfter(self):
        if False:
            print('Hello World!')
        with context.graph_mode(), self.cached_session():
            v = resource_variable_ops.ResourceVariable(1.0)
            self.evaluate(variables.global_variables_initializer())
            p = array_ops.placeholder(dtype=dtypes.bool)
            with acd.AutomaticControlDependencies() as c:

                def true_fn():
                    if False:
                        for i in range(10):
                            print('nop')
                    return 0.0

                def false_fn():
                    if False:
                        while True:
                            i = 10
                    v.assign(v + 4)
                    return 1.0
                cond.cond(p, true_fn, false_fn)
                v.assign(v * 2)
                val = v.read_value()
                val = c.mark_as_return(val)
            self.assertAllEqual(val.eval(feed_dict={p: False}), 10.0)
            self.assertAllEqual(val.eval(feed_dict={p: True}), 20.0)

    def testFunctionWhileLoopWithCapturedLoopVars(self):
        if False:
            while True:
                i = 10
        n = 3
        x = constant_op.constant(list(range(n)))

        @def_function.function
        def loop():
            if False:
                print('Hello World!')
            c = lambda i, x: i < n
            b = lambda i, x: (i + 1, x + 1)
            (i, out) = while_loop.while_loop(c, b, (0, x))
            return (i, out)
        (i, out) = loop()
        self.assertEqual(int(i), 3)
        self.assertAllEqual(out, [3, 4, 5])

    def testDecorator(self):
        if False:
            print('Hello World!')
        with context.graph_mode(), self.cached_session():
            v = resource_variable_ops.ResourceVariable(1.0)
            self.evaluate(variables.global_variables_initializer())

            @acd.automatic_control_dependencies
            def f():
                if False:
                    print('Hello World!')
                v.assign(v + 1)
                v.assign(2 * v)
                return v.read_value()
            self.assertAllEqual(f(), 4.0)

    def testOptimizerInFunction(self):
        if False:
            for i in range(10):
                print('nop')

        def loss(v):
            if False:
                for i in range(10):
                    print('nop')
            return v ** 2
        optimizer = momentum.MomentumOptimizer(learning_rate=1.0, momentum=1.0)

        @def_function.function
        def train():
            if False:
                return 10
            grad = backprop.implicit_grad(loss)(self.v)
            optimizer.apply_gradients(grad)
            return self.v.read_value()
        self.v = resource_variable_ops.ResourceVariable(1.0)
        value = train()
        self.assertEqual(value.numpy(), -1.0)

    def testReturningNonTensorRaisesError(self):
        if False:
            print('Hello World!')
        optimizer = momentum.MomentumOptimizer(learning_rate=1.0, momentum=1.0)
        optimizer.apply_gradients = def_function.function(optimizer.apply_gradients)
        v = resource_variable_ops.ResourceVariable(1.0)
        grad = backprop.implicit_grad(lambda v: v ** 2)(v)
        with self.assertRaisesRegex(TypeError, '.*must return zero or more Tensors.*'):
            optimizer.apply_gradients(grad)

    def testOptimizerNonSlotVarsInFunctionNoError(self):
        if False:
            return 10

        def loss(v):
            if False:
                print('Hello World!')
            return v ** 2
        optimizer = adam.AdamOptimizer(learning_rate=1.0)

        @def_function.function
        def train():
            if False:
                print('Hello World!')
            grad = backprop.implicit_grad(loss)(self.v)
            optimizer.apply_gradients(grad)
            return self.v.read_value()
        self.v = resource_variable_ops.ResourceVariable(1.0)
        train()

    def testOptimizerInFunctionWithCapturedVariable(self):
        if False:
            print('Hello World!')
        v = resource_variable_ops.ResourceVariable(1.0)

        def loss():
            if False:
                while True:
                    i = 10
            return v ** 2
        optimizer = momentum.MomentumOptimizer(learning_rate=1.0, momentum=1.0)

        @def_function.function
        def train():
            if False:
                i = 10
                return i + 15
            grad = backprop.implicit_grad(loss)()
            optimizer.apply_gradients(grad)
        train()
        self.assertEqual(v.numpy(), -1.0)

    def testRepeatedResourceInput(self):
        if False:
            print('Hello World!')
        var = resource_variable_ops.ResourceVariable(1.0)

        @def_function.function
        def inner(var1, var2):
            if False:
                while True:
                    i = 10
            return resource_variable_ops.read_variable_op(var1, dtypes.float32) + resource_variable_ops.read_variable_op(var2, dtypes.float32)

        @def_function.function
        def outer():
            if False:
                while True:
                    i = 10
            return inner(var.handle, var.handle)
        self.assertEqual(self.evaluate(outer()), 2.0)

    def testManualControlDepMonitoringAttrNotAdded(self):
        if False:
            for i in range(10):
                print('nop')
        with context.graph_mode(), self.cached_session():
            v = resource_variable_ops.ResourceVariable(1.0)
            self.evaluate(variables.global_variables_initializer())
            with acd.AutomaticControlDependencies():
                read_op1 = gen_resource_variable_ops.read_variable_op(v.handle, v.dtype).op
                read_op2 = gen_resource_variable_ops.read_variable_op(v.handle, v.dtype).op
                assign_op = gen_resource_variable_ops.assign_variable_op(v.handle, v + 1)
            self.assertIn(read_op1, assign_op.control_inputs)
            self.assertIn(read_op2, assign_op.control_inputs)
            with self.assertRaises(ValueError):
                assign_op.get_attr('_has_manual_control_dependencies')
            with self.assertRaises(ValueError):
                read_op1.get_attr('_has_manual_control_dependencies')
            with self.assertRaises(ValueError):
                read_op2.get_attr('_has_manual_control_dependencies')
if __name__ == '__main__':
    ops.enable_eager_execution()
    test.main()