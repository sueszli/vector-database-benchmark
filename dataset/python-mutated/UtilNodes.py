from __future__ import absolute_import
from . import Nodes
from . import ExprNodes
from .Nodes import Node
from .ExprNodes import AtomicExprNode
from .PyrexTypes import c_ptr_type, c_bint_type

class TempHandle(object):
    temp = None
    needs_xdecref = False

    def __init__(self, type, needs_cleanup=None):
        if False:
            print('Hello World!')
        self.type = type
        if needs_cleanup is None:
            self.needs_cleanup = type.is_pyobject
        else:
            self.needs_cleanup = needs_cleanup

    def ref(self, pos):
        if False:
            while True:
                i = 10
        return TempRefNode(pos, handle=self, type=self.type)

class TempRefNode(AtomicExprNode):

    def analyse_types(self, env):
        if False:
            i = 10
            return i + 15
        assert self.type == self.handle.type
        return self

    def analyse_target_types(self, env):
        if False:
            for i in range(10):
                print('nop')
        assert self.type == self.handle.type
        return self

    def analyse_target_declaration(self, env):
        if False:
            print('Hello World!')
        pass

    def calculate_result_code(self):
        if False:
            i = 10
            return i + 15
        result = self.handle.temp
        if result is None:
            result = '<error>'
        return result

    def generate_result_code(self, code):
        if False:
            while True:
                i = 10
        pass

    def generate_assignment_code(self, rhs, code, overloaded_assignment=False):
        if False:
            print('Hello World!')
        if self.type.is_pyobject:
            rhs.make_owned_reference(code)
            code.put_xdecref(self.result(), self.ctype())
        code.putln('%s = %s;' % (self.result(), rhs.result() if overloaded_assignment else rhs.result_as(self.ctype())))
        rhs.generate_post_assignment_code(code)
        rhs.free_temps(code)

class TempsBlockNode(Node):
    """
    Creates a block which allocates temporary variables.
    This is used by transforms to output constructs that need
    to make use of a temporary variable. Simply pass the types
    of the needed temporaries to the constructor.

    The variables can be referred to using a TempRefNode
    (which can be constructed by calling get_ref_node).
    """
    child_attrs = ['body']

    def generate_execution_code(self, code):
        if False:
            print('Hello World!')
        for handle in self.temps:
            handle.temp = code.funcstate.allocate_temp(handle.type, manage_ref=handle.needs_cleanup)
        self.body.generate_execution_code(code)
        for handle in self.temps:
            if handle.needs_cleanup:
                if handle.needs_xdecref:
                    code.put_xdecref_clear(handle.temp, handle.type)
                else:
                    code.put_decref_clear(handle.temp, handle.type)
            code.funcstate.release_temp(handle.temp)

    def analyse_declarations(self, env):
        if False:
            print('Hello World!')
        self.body.analyse_declarations(env)

    def analyse_expressions(self, env):
        if False:
            for i in range(10):
                print('nop')
        self.body = self.body.analyse_expressions(env)
        return self

    def generate_function_definitions(self, env, code):
        if False:
            for i in range(10):
                print('nop')
        self.body.generate_function_definitions(env, code)

    def annotate(self, code):
        if False:
            for i in range(10):
                print('nop')
        self.body.annotate(code)

class ResultRefNode(AtomicExprNode):
    subexprs = []
    lhs_of_first_assignment = False

    def __init__(self, expression=None, pos=None, type=None, may_hold_none=True, is_temp=False):
        if False:
            return 10
        self.expression = expression
        self.pos = None
        self.may_hold_none = may_hold_none
        if expression is not None:
            self.pos = expression.pos
            self.type = getattr(expression, 'type', None)
        if pos is not None:
            self.pos = pos
        if type is not None:
            self.type = type
        if is_temp:
            self.is_temp = True
        assert self.pos is not None

    def clone_node(self):
        if False:
            for i in range(10):
                print('nop')
        return self

    def type_dependencies(self, env):
        if False:
            for i in range(10):
                print('nop')
        if self.expression:
            return self.expression.type_dependencies(env)
        else:
            return ()

    def update_expression(self, expression):
        if False:
            while True:
                i = 10
        self.expression = expression
        type = getattr(expression, 'type', None)
        if type:
            self.type = type

    def analyse_target_declaration(self, env):
        if False:
            for i in range(10):
                print('nop')
        pass

    def analyse_types(self, env):
        if False:
            return 10
        if self.expression is not None:
            if not self.expression.type:
                self.expression = self.expression.analyse_types(env)
            self.type = self.expression.type
        return self

    def infer_type(self, env):
        if False:
            return 10
        if self.type is not None:
            return self.type
        if self.expression is not None:
            if self.expression.type is not None:
                return self.expression.type
            return self.expression.infer_type(env)
        assert False, 'cannot infer type of ResultRefNode'

    def may_be_none(self):
        if False:
            while True:
                i = 10
        if not self.type.is_pyobject:
            return False
        return self.may_hold_none

    def _DISABLED_may_be_none(self):
        if False:
            return 10
        if self.expression is not None:
            return self.expression.may_be_none()
        if self.type is not None:
            return self.type.is_pyobject
        return True

    def is_simple(self):
        if False:
            for i in range(10):
                print('nop')
        return True

    def result(self):
        if False:
            i = 10
            return i + 15
        try:
            return self.result_code
        except AttributeError:
            if self.expression is not None:
                self.result_code = self.expression.result()
        return self.result_code

    def generate_evaluation_code(self, code):
        if False:
            return 10
        pass

    def generate_result_code(self, code):
        if False:
            i = 10
            return i + 15
        pass

    def generate_disposal_code(self, code):
        if False:
            print('Hello World!')
        pass

    def generate_assignment_code(self, rhs, code, overloaded_assignment=False):
        if False:
            print('Hello World!')
        if self.type.is_pyobject:
            rhs.make_owned_reference(code)
            if not self.lhs_of_first_assignment:
                code.put_decref(self.result(), self.ctype())
        code.putln('%s = %s;' % (self.result(), rhs.result() if overloaded_assignment else rhs.result_as(self.ctype())))
        rhs.generate_post_assignment_code(code)
        rhs.free_temps(code)

    def allocate_temps(self, env):
        if False:
            return 10
        pass

    def release_temp(self, env):
        if False:
            i = 10
            return i + 15
        pass

    def free_temps(self, code):
        if False:
            print('Hello World!')
        pass

class LetNodeMixin:

    def set_temp_expr(self, lazy_temp):
        if False:
            print('Hello World!')
        self.lazy_temp = lazy_temp
        self.temp_expression = lazy_temp.expression

    def setup_temp_expr(self, code):
        if False:
            print('Hello World!')
        self.temp_expression.generate_evaluation_code(code)
        self.temp_type = self.temp_expression.type
        if self.temp_type.is_array:
            self.temp_type = c_ptr_type(self.temp_type.base_type)
        self._result_in_temp = self.temp_expression.result_in_temp()
        if self._result_in_temp:
            self.temp = self.temp_expression.result()
        else:
            if self.temp_type.is_memoryviewslice:
                self.temp_expression.make_owned_memoryviewslice(code)
            else:
                self.temp_expression.make_owned_reference(code)
            self.temp = code.funcstate.allocate_temp(self.temp_type, manage_ref=True)
            code.putln('%s = %s;' % (self.temp, self.temp_expression.result()))
            self.temp_expression.generate_disposal_code(code)
            self.temp_expression.free_temps(code)
        self.lazy_temp.result_code = self.temp

    def teardown_temp_expr(self, code):
        if False:
            print('Hello World!')
        if self._result_in_temp:
            self.temp_expression.generate_disposal_code(code)
            self.temp_expression.free_temps(code)
        else:
            if self.temp_type.needs_refcounting:
                code.put_decref_clear(self.temp, self.temp_type)
            code.funcstate.release_temp(self.temp)

class EvalWithTempExprNode(ExprNodes.ExprNode, LetNodeMixin):
    subexprs = ['temp_expression', 'subexpression']

    def __init__(self, lazy_temp, subexpression):
        if False:
            i = 10
            return i + 15
        self.set_temp_expr(lazy_temp)
        self.pos = subexpression.pos
        self.subexpression = subexpression
        self.type = self.subexpression.type

    def infer_type(self, env):
        if False:
            print('Hello World!')
        return self.subexpression.infer_type(env)

    def may_be_none(self):
        if False:
            i = 10
            return i + 15
        return self.subexpression.may_be_none()

    def result(self):
        if False:
            return 10
        return self.subexpression.result()

    def analyse_types(self, env):
        if False:
            print('Hello World!')
        self.temp_expression = self.temp_expression.analyse_types(env)
        self.lazy_temp.update_expression(self.temp_expression)
        self.subexpression = self.subexpression.analyse_types(env)
        self.type = self.subexpression.type
        return self

    def free_subexpr_temps(self, code):
        if False:
            return 10
        self.subexpression.free_temps(code)

    def generate_subexpr_disposal_code(self, code):
        if False:
            while True:
                i = 10
        self.subexpression.generate_disposal_code(code)

    def generate_evaluation_code(self, code):
        if False:
            for i in range(10):
                print('nop')
        self.setup_temp_expr(code)
        self.subexpression.generate_evaluation_code(code)
        self.teardown_temp_expr(code)
LetRefNode = ResultRefNode

class LetNode(Nodes.StatNode, LetNodeMixin):
    child_attrs = ['temp_expression', 'body']

    def __init__(self, lazy_temp, body):
        if False:
            print('Hello World!')
        self.set_temp_expr(lazy_temp)
        self.pos = body.pos
        self.body = body

    def analyse_declarations(self, env):
        if False:
            return 10
        self.temp_expression.analyse_declarations(env)
        self.body.analyse_declarations(env)

    def analyse_expressions(self, env):
        if False:
            i = 10
            return i + 15
        self.temp_expression = self.temp_expression.analyse_expressions(env)
        self.body = self.body.analyse_expressions(env)
        return self

    def generate_execution_code(self, code):
        if False:
            print('Hello World!')
        self.setup_temp_expr(code)
        self.body.generate_execution_code(code)
        self.teardown_temp_expr(code)

    def generate_function_definitions(self, env, code):
        if False:
            while True:
                i = 10
        self.temp_expression.generate_function_definitions(env, code)
        self.body.generate_function_definitions(env, code)

class TempResultFromStatNode(ExprNodes.ExprNode):
    subexprs = []
    child_attrs = ['body']

    def __init__(self, result_ref, body):
        if False:
            return 10
        self.result_ref = result_ref
        self.pos = body.pos
        self.body = body
        self.type = result_ref.type
        self.is_temp = 1

    def analyse_declarations(self, env):
        if False:
            while True:
                i = 10
        self.body.analyse_declarations(env)

    def analyse_types(self, env):
        if False:
            return 10
        self.body = self.body.analyse_expressions(env)
        return self

    def may_be_none(self):
        if False:
            for i in range(10):
                print('nop')
        return self.result_ref.may_be_none()

    def generate_result_code(self, code):
        if False:
            print('Hello World!')
        self.result_ref.result_code = self.result()
        self.body.generate_execution_code(code)

    def generate_function_definitions(self, env, code):
        if False:
            print('Hello World!')
        self.body.generate_function_definitions(env, code)

class HasGilNode(AtomicExprNode):
    """
    Simple node that evaluates to 0 or 1 depending on whether we're
    in a nogil context
    """
    type = c_bint_type

    def analyse_types(self, env):
        if False:
            while True:
                i = 10
        return self

    def generate_result_code(self, code):
        if False:
            while True:
                i = 10
        self.has_gil = code.funcstate.gil_owned

    def calculate_result_code(self):
        if False:
            print('Hello World!')
        return '1' if self.has_gil else '0'