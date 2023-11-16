from sympy.external import import_module
import os
cin = import_module('clang.cindex', import_kwargs={'fromlist': ['cindex']})
"\nThis module contains all the necessary Classes and Function used to Parse C and\nC++ code into SymPy expression\nThe module serves as a backend for SymPyExpression to parse C code\nIt is also dependent on Clang's AST and SymPy's Codegen AST.\nThe module only supports the features currently supported by the Clang and\ncodegen AST which will be updated as the development of codegen AST and this\nmodule progresses.\nYou might find unexpected bugs and exceptions while using the module, feel free\nto report them to the SymPy Issue Tracker\n\nFeatures Supported\n==================\n\n- Variable Declarations (integers and reals)\n- Assignment (using integer & floating literal and function calls)\n- Function Definitions and Declaration\n- Function Calls\n- Compound statements, Return statements\n\nNotes\n=====\n\nThe module is dependent on an external dependency which needs to be installed\nto use the features of this module.\n\nClang: The C and C++ compiler which is used to extract an AST from the provided\nC source code.\n\nReferences\n==========\n\n.. [1] https://github.com/sympy/sympy/issues\n.. [2] https://clang.llvm.org/docs/\n.. [3] https://clang.llvm.org/docs/IntroductionToTheClangAST.html\n\n"
if cin:
    from sympy.codegen.ast import Variable, Integer, Float, FunctionPrototype, FunctionDefinition, FunctionCall, none, Return, Assignment, intc, int8, int16, int64, uint8, uint16, uint32, uint64, float32, float64, float80, aug_assign, bool_, While, CodeBlock
    from sympy.codegen.cnodes import PreDecrement, PostDecrement, PreIncrement, PostIncrement
    from sympy.core import Add, Mod, Mul, Pow, Rel
    from sympy.logic.boolalg import And, as_Boolean, Not, Or
    from sympy.core.symbol import Symbol
    from sympy.core.sympify import sympify
    from sympy.logic.boolalg import false, true
    import sys
    import tempfile

    class BaseParser:
        """Base Class for the C parser"""

        def __init__(self):
            if False:
                i = 10
                return i + 15
            'Initializes the Base parser creating a Clang AST index'
            self.index = cin.Index.create()

        def diagnostics(self, out):
            if False:
                while True:
                    i = 10
            'Diagostics function for the Clang AST'
            for diag in self.tu.diagnostics:
                print('%s %s (line %s, col %s) %s' % ({4: 'FATAL', 3: 'ERROR', 2: 'WARNING', 1: 'NOTE', 0: 'IGNORED'}[diag.severity], diag.location.file, diag.location.line, diag.location.column, diag.spelling), file=out)

    class CCodeConverter(BaseParser):
        """The Code Convereter for Clang AST

        The converter object takes the C source code or file as input and
        converts them to SymPy Expressions.
        """

        def __init__(self):
            if False:
                while True:
                    i = 10
            'Initializes the code converter'
            super().__init__()
            self._py_nodes = []
            self._data_types = {'void': {cin.TypeKind.VOID: none}, 'bool': {cin.TypeKind.BOOL: bool_}, 'int': {cin.TypeKind.SCHAR: int8, cin.TypeKind.SHORT: int16, cin.TypeKind.INT: intc, cin.TypeKind.LONG: int64, cin.TypeKind.UCHAR: uint8, cin.TypeKind.USHORT: uint16, cin.TypeKind.UINT: uint32, cin.TypeKind.ULONG: uint64}, 'float': {cin.TypeKind.FLOAT: float32, cin.TypeKind.DOUBLE: float64, cin.TypeKind.LONGDOUBLE: float80}}

        def parse(self, filename, flags):
            if False:
                return 10
            'Function to parse a file with C source code\n\n            It takes the filename as an attribute and creates a Clang AST\n            Translation Unit parsing the file.\n            Then the transformation function is called on the translation unit,\n            whose reults are collected into a list which is returned by the\n            function.\n\n            Parameters\n            ==========\n\n            filename : string\n                Path to the C file to be parsed\n\n            flags: list\n                Arguments to be passed to Clang while parsing the C code\n\n            Returns\n            =======\n\n            py_nodes: list\n                A list of SymPy AST nodes\n\n            '
            filepath = os.path.abspath(filename)
            self.tu = self.index.parse(filepath, args=flags, options=cin.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD)
            for child in self.tu.cursor.get_children():
                if child.kind == cin.CursorKind.VAR_DECL or child.kind == cin.CursorKind.FUNCTION_DECL:
                    self._py_nodes.append(self.transform(child))
            return self._py_nodes

        def parse_str(self, source, flags):
            if False:
                return 10
            'Function to parse a string with C source code\n\n            It takes the source code as an attribute, stores it in a temporary\n            file and creates a Clang AST Translation Unit parsing the file.\n            Then the transformation function is called on the translation unit,\n            whose reults are collected into a list which is returned by the\n            function.\n\n            Parameters\n            ==========\n\n            source : string\n                A string containing the C source code to be parsed\n\n            flags: list\n                Arguments to be passed to Clang while parsing the C code\n\n            Returns\n            =======\n\n            py_nodes: list\n                A list of SymPy AST nodes\n\n            '
            file = tempfile.NamedTemporaryFile(mode='w+', suffix='.cpp')
            file.write(source)
            file.seek(0)
            self.tu = self.index.parse(file.name, args=flags, options=cin.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD)
            file.close()
            for child in self.tu.cursor.get_children():
                if child.kind == cin.CursorKind.VAR_DECL or child.kind == cin.CursorKind.FUNCTION_DECL:
                    self._py_nodes.append(self.transform(child))
            return self._py_nodes

        def transform(self, node):
            if False:
                return 10
            'Transformation Function for Clang AST nodes\n\n            It determines the kind of node and calls the respective\n            transformation function for that node.\n\n            Raises\n            ======\n\n            NotImplementedError : if the transformation for the provided node\n            is not implemented\n\n            '
            handler = getattr(self, 'transform_%s' % node.kind.name.lower(), None)
            if handler is None:
                print('Ignoring node of type %s (%s)' % (node.kind, ' '.join((t.spelling for t in node.get_tokens()))), file=sys.stderr)
            return handler(node)

        def transform_var_decl(self, node):
            if False:
                i = 10
                return i + 15
            'Transformation Function for Variable Declaration\n\n            Used to create nodes for variable declarations and assignments with\n            values or function call for the respective nodes in the clang AST\n\n            Returns\n            =======\n\n            A variable node as Declaration, with the initial value if given\n\n            Raises\n            ======\n\n            NotImplementedError : if called for data types not currently\n            implemented\n\n            Notes\n            =====\n\n            The function currently supports following data types:\n\n            Boolean:\n                bool, _Bool\n\n            Integer:\n                8-bit: signed char and unsigned char\n                16-bit: short, short int, signed short,\n                    signed short int, unsigned short, unsigned short int\n                32-bit: int, signed int, unsigned int\n                64-bit: long, long int, signed long,\n                    signed long int, unsigned long, unsigned long int\n\n            Floating point:\n                Single Precision: float\n                Double Precision: double\n                Extended Precision: long double\n\n            '
            if node.type.kind in self._data_types['int']:
                type = self._data_types['int'][node.type.kind]
            elif node.type.kind in self._data_types['float']:
                type = self._data_types['float'][node.type.kind]
            elif node.type.kind in self._data_types['bool']:
                type = self._data_types['bool'][node.type.kind]
            else:
                raise NotImplementedError('Only bool, int and float are supported')
            try:
                children = node.get_children()
                child = next(children)
                while child.kind == cin.CursorKind.NAMESPACE_REF or child.kind == cin.CursorKind.TYPE_REF:
                    child = next(children)
                val = self.transform(child)
                supported_rhs = [cin.CursorKind.INTEGER_LITERAL, cin.CursorKind.FLOATING_LITERAL, cin.CursorKind.UNEXPOSED_EXPR, cin.CursorKind.BINARY_OPERATOR, cin.CursorKind.PAREN_EXPR, cin.CursorKind.UNARY_OPERATOR, cin.CursorKind.CXX_BOOL_LITERAL_EXPR]
                if child.kind in supported_rhs:
                    if isinstance(val, str):
                        value = Symbol(val)
                    elif isinstance(val, bool):
                        if node.type.kind in self._data_types['int']:
                            value = Integer(0) if val == False else Integer(1)
                        elif node.type.kind in self._data_types['float']:
                            value = Float(0.0) if val == False else Float(1.0)
                        elif node.type.kind in self._data_types['bool']:
                            value = sympify(val)
                    elif isinstance(val, (Integer, int, Float, float)):
                        if node.type.kind in self._data_types['int']:
                            value = Integer(val)
                        elif node.type.kind in self._data_types['float']:
                            value = Float(val)
                        elif node.type.kind in self._data_types['bool']:
                            value = sympify(bool(val))
                    else:
                        value = val
                    return Variable(node.spelling).as_Declaration(type=type, value=value)
                elif child.kind == cin.CursorKind.CALL_EXPR:
                    return Variable(node.spelling).as_Declaration(value=val)
                else:
                    raise NotImplementedError('Given variable declaration "{}" is not possible to parse yet!'.format(' '.join((t.spelling for t in node.get_tokens()))))
            except StopIteration:
                return Variable(node.spelling).as_Declaration(type=type)

        def transform_function_decl(self, node):
            if False:
                i = 10
                return i + 15
            'Transformation Function For Function Declaration\n\n            Used to create nodes for function declarations and definitions for\n            the respective nodes in the clang AST\n\n            Returns\n            =======\n\n            function : Codegen AST node\n                - FunctionPrototype node if function body is not present\n                - FunctionDefinition node if the function body is present\n\n\n            '
            if node.result_type.kind in self._data_types['int']:
                ret_type = self._data_types['int'][node.result_type.kind]
            elif node.result_type.kind in self._data_types['float']:
                ret_type = self._data_types['float'][node.result_type.kind]
            elif node.result_type.kind in self._data_types['bool']:
                ret_type = self._data_types['bool'][node.result_type.kind]
            elif node.result_type.kind in self._data_types['void']:
                ret_type = self._data_types['void'][node.result_type.kind]
            else:
                raise NotImplementedError('Only void, bool, int and float are supported')
            body = []
            param = []
            for child in node.get_children():
                decl = self.transform(child)
                if child.kind == cin.CursorKind.PARM_DECL:
                    param.append(decl)
                elif child.kind == cin.CursorKind.COMPOUND_STMT:
                    for val in decl:
                        body.append(val)
                else:
                    body.append(decl)
            if body == []:
                function = FunctionPrototype(return_type=ret_type, name=node.spelling, parameters=param)
            else:
                function = FunctionDefinition(return_type=ret_type, name=node.spelling, parameters=param, body=body)
            return function

        def transform_parm_decl(self, node):
            if False:
                i = 10
                return i + 15
            'Transformation function for Parameter Declaration\n\n            Used to create parameter nodes for the required functions for the\n            respective nodes in the clang AST\n\n            Returns\n            =======\n\n            param : Codegen AST Node\n                Variable node with the value and type of the variable\n\n            Raises\n            ======\n\n            ValueError if multiple children encountered in the parameter node\n\n            '
            if node.type.kind in self._data_types['int']:
                type = self._data_types['int'][node.type.kind]
            elif node.type.kind in self._data_types['float']:
                type = self._data_types['float'][node.type.kind]
            elif node.type.kind in self._data_types['bool']:
                type = self._data_types['bool'][node.type.kind]
            else:
                raise NotImplementedError('Only bool, int and float are supported')
            try:
                children = node.get_children()
                child = next(children)
                while child.kind in [cin.CursorKind.NAMESPACE_REF, cin.CursorKind.TYPE_REF, cin.CursorKind.TEMPLATE_REF]:
                    child = next(children)
                lit = self.transform(child)
                if node.type.kind in self._data_types['int']:
                    val = Integer(lit)
                elif node.type.kind in self._data_types['float']:
                    val = Float(lit)
                elif node.type.kind in self._data_types['bool']:
                    val = sympify(bool(lit))
                else:
                    raise NotImplementedError('Only bool, int and float are supported')
                param = Variable(node.spelling).as_Declaration(type=type, value=val)
            except StopIteration:
                param = Variable(node.spelling).as_Declaration(type=type)
            try:
                self.transform(next(children))
                raise ValueError("Can't handle multiple children on parameter")
            except StopIteration:
                pass
            return param

        def transform_integer_literal(self, node):
            if False:
                i = 10
                return i + 15
            'Transformation function for integer literal\n\n            Used to get the value and type of the given integer literal.\n\n            Returns\n            =======\n\n            val : list\n                List with two arguments type and Value\n                type contains the type of the integer\n                value contains the value stored in the variable\n\n            Notes\n            =====\n\n            Only Base Integer type supported for now\n\n            '
            try:
                value = next(node.get_tokens()).spelling
            except StopIteration:
                value = node.literal
            return int(value)

        def transform_floating_literal(self, node):
            if False:
                for i in range(10):
                    print('nop')
            'Transformation function for floating literal\n\n            Used to get the value and type of the given floating literal.\n\n            Returns\n            =======\n\n            val : list\n                List with two arguments type and Value\n                type contains the type of float\n                value contains the value stored in the variable\n\n            Notes\n            =====\n\n            Only Base Float type supported for now\n\n            '
            try:
                value = next(node.get_tokens()).spelling
            except (StopIteration, ValueError):
                value = node.literal
            return float(value)

        def transform_string_literal(self, node):
            if False:
                return 10
            pass

        def transform_character_literal(self, node):
            if False:
                print('Hello World!')
            'Transformation function for character literal\n\n            Used to get the value of the given character literal.\n\n            Returns\n            =======\n\n            val : int\n                val contains the ascii value of the character literal\n\n            Notes\n            =====\n\n            Only for cases where character is assigned to a integer value,\n            since character literal is not in SymPy AST\n\n            '
            try:
                value = next(node.get_tokens()).spelling
            except (StopIteration, ValueError):
                value = node.literal
            return ord(str(value[1]))

        def transform_cxx_bool_literal_expr(self, node):
            if False:
                print('Hello World!')
            'Transformation function for boolean literal\n\n            Used to get the value of the given boolean literal.\n\n            Returns\n            =======\n\n            value : bool\n                value contains the boolean value of the variable\n\n            '
            try:
                value = next(node.get_tokens()).spelling
            except (StopIteration, ValueError):
                value = node.literal
            return True if value == 'true' else False

        def transform_unexposed_decl(self, node):
            if False:
                return 10
            'Transformation function for unexposed declarations'
            pass

        def transform_unexposed_expr(self, node):
            if False:
                return 10
            'Transformation function for unexposed expression\n\n            Unexposed expressions are used to wrap float, double literals and\n            expressions\n\n            Returns\n            =======\n\n            expr : Codegen AST Node\n                the result from the wrapped expression\n\n            None : NoneType\n                No childs are found for the node\n\n            Raises\n            ======\n\n            ValueError if the expression contains multiple children\n\n            '
            try:
                children = node.get_children()
                expr = self.transform(next(children))
            except StopIteration:
                return None
            try:
                next(children)
                raise ValueError('Unexposed expression has > 1 children.')
            except StopIteration:
                pass
            return expr

        def transform_decl_ref_expr(self, node):
            if False:
                i = 10
                return i + 15
            'Returns the name of the declaration reference'
            return node.spelling

        def transform_call_expr(self, node):
            if False:
                return 10
            'Transformation function for a call expression\n\n            Used to create function call nodes for the function calls present\n            in the C code\n\n            Returns\n            =======\n\n            FunctionCall : Codegen AST Node\n                FunctionCall node with parameters if any parameters are present\n\n            '
            param = []
            children = node.get_children()
            child = next(children)
            while child.kind == cin.CursorKind.NAMESPACE_REF:
                child = next(children)
            while child.kind == cin.CursorKind.TYPE_REF:
                child = next(children)
            first_child = self.transform(child)
            try:
                for child in children:
                    arg = self.transform(child)
                    if child.kind == cin.CursorKind.INTEGER_LITERAL:
                        param.append(Integer(arg))
                    elif child.kind == cin.CursorKind.FLOATING_LITERAL:
                        param.append(Float(arg))
                    else:
                        param.append(arg)
                return FunctionCall(first_child, param)
            except StopIteration:
                return FunctionCall(first_child)

        def transform_return_stmt(self, node):
            if False:
                return 10
            'Returns the Return Node for a return statement'
            return Return(next(node.get_children()).spelling)

        def transform_compound_stmt(self, node):
            if False:
                i = 10
                return i + 15
            'Transformation function for compond statemets\n\n            Returns\n            =======\n\n            expr : list\n                list of Nodes for the expressions present in the statement\n\n            None : NoneType\n                if the compound statement is empty\n\n            '
            expr = []
            children = node.get_children()
            for child in children:
                expr.append(self.transform(child))
            return expr

        def transform_decl_stmt(self, node):
            if False:
                for i in range(10):
                    print('nop')
            'Transformation function for declaration statements\n\n            These statements are used to wrap different kinds of declararions\n            like variable or function declaration\n            The function calls the transformer function for the child of the\n            given node\n\n            Returns\n            =======\n\n            statement : Codegen AST Node\n                contains the node returned by the children node for the type of\n                declaration\n\n            Raises\n            ======\n\n            ValueError if multiple children present\n\n            '
            try:
                children = node.get_children()
                statement = self.transform(next(children))
            except StopIteration:
                pass
            try:
                self.transform(next(children))
                raise ValueError("Don't know how to handle multiple statements")
            except StopIteration:
                pass
            return statement

        def transform_paren_expr(self, node):
            if False:
                return 10
            'Transformation function for Parenthesized expressions\n\n            Returns the result from its children nodes\n\n            '
            return self.transform(next(node.get_children()))

        def transform_compound_assignment_operator(self, node):
            if False:
                while True:
                    i = 10
            'Transformation function for handling shorthand operators\n\n            Returns\n            =======\n\n            augmented_assignment_expression: Codegen AST node\n                    shorthand assignment expression represented as Codegen AST\n\n            Raises\n            ======\n\n            NotImplementedError\n                If the shorthand operator for bitwise operators\n                (~=, ^=, &=, |=, <<=, >>=) is encountered\n\n            '
            return self.transform_binary_operator(node)

        def transform_unary_operator(self, node):
            if False:
                i = 10
                return i + 15
            'Transformation function for handling unary operators\n\n            Returns\n            =======\n\n            unary_expression: Codegen AST node\n                    simplified unary expression represented as Codegen AST\n\n            Raises\n            ======\n\n            NotImplementedError\n                If dereferencing operator(*), address operator(&) or\n                bitwise NOT operator(~) is encountered\n\n            '
            operators_list = ['+', '-', '++', '--', '!']
            tokens = list(node.get_tokens())
            if tokens[0].spelling in operators_list:
                child = self.transform(next(node.get_children()))
                if isinstance(child, str):
                    if tokens[0].spelling == '+':
                        return Symbol(child)
                    if tokens[0].spelling == '-':
                        return Mul(Symbol(child), -1)
                    if tokens[0].spelling == '++':
                        return PreIncrement(Symbol(child))
                    if tokens[0].spelling == '--':
                        return PreDecrement(Symbol(child))
                    if tokens[0].spelling == '!':
                        return Not(Symbol(child))
                else:
                    if tokens[0].spelling == '+':
                        return child
                    if tokens[0].spelling == '-':
                        return Mul(child, -1)
                    if tokens[0].spelling == '!':
                        return Not(sympify(bool(child)))
            elif tokens[1].spelling in ['++', '--']:
                child = self.transform(next(node.get_children()))
                if tokens[1].spelling == '++':
                    return PostIncrement(Symbol(child))
                if tokens[1].spelling == '--':
                    return PostDecrement(Symbol(child))
            else:
                raise NotImplementedError('Dereferencing operator, Address operator and bitwise NOT operator have not been implemented yet!')

        def transform_binary_operator(self, node):
            if False:
                i = 10
                return i + 15
            'Transformation function for handling binary operators\n\n            Returns\n            =======\n\n            binary_expression: Codegen AST node\n                    simplified binary expression represented as Codegen AST\n\n            Raises\n            ======\n\n            NotImplementedError\n                If a bitwise operator or\n                unary operator(which is a child of any binary\n                operator in Clang AST) is encountered\n\n            '
            tokens = list(node.get_tokens())
            operators_list = ['+', '-', '*', '/', '%', '=', '>', '>=', '<', '<=', '==', '!=', '&&', '||', '+=', '-=', '*=', '/=', '%=']
            combined_variables_stack = []
            operators_stack = []
            for token in tokens:
                if token.kind == cin.TokenKind.PUNCTUATION:
                    if token.spelling == '(':
                        operators_stack.append('(')
                    elif token.spelling == ')':
                        while operators_stack and operators_stack[-1] != '(':
                            if len(combined_variables_stack) < 2:
                                raise NotImplementedError('Unary operators as a part of binary operators is not supported yet!')
                            rhs = combined_variables_stack.pop()
                            lhs = combined_variables_stack.pop()
                            operator = operators_stack.pop()
                            combined_variables_stack.append(self.perform_operation(lhs, rhs, operator))
                        operators_stack.pop()
                    elif token.spelling in operators_list:
                        while operators_stack and self.priority_of(token.spelling) <= self.priority_of(operators_stack[-1]):
                            if len(combined_variables_stack) < 2:
                                raise NotImplementedError('Unary operators as a part of binary operators is not supported yet!')
                            rhs = combined_variables_stack.pop()
                            lhs = combined_variables_stack.pop()
                            operator = operators_stack.pop()
                            combined_variables_stack.append(self.perform_operation(lhs, rhs, operator))
                        operators_stack.append(token.spelling)
                    elif token.spelling in ['&', '|', '^', '<<', '>>']:
                        raise NotImplementedError('Bitwise operator has not been implemented yet!')
                    elif token.spelling in ['&=', '|=', '^=', '<<=', '>>=']:
                        raise NotImplementedError('Shorthand bitwise operator has not been implemented yet!')
                    else:
                        raise NotImplementedError('Given token {} is not implemented yet!'.format(token.spelling))
                elif token.kind == cin.TokenKind.IDENTIFIER:
                    combined_variables_stack.append([token.spelling, 'identifier'])
                elif token.kind == cin.TokenKind.LITERAL:
                    combined_variables_stack.append([token.spelling, 'literal'])
                elif token.kind == cin.TokenKind.KEYWORD and token.spelling in ['true', 'false']:
                    combined_variables_stack.append([token.spelling, 'boolean'])
                else:
                    raise NotImplementedError('Given token {} is not implemented yet!'.format(token.spelling))
            while operators_stack:
                if len(combined_variables_stack) < 2:
                    raise NotImplementedError('Unary operators as a part of binary operators is not supported yet!')
                rhs = combined_variables_stack.pop()
                lhs = combined_variables_stack.pop()
                operator = operators_stack.pop()
                combined_variables_stack.append(self.perform_operation(lhs, rhs, operator))
            return combined_variables_stack[-1][0]

        def priority_of(self, op):
            if False:
                while True:
                    i = 10
            'To get the priority of given operator'
            if op in ['=', '+=', '-=', '*=', '/=', '%=']:
                return 1
            if op in ['&&', '||']:
                return 2
            if op in ['<', '<=', '>', '>=', '==', '!=']:
                return 3
            if op in ['+', '-']:
                return 4
            if op in ['*', '/', '%']:
                return 5
            return 0

        def perform_operation(self, lhs, rhs, op):
            if False:
                return 10
            'Performs operation supported by the SymPy core\n\n            Returns\n            =======\n\n            combined_variable: list\n                contains variable content and type of variable\n\n            '
            lhs_value = self.get_expr_for_operand(lhs)
            rhs_value = self.get_expr_for_operand(rhs)
            if op == '+':
                return [Add(lhs_value, rhs_value), 'expr']
            if op == '-':
                return [Add(lhs_value, -rhs_value), 'expr']
            if op == '*':
                return [Mul(lhs_value, rhs_value), 'expr']
            if op == '/':
                return [Mul(lhs_value, Pow(rhs_value, Integer(-1))), 'expr']
            if op == '%':
                return [Mod(lhs_value, rhs_value), 'expr']
            if op in ['<', '<=', '>', '>=', '==', '!=']:
                return [Rel(lhs_value, rhs_value, op), 'expr']
            if op == '&&':
                return [And(as_Boolean(lhs_value), as_Boolean(rhs_value)), 'expr']
            if op == '||':
                return [Or(as_Boolean(lhs_value), as_Boolean(rhs_value)), 'expr']
            if op == '=':
                return [Assignment(Variable(lhs_value), rhs_value), 'expr']
            if op in ['+=', '-=', '*=', '/=', '%=']:
                return [aug_assign(Variable(lhs_value), op[0], rhs_value), 'expr']

        def get_expr_for_operand(self, combined_variable):
            if False:
                print('Hello World!')
            'Gives out SymPy Codegen AST node\n\n            AST node returned is corresponding to\n            combined variable passed.Combined variable contains\n            variable content and type of variable\n\n            '
            if combined_variable[1] == 'identifier':
                return Symbol(combined_variable[0])
            if combined_variable[1] == 'literal':
                if '.' in combined_variable[0]:
                    return Float(float(combined_variable[0]))
                else:
                    return Integer(int(combined_variable[0]))
            if combined_variable[1] == 'expr':
                return combined_variable[0]
            if combined_variable[1] == 'boolean':
                return true if combined_variable[0] == 'true' else false

        def transform_null_stmt(self, node):
            if False:
                while True:
                    i = 10
            'Handles Null Statement and returns None'
            return none

        def transform_while_stmt(self, node):
            if False:
                while True:
                    i = 10
            'Transformation function for handling while statement\n\n            Returns\n            =======\n\n            while statement : Codegen AST Node\n                contains the while statement node having condition and\n                statement block\n\n            '
            children = node.get_children()
            condition = self.transform(next(children))
            statements = self.transform(next(children))
            if isinstance(statements, list):
                statement_block = CodeBlock(*statements)
            else:
                statement_block = CodeBlock(statements)
            return While(condition, statement_block)
else:

    class CCodeConverter:

        def __init__(self, *args, **kwargs):
            if False:
                return 10
            raise ImportError('Module not Installed')

def parse_c(source):
    if False:
        while True:
            i = 10
    'Function for converting a C source code\n\n    The function reads the source code present in the given file and parses it\n    to give out SymPy Expressions\n\n    Returns\n    =======\n\n    src : list\n        List of Python expression strings\n\n    '
    converter = CCodeConverter()
    if os.path.exists(source):
        src = converter.parse(source, flags=[])
    else:
        src = converter.parse_str(source, flags=[])
    return src