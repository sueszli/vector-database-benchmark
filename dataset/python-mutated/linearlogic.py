from nltk.internals import Counter
from nltk.sem.logic import APP, LogicParser
_counter = Counter()

class Tokens:
    OPEN = '('
    CLOSE = ')'
    IMP = '-o'
    PUNCT = [OPEN, CLOSE]
    TOKENS = PUNCT + [IMP]

class LinearLogicParser(LogicParser):
    """A linear logic expression parser."""

    def __init__(self):
        if False:
            return 10
        LogicParser.__init__(self)
        self.operator_precedence = {APP: 1, Tokens.IMP: 2, None: 3}
        self.right_associated_operations += [Tokens.IMP]

    def get_all_symbols(self):
        if False:
            print('Hello World!')
        return Tokens.TOKENS

    def handle(self, tok, context):
        if False:
            while True:
                i = 10
        if tok not in Tokens.TOKENS:
            return self.handle_variable(tok, context)
        elif tok == Tokens.OPEN:
            return self.handle_open(tok, context)

    def get_BooleanExpression_factory(self, tok):
        if False:
            for i in range(10):
                print('nop')
        if tok == Tokens.IMP:
            return ImpExpression
        else:
            return None

    def make_BooleanExpression(self, factory, first, second):
        if False:
            return 10
        return factory(first, second)

    def attempt_ApplicationExpression(self, expression, context):
        if False:
            i = 10
            return i + 15
        'Attempt to make an application expression.  If the next tokens\n        are an argument in parens, then the argument expression is a\n        function being applied to the arguments.  Otherwise, return the\n        argument expression.'
        if self.has_priority(APP, context):
            if self.inRange(0) and self.token(0) == Tokens.OPEN:
                self.token()
                argument = self.process_next_expression(APP)
                self.assertNextToken(Tokens.CLOSE)
                expression = ApplicationExpression(expression, argument, None)
        return expression

    def make_VariableExpression(self, name):
        if False:
            while True:
                i = 10
        if name[0].isupper():
            return VariableExpression(name)
        else:
            return ConstantExpression(name)

class Expression:
    _linear_logic_parser = LinearLogicParser()

    @classmethod
    def fromstring(cls, s):
        if False:
            i = 10
            return i + 15
        return cls._linear_logic_parser.parse(s)

    def applyto(self, other, other_indices=None):
        if False:
            return 10
        return ApplicationExpression(self, other, other_indices)

    def __call__(self, other):
        if False:
            return 10
        return self.applyto(other)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return f'<{self.__class__.__name__} {self}>'

class AtomicExpression(Expression):

    def __init__(self, name, dependencies=None):
        if False:
            return 10
        '\n        :param name: str for the constant name\n        :param dependencies: list of int for the indices on which this atom is dependent\n        '
        assert isinstance(name, str)
        self.name = name
        if not dependencies:
            dependencies = []
        self.dependencies = dependencies

    def simplify(self, bindings=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        If 'self' is bound by 'bindings', return the atomic to which it is bound.\n        Otherwise, return self.\n\n        :param bindings: ``BindingDict`` A dictionary of bindings used to simplify\n        :return: ``AtomicExpression``\n        "
        if bindings and self in bindings:
            return bindings[self]
        else:
            return self

    def compile_pos(self, index_counter, glueFormulaFactory):
        if False:
            for i in range(10):
                print('nop')
        "\n        From Iddo Lev's PhD Dissertation p108-109\n\n        :param index_counter: ``Counter`` for unique indices\n        :param glueFormulaFactory: ``GlueFormula`` for creating new glue formulas\n        :return: (``Expression``,set) for the compiled linear logic and any newly created glue formulas\n        "
        self.dependencies = []
        return (self, [])

    def compile_neg(self, index_counter, glueFormulaFactory):
        if False:
            i = 10
            return i + 15
        "\n        From Iddo Lev's PhD Dissertation p108-109\n\n        :param index_counter: ``Counter`` for unique indices\n        :param glueFormulaFactory: ``GlueFormula`` for creating new glue formulas\n        :return: (``Expression``,set) for the compiled linear logic and any newly created glue formulas\n        "
        self.dependencies = []
        return (self, [])

    def initialize_labels(self, fstruct):
        if False:
            return 10
        self.name = fstruct.initialize_label(self.name.lower())

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        return self.__class__ == other.__class__ and self.name == other.name

    def __ne__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return not self == other

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        accum = self.name
        if self.dependencies:
            accum += '%s' % self.dependencies
        return accum

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return hash(self.name)

class ConstantExpression(AtomicExpression):

    def unify(self, other, bindings):
        if False:
            print('Hello World!')
        "\n        If 'other' is a constant, then it must be equal to 'self'.  If 'other' is a variable,\n        then it must not be bound to anything other than 'self'.\n\n        :param other: ``Expression``\n        :param bindings: ``BindingDict`` A dictionary of all current bindings\n        :return: ``BindingDict`` A new combined dictionary of of 'bindings' and any new binding\n        :raise UnificationException: If 'self' and 'other' cannot be unified in the context of 'bindings'\n        "
        assert isinstance(other, Expression)
        if isinstance(other, VariableExpression):
            try:
                return bindings + BindingDict([(other, self)])
            except VariableBindingException:
                pass
        elif self == other:
            return bindings
        raise UnificationException(self, other, bindings)

class VariableExpression(AtomicExpression):

    def unify(self, other, bindings):
        if False:
            return 10
        "\n        'self' must not be bound to anything other than 'other'.\n\n        :param other: ``Expression``\n        :param bindings: ``BindingDict`` A dictionary of all current bindings\n        :return: ``BindingDict`` A new combined dictionary of of 'bindings' and the new binding\n        :raise UnificationException: If 'self' and 'other' cannot be unified in the context of 'bindings'\n        "
        assert isinstance(other, Expression)
        try:
            if self == other:
                return bindings
            else:
                return bindings + BindingDict([(self, other)])
        except VariableBindingException as e:
            raise UnificationException(self, other, bindings) from e

class ImpExpression(Expression):

    def __init__(self, antecedent, consequent):
        if False:
            print('Hello World!')
        '\n        :param antecedent: ``Expression`` for the antecedent\n        :param consequent: ``Expression`` for the consequent\n        '
        assert isinstance(antecedent, Expression)
        assert isinstance(consequent, Expression)
        self.antecedent = antecedent
        self.consequent = consequent

    def simplify(self, bindings=None):
        if False:
            while True:
                i = 10
        return self.__class__(self.antecedent.simplify(bindings), self.consequent.simplify(bindings))

    def unify(self, other, bindings):
        if False:
            i = 10
            return i + 15
        "\n        Both the antecedent and consequent of 'self' and 'other' must unify.\n\n        :param other: ``ImpExpression``\n        :param bindings: ``BindingDict`` A dictionary of all current bindings\n        :return: ``BindingDict`` A new combined dictionary of of 'bindings' and any new bindings\n        :raise UnificationException: If 'self' and 'other' cannot be unified in the context of 'bindings'\n        "
        assert isinstance(other, ImpExpression)
        try:
            return bindings + self.antecedent.unify(other.antecedent, bindings) + self.consequent.unify(other.consequent, bindings)
        except VariableBindingException as e:
            raise UnificationException(self, other, bindings) from e

    def compile_pos(self, index_counter, glueFormulaFactory):
        if False:
            i = 10
            return i + 15
        "\n        From Iddo Lev's PhD Dissertation p108-109\n\n        :param index_counter: ``Counter`` for unique indices\n        :param glueFormulaFactory: ``GlueFormula`` for creating new glue formulas\n        :return: (``Expression``,set) for the compiled linear logic and any newly created glue formulas\n        "
        (a, a_new) = self.antecedent.compile_neg(index_counter, glueFormulaFactory)
        (c, c_new) = self.consequent.compile_pos(index_counter, glueFormulaFactory)
        return (ImpExpression(a, c), a_new + c_new)

    def compile_neg(self, index_counter, glueFormulaFactory):
        if False:
            for i in range(10):
                print('nop')
        "\n        From Iddo Lev's PhD Dissertation p108-109\n\n        :param index_counter: ``Counter`` for unique indices\n        :param glueFormulaFactory: ``GlueFormula`` for creating new glue formulas\n        :return: (``Expression``,list of ``GlueFormula``) for the compiled linear logic and any newly created glue formulas\n        "
        (a, a_new) = self.antecedent.compile_pos(index_counter, glueFormulaFactory)
        (c, c_new) = self.consequent.compile_neg(index_counter, glueFormulaFactory)
        fresh_index = index_counter.get()
        c.dependencies.append(fresh_index)
        new_v = glueFormulaFactory('v%s' % fresh_index, a, {fresh_index})
        return (c, a_new + c_new + [new_v])

    def initialize_labels(self, fstruct):
        if False:
            for i in range(10):
                print('nop')
        self.antecedent.initialize_labels(fstruct)
        self.consequent.initialize_labels(fstruct)

    def __eq__(self, other):
        if False:
            return 10
        return self.__class__ == other.__class__ and self.antecedent == other.antecedent and (self.consequent == other.consequent)

    def __ne__(self, other):
        if False:
            return 10
        return not self == other

    def __str__(self):
        if False:
            return 10
        return '{}{} {} {}{}'.format(Tokens.OPEN, self.antecedent, Tokens.IMP, self.consequent, Tokens.CLOSE)

    def __hash__(self):
        if False:
            return 10
        return hash(f'{hash(self.antecedent)}{Tokens.IMP}{hash(self.consequent)}')

class ApplicationExpression(Expression):

    def __init__(self, function, argument, argument_indices=None):
        if False:
            print('Hello World!')
        "\n        :param function: ``Expression`` for the function\n        :param argument: ``Expression`` for the argument\n        :param argument_indices: set for the indices of the glue formula from which the argument came\n        :raise LinearLogicApplicationException: If 'function' cannot be applied to 'argument' given 'argument_indices'.\n        "
        function_simp = function.simplify()
        argument_simp = argument.simplify()
        assert isinstance(function_simp, ImpExpression)
        assert isinstance(argument_simp, Expression)
        bindings = BindingDict()
        try:
            if isinstance(function, ApplicationExpression):
                bindings += function.bindings
            if isinstance(argument, ApplicationExpression):
                bindings += argument.bindings
            bindings += function_simp.antecedent.unify(argument_simp, bindings)
        except UnificationException as e:
            raise LinearLogicApplicationException(f'Cannot apply {function_simp} to {argument_simp}. {e}') from e
        if argument_indices:
            if not set(function_simp.antecedent.dependencies) < argument_indices:
                raise LinearLogicApplicationException('Dependencies unfulfilled when attempting to apply Linear Logic formula %s to %s' % (function_simp, argument_simp))
            if set(function_simp.antecedent.dependencies) == argument_indices:
                raise LinearLogicApplicationException('Dependencies not a proper subset of indices when attempting to apply Linear Logic formula %s to %s' % (function_simp, argument_simp))
        self.function = function
        self.argument = argument
        self.bindings = bindings

    def simplify(self, bindings=None):
        if False:
            print('Hello World!')
        '\n        Since function is an implication, return its consequent.  There should be\n        no need to check that the application is valid since the checking is done\n        by the constructor.\n\n        :param bindings: ``BindingDict`` A dictionary of bindings used to simplify\n        :return: ``Expression``\n        '
        if not bindings:
            bindings = self.bindings
        return self.function.simplify(bindings).consequent

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        return self.__class__ == other.__class__ and self.function == other.function and (self.argument == other.argument)

    def __ne__(self, other):
        if False:
            print('Hello World!')
        return not self == other

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return '%s' % self.function + Tokens.OPEN + '%s' % self.argument + Tokens.CLOSE

    def __hash__(self):
        if False:
            while True:
                i = 10
        return hash(f'{hash(self.antecedent)}{Tokens.OPEN}{hash(self.consequent)}')

class BindingDict:

    def __init__(self, bindings=None):
        if False:
            i = 10
            return i + 15
        '\n        :param bindings:\n            list [(``VariableExpression``, ``AtomicExpression``)] to initialize the dictionary\n            dict {``VariableExpression``: ``AtomicExpression``} to initialize the dictionary\n        '
        self.d = {}
        if isinstance(bindings, dict):
            bindings = bindings.items()
        if bindings:
            for (v, b) in bindings:
                self[v] = b

    def __setitem__(self, variable, binding):
        if False:
            i = 10
            return i + 15
        "\n        A binding is consistent with the dict if its variable is not already bound, OR if its\n        variable is already bound to its argument.\n\n        :param variable: ``VariableExpression`` The variable bind\n        :param binding: ``Expression`` The expression to which 'variable' should be bound\n        :raise VariableBindingException: If the variable cannot be bound in this dictionary\n        "
        assert isinstance(variable, VariableExpression)
        assert isinstance(binding, Expression)
        assert variable != binding
        existing = self.d.get(variable, None)
        if not existing or binding == existing:
            self.d[variable] = binding
        else:
            raise VariableBindingException('Variable %s already bound to another value' % variable)

    def __getitem__(self, variable):
        if False:
            print('Hello World!')
        "\n        Return the expression to which 'variable' is bound\n        "
        assert isinstance(variable, VariableExpression)
        intermediate = self.d[variable]
        while intermediate:
            try:
                intermediate = self.d[intermediate]
            except KeyError:
                return intermediate

    def __contains__(self, item):
        if False:
            return 10
        return item in self.d

    def __add__(self, other):
        if False:
            while True:
                i = 10
        '\n        :param other: ``BindingDict`` The dict with which to combine self\n        :return: ``BindingDict`` A new dict containing all the elements of both parameters\n        :raise VariableBindingException: If the parameter dictionaries are not consistent with each other\n        '
        try:
            combined = BindingDict()
            for v in self.d:
                combined[v] = self.d[v]
            for v in other.d:
                combined[v] = other.d[v]
            return combined
        except VariableBindingException as e:
            raise VariableBindingException('Attempting to add two contradicting VariableBindingsLists: %s, %s' % (self, other)) from e

    def __ne__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return not self == other

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        if not isinstance(other, BindingDict):
            raise TypeError
        return self.d == other.d

    def __str__(self):
        if False:
            return 10
        return '{' + ', '.join((f'{v}: {self.d[v]}' for v in sorted(self.d.keys()))) + '}'

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'BindingDict: %s' % self

class VariableBindingException(Exception):
    pass

class UnificationException(Exception):

    def __init__(self, a, b, bindings):
        if False:
            while True:
                i = 10
        Exception.__init__(self, f'Cannot unify {a} with {b} given {bindings}')

class LinearLogicApplicationException(Exception):
    pass

def demo():
    if False:
        return 10
    lexpr = Expression.fromstring
    print(lexpr('f'))
    print(lexpr('(g -o f)'))
    print(lexpr('((g -o G) -o G)'))
    print(lexpr('g -o h -o f'))
    print(lexpr('(g -o f)(g)').simplify())
    print(lexpr('(H -o f)(g)').simplify())
    print(lexpr('((g -o G) -o G)((g -o f))').simplify())
    print(lexpr('(H -o H)((g -o f))').simplify())
if __name__ == '__main__':
    demo()