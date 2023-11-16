"""A parser that generates a parse tree for math expressions.

It uses the following grammar in Backus-Naur form:

# Non-terminals
<expr> ::= <mul_expr> (('+' | '-') <mul_expr>)*
<mul_expr> ::= <pow_expr> (('*' | '/') <pow_expr>)*
<pow_expr> ::= '-' <pow_expr> | '+' <pow_expr> | <unit> ('^' <pow_expr>)?
<unit> ::= <identifier> | <number> | '(' <expr> ')' | <function> '(' <expr> ')'

# Terminals
<number> ::= r'[0-9]+.[0-9]+|[0-9]+'
<identifier> ::= r'[a-zA-Z]' | 'alpha' | 'beta' | 'gamma' | 'theta' | 'epsilon'
| 'pi' | 'omega'
<function> ::= 'sqrt' | 'abs' | 'cos' | 'sin' | 'tan' | 'cot' | 'sec' | 'cosec'
"""
from __future__ import annotations
import collections
import re
from core.constants import constants
from typing import Final, List, Optional
_OPENING_PARENS: List[str] = ['[', '{', '(']
_CLOSING_PARENS: List[str] = [')', '}', ']']
_VALID_OPERATORS: List[str] = _OPENING_PARENS + _CLOSING_PARENS + ['+', '-', '/', '*', '^']
_TOKEN_CATEGORY_IDENTIFIER: Final = 'identifier'
_TOKEN_CATEGORY_FUNCTION: Final = 'function'
_TOKEN_CATEGORY_NUMBER: Final = 'number'
_TOKEN_CATEGORY_OPERATOR: Final = 'operator'
_OPENING_CATEGORIES: Final = (_TOKEN_CATEGORY_IDENTIFIER, _TOKEN_CATEGORY_FUNCTION, _TOKEN_CATEGORY_NUMBER)
_CLOSING_CATEGORIES: Final = (_TOKEN_CATEGORY_IDENTIFIER, _TOKEN_CATEGORY_NUMBER)

def contains_balanced_brackets(expression: str) -> bool:
    if False:
        while True:
            i = 10
    'Checks if the given expression contains a balanced bracket sequence.\n\n    Args:\n        expression: str. A math expression (algebraic/numeric).\n\n    Returns:\n        bool. Whether the given expression contains a balanced bracket sequence.\n    '
    (openers, closers) = ('({[', ')}]')
    stack = []
    for character in expression:
        if character in openers:
            stack.append(character)
        elif character in closers:
            if len(stack) == 0:
                return False
            top_element = stack.pop()
            if openers.index(top_element) != closers.index(character):
                return False
    return len(stack) == 0

def contains_at_least_one_variable(expression: str) -> bool:
    if False:
        print('Hello World!')
    'Checks if the given expression contains at least one valid identifier\n    (latin letter or greek symbol name).\n\n    Args:\n        expression: str. A math expression.\n\n    Returns:\n        bool. Whether the given expression contains at least one single latin\n        letter or greek symbol name.\n\n    Raises:\n        Exception. Invalid syntax.\n    '
    Parser().parse(expression)
    token_list = tokenize(expression)
    return any((token.category == _TOKEN_CATEGORY_IDENTIFIER for token in token_list))

def tokenize(expression: str) -> List[Token]:
    if False:
        i = 10
        return i + 15
    'Splits the given expression into separate tokens based on the grammar\n    definitions.\n\n    Args:\n        expression: str. A math expression.\n\n    Returns:\n        list(Token). A list containing token objects formed from the given math\n        expression.\n\n    Raises:\n        Exception. Invalid token.\n    '
    expression = expression.replace(' ', '')
    re_string = '(%s|[a-zA-Z]|[0-9]+\\.[0-9]+|[0-9]+|[%s])' % ('|'.join(sorted(list(constants.GREEK_LETTER_NAMES_TO_SYMBOLS.keys()) + constants.MATH_FUNCTION_NAMES, reverse=True, key=len)), '\\'.join(_VALID_OPERATORS))
    token_texts = re.findall(re_string, expression)
    original_exp_frequency = collections.Counter(expression)
    tokenized_exp_frequency = collections.Counter(''.join(token_texts))
    for character in original_exp_frequency:
        if original_exp_frequency[character] != tokenized_exp_frequency[character]:
            raise Exception('Invalid token: %s.' % character)
    token_list = []
    for token_text in token_texts:
        if token_text in ['[', '{']:
            token_list.append(Token('('))
        elif token_text in [']', '}']:
            token_list.append(Token(')'))
        else:
            token_list.append(Token(token_text))
    final_token_list = []
    for (i, token) in enumerate(token_list):
        final_token_list.append(token)
        if i != len(token_list) - 1:
            if (token.category in _CLOSING_CATEGORIES or token.text in _CLOSING_PARENS) and (token_list[i + 1].category in _OPENING_CATEGORIES or token_list[i + 1].text in _OPENING_PARENS):
                final_token_list.append(Token('*'))
    return final_token_list

def get_variables(expression: str) -> List[str]:
    if False:
        return 10
    'Extracts all variables along with pi and e from a given expression.\n\n    Args:\n        expression: str. A math expression.\n\n    Returns:\n        list(str). A list containing all the variables present in the given\n        expression.\n    '
    if '=' in expression:
        (lhs, rhs) = expression.split('=')
        token_list = tokenize(lhs) + tokenize(rhs)
    else:
        token_list = tokenize(expression)
    variables = set()
    for token in token_list:
        if token.category == _TOKEN_CATEGORY_IDENTIFIER or token.text in ['pi', 'e']:
            variables.add(token.text)
    return list(variables)

class Token:
    """Class for tokens of the math expression."""

    def __init__(self, text: str) -> None:
        if False:
            return 10
        'Initializes a Token object.\n\n        Args:\n            text: str. String representation of the token.\n\n        Raises:\n            Exception. Invalid token.\n        '
        self.text = text
        if self.is_number(text):
            self.category = _TOKEN_CATEGORY_NUMBER
        elif self.is_identifier(text):
            self.category = _TOKEN_CATEGORY_IDENTIFIER
        elif self.is_function(text):
            self.category = _TOKEN_CATEGORY_FUNCTION
        elif self.is_operator(text):
            self.category = _TOKEN_CATEGORY_OPERATOR
        else:
            raise Exception('Invalid token: %s.' % text)

    def is_function(self, text: str) -> bool:
        if False:
            i = 10
            return i + 15
        'Checks if the given token represents a valid math function.\n\n        Args:\n            text: str. String representation of the token.\n\n        Returns:\n            bool. Whether the given string represents a valid math function.\n        '
        return text in constants.MATH_FUNCTION_NAMES

    def is_identifier(self, text: str) -> bool:
        if False:
            return 10
        'Checks if the given token represents a valid identifier. A valid\n        identifier could be a single latin letter (uppercase/lowercase) or a\n        greek letter represented by the symbol name.\n\n        Args:\n            text: str. String representation of the token.\n\n        Returns:\n            bool. Whether the given string represents a valid identifier.\n        '
        return text in constants.VALID_ALGEBRAIC_IDENTIFIERS

    def is_number(self, text: str) -> bool:
        if False:
            while True:
                i = 10
        "Checks if the given token represents a valid real number without a\n        '+'/'-' sign. 'pi' and 'e' are also considered as numeric values.\n\n        Args:\n            text: str. String representation of the token.\n\n        Returns:\n            bool. Whether the given string represents a valid real number.\n        "
        return text.replace('.', '', 1).isdigit() or text in ('pi', 'e')

    def is_operator(self, text: str) -> bool:
        if False:
            i = 10
            return i + 15
        'Checks if the given token represents a valid math operator.\n\n        Args:\n            text: str. String representation of the token.\n\n        Returns:\n            bool. Whether the given string represents a valid math operator.\n        '
        return text in _VALID_OPERATORS

class Node:
    """Instances of the classes that inherit this class act as nodes of the
    parse tree. These could be internal as well as leaf nodes. For leaf nodes,
    the children parameter would be an empty list.

    NOTE: This class is not supposed to be used independently, but should be
    inherited. If the child class represents an identifier or a function, it
    should have an attribute that denotes the text that the node represents. For
    the operator nodes, the class name should represent the type of operator.
    """

    def __init__(self, children: List[Node]) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Initializes a Node object. For ex. 'a + b' will have root node as\n        '+' and children as ['a', 'b'].\n\n        Args:\n            children: list(Node). Child nodes of the current node.\n        "
        self.children = children

class AdditionOperatorNode(Node):
    """Class representing the addition operator node."""

    def __init__(self, left: Node, right: Node) -> None:
        if False:
            print('Hello World!')
        'Initializes an AdditionOperatorNode object.\n\n        Args:\n            left: Node. Left child of the operator.\n            right: Node. Right child of the operator.\n        '
        super().__init__([left, right])

class SubtractionOperatorNode(Node):
    """Class representing the subtraction operator node."""

    def __init__(self, left: Node, right: Node) -> None:
        if False:
            i = 10
            return i + 15
        'Initializes an SubtractionOperatorNode object.\n\n        Args:\n            left: Node. Left child of the operator.\n            right: Node. Right child of the operator.\n        '
        super().__init__([left, right])

class MultiplicationOperatorNode(Node):
    """Class representing the multiplication operator node."""

    def __init__(self, left: Node, right: Node) -> None:
        if False:
            return 10
        'Initializes an MultiplicationOperatorNode object.\n\n        Args:\n            left: Node. Left child of the operator.\n            right: Node. Right child of the operator.\n        '
        super().__init__([left, right])

class DivisionOperatorNode(Node):
    """Class representing the division operator node."""

    def __init__(self, left: Node, right: Node) -> None:
        if False:
            print('Hello World!')
        'Initializes an DivisionOperatorNode object.\n\n        Args:\n            left: Node. Left child of the operator.\n            right: Node. Right child of the operator.\n        '
        super().__init__([left, right])

class PowerOperatorNode(Node):
    """Class representing the power operator node."""

    def __init__(self, left: Node, right: Node) -> None:
        if False:
            i = 10
            return i + 15
        'Initializes an PowerOperatorNode object.\n\n        Args:\n            left: Node. Left child of the operator.\n            right: Node. Right child of the operator.\n        '
        super().__init__([left, right])

class IdentifierNode(Node):
    """Class representing the identifier node. An identifier could be a single
    latin letter (uppercase/lowercase) or a greek letter represented by the
    symbol name.
    """

    def __init__(self, token: Token) -> None:
        if False:
            return 10
        'Initializes an IdentifierNode object.\n\n        Args:\n            token: Token. The token representing the identifier.\n        '
        self.token = token
        super().__init__([])

class NumberNode(Node):
    """Class representing the number node."""

    def __init__(self, token: Token) -> None:
        if False:
            while True:
                i = 10
        'Initializes a NumberNode object.\n\n        Args:\n            token: Token. The token representing a real number.\n        '
        self.token = token
        super().__init__([])

class UnaryFunctionNode(Node):
    """Class representing the function node. The functions represented by this
    class must have exactly one parameter.
    """

    def __init__(self, token: Token, child: Node) -> None:
        if False:
            return 10
        'Initializes a UnaryFunctionNode object.\n\n        Args:\n            token: Token. The token representing the math function.\n            child: Node. The parameter of the function.\n        '
        self.token = token
        super().__init__([child])

class Parser:
    """Class representing the math expression parser.
    Implements a greedy, recursive-descent parser that tries to consume
    as many tokens as possible while obeying the grammar.
    More info about recursive-descent parsers:
    https://en.wikipedia.org/wiki/Recursive_descent_parser
    """

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        'Initializes the Parser object.'
        self._next_token_index = 0

    def parse(self, expression: str) -> Node:
        if False:
            return 10
        'A wrapper around the _parse_expr method. This method creates a list\n        of tokens present in the expression and calls the _parse_expr method.\n\n        Args:\n            expression: str. String representing the math expression.\n\n        Returns:\n            Node. Root node of the generated parse tree.\n\n        Raises:\n            Exception. Invalid syntax: Unexpected end of expression.\n            Exception. Invalid character.\n            Exception. Invalid bracket pairing.\n        '
        for character in expression:
            if not bool(re.match('(\\s|\\d|\\w|\\.)', character)) and character not in _VALID_OPERATORS:
                raise Exception('Invalid character: %s.' % character)
        if not contains_balanced_brackets(expression):
            raise Exception('Invalid bracket pairing.')
        token_list = tokenize(expression)
        self._next_token_index = 0
        return self._parse_expr(token_list)

    def _parse_expr(self, token_list: List[Token]) -> Node:
        if False:
            print('Hello World!')
        "Function representing the following production rule of the grammar:\n        <expr> ::= <mul_expr> (('+' | '-') <mul_expr>)*\n\n        Args:\n            token_list: list(Token). A list containing token objects formed from\n                the given math expression.\n\n        Returns:\n            Node. Root node of the generated parse tree.\n        "
        parsed_expr = self._parse_mul_expr(token_list)
        operator_token = self._get_next_token_if_text_in(['+', '-'], token_list)
        while operator_token:
            parsed_right = self._parse_mul_expr(token_list)
            if operator_token.text == '+':
                parsed_expr = AdditionOperatorNode(parsed_expr, parsed_right)
            else:
                parsed_expr = SubtractionOperatorNode(parsed_expr, parsed_right)
            operator_token = self._get_next_token_if_text_in(['+', '-'], token_list)
        return parsed_expr

    def _parse_mul_expr(self, token_list: List[Token]) -> Node:
        if False:
            i = 10
            return i + 15
        "Function representing the following production rule of the grammar:\n        <mul_expr> ::= <pow_expr> (('*' | '/') <pow_expr>)*\n\n        Args:\n            token_list: list(Token). A list containing token objects formed from\n                the given math expression.\n\n        Returns:\n            Node. Root node of the generated parse tree.\n        "
        parsed_expr = self._parse_pow_expr(token_list)
        operator_token = self._get_next_token_if_text_in(['*', '/'], token_list)
        while operator_token:
            parsed_right = self._parse_pow_expr(token_list)
            if operator_token.text == '*':
                parsed_expr = MultiplicationOperatorNode(parsed_expr, parsed_right)
            else:
                parsed_expr = DivisionOperatorNode(parsed_expr, parsed_right)
            operator_token = self._get_next_token_if_text_in(['*', '/'], token_list)
        return parsed_expr

    def _parse_pow_expr(self, token_list: List[Token]) -> Node:
        if False:
            while True:
                i = 10
        "Function representing the following production rule of the grammar:\n        <pow_expr> ::= '-' <pow_expr> | '+' <pow_expr> |\n        <unit> ('^' <pow_expr>)?\n\n        Args:\n            token_list: list(Token). A list containing token objects formed from\n                the given math expression.\n\n        Returns:\n            Node. Root node of the generated parse tree.\n        "
        while self._get_next_token_if_text_in(['+', '-'], token_list):
            pass
        parsed_expr = self._parse_unit(token_list)
        operator_token = self._get_next_token_if_text_in(['^'], token_list)
        if operator_token:
            parsed_right = self._parse_pow_expr(token_list)
            return PowerOperatorNode(parsed_expr, parsed_right)
        return parsed_expr

    def _parse_unit(self, token_list: List[Token]) -> Node:
        if False:
            while True:
                i = 10
        "Function representing the following production rule of the grammar:\n        <unit> ::= <identifier> | <number> | '(' <expr> ')' |\n        <function> '(' <expr> ')'\n\n        Args:\n            token_list: list(Token). A list containing token objects formed from\n                the given math expression.\n\n        Returns:\n            Node. Root node of the generated parse tree.\n\n        Raises:\n            Exception. Invalid token.\n        "
        token = self._get_next_token(token_list)
        if token.category == _TOKEN_CATEGORY_IDENTIFIER:
            return IdentifierNode(token)
        if token.category == _TOKEN_CATEGORY_FUNCTION:
            if self._get_next_token_if_text_in(['('], token_list):
                parsed_child = self._parse_expr(token_list)
                next_token = self._get_next_token_if_text_in([')'], token_list)
                assert next_token is not None
                return UnaryFunctionNode(next_token, parsed_child)
        if token.category == _TOKEN_CATEGORY_NUMBER:
            return NumberNode(token)
        if token.text == '(':
            parsed_expr = self._parse_expr(token_list)
            next_token = self._get_next_token_if_text_in([')'], token_list)
            return parsed_expr
        raise Exception('Invalid token: %s.' % token.text)

    def _get_next_token(self, token_list: List[Token]) -> Token:
        if False:
            print('Hello World!')
        'Function to retrieve the token at the next position and then\n        increment the _next_token_index.\n\n        Args:\n            token_list: list(Token). A list containing token objects formed from\n                the given math expression.\n\n        Returns:\n            Token. Token at the next position.\n\n        Raises:\n            Exception. Invalid syntax: Unexpected end of expression.\n        '
        if self._next_token_index < len(token_list):
            token = token_list[self._next_token_index]
            self._next_token_index += 1
            return token
        raise Exception('Invalid syntax: Unexpected end of expression.')

    def _get_next_token_if_text_in(self, allowed_token_texts: List[str], token_list: List[Token]) -> Optional[Token]:
        if False:
            while True:
                i = 10
        'Function to verify that there is at least one more token remaining\n        and that the next token text is among the allowed_token_texts provided.\n        If true, returns the token; otherwise, returns None.\n\n        Args:\n            allowed_token_texts: list(str). List of strings containing the\n                allowed token texts at the next position.\n            token_list: list(Token). A list containing token objects formed from\n                the given math expression.\n\n        Returns:\n            Token|None. Token at the next position. Returns None if there are no\n            more tokens left or the next token text is not in the\n            allowed_token_texts.\n        '
        if self._next_token_index < len(token_list):
            text = token_list[self._next_token_index].text
            if text in allowed_token_texts:
                token = token_list[self._next_token_index]
                self._next_token_index += 1
                return token
        return None

def is_valid_expression(expression: str) -> bool:
    if False:
        while True:
            i = 10
    'Checks if the given math expression is syntactically valid.\n\n    Args:\n        expression: str. String representation of the math expression.\n\n    Returns:\n        bool. Whether the given math expression is syntactically valid.\n    '
    try:
        Parser().parse(expression)
    except Exception:
        return False
    return True