"""
Turtle DSL
==========

Implements a LOGO-like toy language for Python’s turtle, with interpreter.
"""
try:
    input = raw_input
except NameError:
    pass
import turtle
from lark import Lark
turtle_grammar = '\n    start: instruction+\n\n    instruction: MOVEMENT NUMBER            -> movement\n               | "c" COLOR [COLOR]          -> change_color\n               | "fill" code_block          -> fill\n               | "repeat" NUMBER code_block -> repeat\n\n    code_block: "{" instruction+ "}"\n\n    MOVEMENT: "f"|"b"|"l"|"r"\n    COLOR: LETTER+\n\n    %import common.LETTER\n    %import common.INT -> NUMBER\n    %import common.WS\n    %ignore WS\n'
parser = Lark(turtle_grammar)

def run_instruction(t):
    if False:
        i = 10
        return i + 15
    if t.data == 'change_color':
        turtle.color(*t.children)
    elif t.data == 'movement':
        (name, number) = t.children
        {'f': turtle.fd, 'b': turtle.bk, 'l': turtle.lt, 'r': turtle.rt}[name](int(number))
    elif t.data == 'repeat':
        (count, block) = t.children
        for i in range(int(count)):
            run_instruction(block)
    elif t.data == 'fill':
        turtle.begin_fill()
        run_instruction(t.children[0])
        turtle.end_fill()
    elif t.data == 'code_block':
        for cmd in t.children:
            run_instruction(cmd)
    else:
        raise SyntaxError('Unknown instruction: %s' % t.data)

def run_turtle(program):
    if False:
        for i in range(10):
            print('nop')
    parse_tree = parser.parse(program)
    for inst in parse_tree.children:
        run_instruction(inst)

def main():
    if False:
        return 10
    while True:
        code = input('> ')
        try:
            run_turtle(code)
        except Exception as e:
            print(e)

def test():
    if False:
        for i in range(10):
            print('nop')
    text = '\n        c red yellow\n        fill { repeat 36 {\n            f200 l170\n        }}\n    '
    run_turtle(text)
if __name__ == '__main__':
    main()