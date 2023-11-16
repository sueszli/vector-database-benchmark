"""
Handling Ambiguity
==================

A demonstration of ambiguity

This example shows how to use get explicit ambiguity from Lark's Earley parser.

"""
import sys
from lark import Lark, tree
grammar = '\n    sentence: noun verb noun        -> simple\n            | noun verb "like" noun -> comparative\n\n    noun: adj? NOUN\n    verb: VERB\n    adj: ADJ\n\n    NOUN: "flies" | "bananas" | "fruit"\n    VERB: "like" | "flies"\n    ADJ: "fruit"\n\n    %import common.WS\n    %ignore WS\n'
parser = Lark(grammar, start='sentence', ambiguity='explicit')
sentence = 'fruit flies like bananas'

def make_png(filename):
    if False:
        for i in range(10):
            print('nop')
    tree.pydot__tree_to_png(parser.parse(sentence), filename)

def make_dot(filename):
    if False:
        for i in range(10):
            print('nop')
    tree.pydot__tree_to_dot(parser.parse(sentence), filename)
if __name__ == '__main__':
    print(parser.parse(sentence).pretty())