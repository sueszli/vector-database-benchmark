"""
Transform a Forest
==================

This example demonstrates how to subclass ``TreeForestTransformer`` to
directly transform a SPPF.
"""
from lark import Lark
from lark.parsers.earley_forest import TreeForestTransformer, handles_ambiguity, Discard

class CustomTransformer(TreeForestTransformer):

    @handles_ambiguity
    def sentence(self, trees):
        if False:
            i = 10
            return i + 15
        return next((tree for tree in trees if tree.data == 'simple'))

    def simple(self, children):
        if False:
            print('Hello World!')
        children.append('.')
        return self.tree_class('simple', children)

    def adj(self, children):
        if False:
            while True:
                i = 10
        return Discard

    def __default_token__(self, token):
        if False:
            print('Hello World!')
        return token.capitalize()
grammar = '\n    sentence: noun verb noun        -> simple\n            | noun verb "like" noun -> comparative\n\n    noun: adj? NOUN\n    verb: VERB\n    adj: ADJ\n\n    NOUN: "flies" | "bananas" | "fruit"\n    VERB: "like" | "flies"\n    ADJ: "fruit"\n\n    %import common.WS\n    %ignore WS\n'
parser = Lark(grammar, start='sentence', ambiguity='forest')
sentence = 'fruit flies like bananas'
forest = parser.parse(sentence)
tree = CustomTransformer(resolve_ambiguity=False).transform(forest)
print(tree.pretty())