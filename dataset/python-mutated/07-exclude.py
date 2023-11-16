from __future__ import print_function
from __future__ import unicode_literals
from builtins import str, bytes, dict, int
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from pattern.search import match
from pattern.en import Sentence, parse

def imperative(sentence):
    if False:
        print('Hello World!')
    for p in ('!could|!would|!should|!to+ VB', '^VB', '^do|VB*'):
        m = match(p, sentence)
        if match(p, sentence) and sentence.string.endswith(('.', '!')):
            return True
    return False
for s in ('Just stop it!', 'Look out!', 'Do your homework!', 'You should do your homework.', 'Could you stop it.', 'To be, or not to be.'):
    s = parse(s)
    s = Sentence(s)
    print(s)
    print(imperative(s))
    print('')