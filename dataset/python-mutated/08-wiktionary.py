from __future__ import print_function
from __future__ import unicode_literals
from builtins import str, bytes, dict, int
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from pattern.web import Wiktionary, DOM
from pattern.db import csv, pd
w = Wiktionary(language='en')
f = csv()
for gender in ('male', 'female'):
    for ch in 'abcdefghijklmnopqrstuvwxyz':
        p = w.search('Appendix:%s_given_names/%s' % (gender.capitalize(), ch.capitalize()), cached=True)
        for name in p.links:
            if not name.startswith('Appendix:'):
                f.append((name, gender[0]))
        f.save(pd('given-names.csv'))
        print(ch, gender)
from pattern.vector import SVM, chngrams, count, kfoldcv

class GenderByName(SVM):

    def train(self, name, gender=None):
        if False:
            print('Hello World!')
        SVM.train(self, self.vector(name), gender)

    def classify(self, name):
        if False:
            print('Hello World!')
        return SVM.classify(self, self.vector(name))

    def vector(self, name):
        if False:
            print('Hello World!')
        ' Returns a dictionary with character bigrams and suffix.\n            For example, "Felix" => {"Fe":1, "el":1, "li":1, "ix":1, "ix$":1, 5:1}\n        '
        v = chngrams(name, n=2)
        v = count(v)
        v[name[-2:] + '$'] = 1
        v[len(name)] = 1
        return v
data = csv(pd('given-names.csv'))
print(kfoldcv(GenderByName, data, folds=3))
g = GenderByName(train=data)
g.save(pd('gender-by-name.svm'), final=True)
g = GenderByName.load(pd('gender-by-name.svm'))
for name in ('Felix', 'Felicia', 'Rover', 'Kitty', 'Legolas', 'Arwen', 'Jabba', 'Leia', 'Flash', 'Barbarella'):
    print(name, g.classify(name))