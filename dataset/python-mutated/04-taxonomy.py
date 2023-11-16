from __future__ import print_function
from __future__ import unicode_literals
from builtins import str, bytes, dict, int
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from pattern.search import search, taxonomy, Classifier
from pattern.en import parsetree
for flower in ('rose', 'lily', 'daisy', 'daffodil', 'begonia'):
    taxonomy.append(flower, type='flower')
print(taxonomy.children('flower'))
print(taxonomy.parents('rose'))
print(taxonomy.classify('rose'))
print('')
t = parsetree('A field of white daffodils.', lemmata=True)
m = search('FLOWER', t)
print(t)
print(m)
print('')
taxonomy.append('chicken', type='food')
taxonomy.append('chicken', type='bird')
taxonomy.append('penguin', type='bird')
taxonomy.append('bird', type='animal')
print(taxonomy.parents('chicken'))
print(taxonomy.children('animal', recursive=True))
print(search('FOOD', "I'm eating chicken."))
print('')
taxonomy.append('windows vista', type='operating system')
taxonomy.append('ubuntu', type='operating system')
t = parsetree('Which do you like more, Windows Vista, or Ubuntu?')
m = search('OPERATING_SYSTEM', t)
print(t)
print(m)
print(m[0].constituents())
print('')

def find_parents(word):
    if False:
        print('Hello World!')
    if word.startswith(('mac os', 'windows', 'ubuntu')):
        return ['operating system']
c = Classifier(parents=find_parents)
taxonomy.classifiers.append(c)
t = parsetree('I like Mac OS X 10.5 better than Windows XP or Ubuntu.')
m = search('OPERATING_SYSTEM', t)
print(t)
print(m)
print(m[0].constituents())
print(m[1].constituents())
print('')