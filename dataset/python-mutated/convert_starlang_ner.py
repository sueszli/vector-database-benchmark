"""
Convert the starlang trees to a NER dataset

Has to hide quite a few trees with missing NER labels
"""
import re
from stanza.models.constituency import tree_reader
import stanza.utils.datasets.constituency.convert_starlang as convert_starlang
TURKISH_WORD_RE = re.compile('[{]turkish=([^}]+)[}]')
TURKISH_LABEL_RE = re.compile('[{]namedEntity=([^}]+)[}]')

def read_tree(text):
    if False:
        print('Hello World!')
    '\n    Reads in a tree, then extracts the word and the NER\n\n    One problem is that it is unknown if there are cases of two separate items occurring consecutively\n\n    Note that this is quite similar to the convert_starlang script for constituency.  \n    '
    trees = tree_reader.read_trees(text)
    if len(trees) > 1:
        raise ValueError('Tree file had two trees!')
    tree = trees[0]
    words = []
    for label in tree.leaf_labels():
        match = TURKISH_WORD_RE.search(label)
        if match is None:
            raise ValueError('Could not find word in |{}|'.format(label))
        word = match.group(1)
        word = word.replace('-LCB-', '{').replace('-RCB-', '}')
        match = TURKISH_LABEL_RE.search(label)
        if match is None:
            raise ValueError('Could not find ner in |{}|'.format(label))
        tag = match.group(1)
        if tag == 'NONE' or tag == 'null':
            tag = 'O'
        words.append((word, tag))
    return words

def read_starlang(paths):
    if False:
        while True:
            i = 10
    return convert_starlang.read_starlang(paths, conversion=read_tree, log=False)

def main():
    if False:
        return 10
    (train, dev, test) = convert_starlang.main(conversion=read_tree, log=False)
if __name__ == '__main__':
    main()