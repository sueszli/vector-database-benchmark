from itertools import chain
from nltk.internals import Counter

class FStructure(dict):

    def safeappend(self, key, item):
        if False:
            print('Hello World!')
        "\n        Append 'item' to the list at 'key'.  If no list exists for 'key', then\n        construct one.\n        "
        if key not in self:
            self[key] = []
        self[key].append(item)

    def __setitem__(self, key, value):
        if False:
            return 10
        dict.__setitem__(self, key.lower(), value)

    def __getitem__(self, key):
        if False:
            i = 10
            return i + 15
        return dict.__getitem__(self, key.lower())

    def __contains__(self, key):
        if False:
            for i in range(10):
                print('nop')
        return dict.__contains__(self, key.lower())

    def to_glueformula_list(self, glue_dict):
        if False:
            i = 10
            return i + 15
        depgraph = self.to_depgraph()
        return glue_dict.to_glueformula_list(depgraph)

    def to_depgraph(self, rel=None):
        if False:
            print('Hello World!')
        from nltk.parse.dependencygraph import DependencyGraph
        depgraph = DependencyGraph()
        nodes = depgraph.nodes
        self._to_depgraph(nodes, 0, 'ROOT')
        for (address, node) in nodes.items():
            for n2 in (n for n in nodes.values() if n['rel'] != 'TOP'):
                if n2['head'] == address:
                    relation = n2['rel']
                    node['deps'].setdefault(relation, [])
                    node['deps'][relation].append(n2['address'])
        depgraph.root = nodes[1]
        return depgraph

    def _to_depgraph(self, nodes, head, rel):
        if False:
            while True:
                i = 10
        index = len(nodes)
        nodes[index].update({'address': index, 'word': self.pred[0], 'tag': self.pred[1], 'head': head, 'rel': rel})
        for feature in sorted(self):
            for item in sorted(self[feature]):
                if isinstance(item, FStructure):
                    item._to_depgraph(nodes, index, feature)
                elif isinstance(item, tuple):
                    new_index = len(nodes)
                    nodes[new_index].update({'address': new_index, 'word': item[0], 'tag': item[1], 'head': index, 'rel': feature})
                elif isinstance(item, list):
                    for n in item:
                        n._to_depgraph(nodes, index, feature)
                else:
                    raise Exception('feature %s is not an FStruct, a list, or a tuple' % feature)

    @staticmethod
    def read_depgraph(depgraph):
        if False:
            print('Hello World!')
        return FStructure._read_depgraph(depgraph.root, depgraph)

    @staticmethod
    def _read_depgraph(node, depgraph, label_counter=None, parent=None):
        if False:
            return 10
        if not label_counter:
            label_counter = Counter()
        if node['rel'].lower() in ['spec', 'punct']:
            return (node['word'], node['tag'])
        else:
            fstruct = FStructure()
            fstruct.pred = None
            fstruct.label = FStructure._make_label(label_counter.get())
            fstruct.parent = parent
            (word, tag) = (node['word'], node['tag'])
            if tag[:2] == 'VB':
                if tag[2:3] == 'D':
                    fstruct.safeappend('tense', ('PAST', 'tense'))
                fstruct.pred = (word, tag[:2])
            if not fstruct.pred:
                fstruct.pred = (word, tag)
            children = [depgraph.nodes[idx] for idx in chain.from_iterable(node['deps'].values())]
            for child in children:
                fstruct.safeappend(child['rel'], FStructure._read_depgraph(child, depgraph, label_counter, fstruct))
            return fstruct

    @staticmethod
    def _make_label(value):
        if False:
            return 10
        '\n        Pick an alphabetic character as identifier for an entity in the model.\n\n        :param value: where to index into the list of characters\n        :type value: int\n        '
        letter = ['f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'a', 'b', 'c', 'd', 'e'][value - 1]
        num = int(value) // 26
        if num > 0:
            return letter + str(num)
        else:
            return letter

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__str__().replace('\n', '')

    def __str__(self):
        if False:
            while True:
                i = 10
        return self.pretty_format()

    def pretty_format(self, indent=3):
        if False:
            for i in range(10):
                print('nop')
        try:
            accum = '%s:[' % self.label
        except NameError:
            accum = '['
        try:
            accum += "pred '%s'" % self.pred[0]
        except NameError:
            pass
        for feature in sorted(self):
            for item in self[feature]:
                if isinstance(item, FStructure):
                    next_indent = indent + len(feature) + 3 + len(self.label)
                    accum += '\n{}{} {}'.format(' ' * indent, feature, item.pretty_format(next_indent))
                elif isinstance(item, tuple):
                    accum += "\n{}{} '{}'".format(' ' * indent, feature, item[0])
                elif isinstance(item, list):
                    accum += '\n{}{} {{{}}}'.format(' ' * indent, feature, ('\n%s' % (' ' * (indent + len(feature) + 2))).join(item))
                else:
                    raise Exception('feature %s is not an FStruct, a list, or a tuple' % feature)
        return accum + ']'

def demo_read_depgraph():
    if False:
        return 10
    from nltk.parse.dependencygraph import DependencyGraph
    dg1 = DependencyGraph('Esso       NNP     2       SUB\nsaid       VBD     0       ROOT\nthe        DT      5       NMOD\nWhiting    NNP     5       NMOD\nfield      NN      6       SUB\nstarted    VBD     2       VMOD\nproduction NN      6       OBJ\nTuesday    NNP     6       VMOD\n')
    dg2 = DependencyGraph('John    NNP     2       SUB\nsees    VBP     0       ROOT\nMary    NNP     2       OBJ\n')
    dg3 = DependencyGraph('a       DT      2       SPEC\nman     NN      3       SUBJ\nwalks   VB      0       ROOT\n')
    dg4 = DependencyGraph('every   DT      2       SPEC\ngirl    NN      3       SUBJ\nchases  VB      0       ROOT\na       DT      5       SPEC\ndog     NN      3       OBJ\n')
    depgraphs = [dg1, dg2, dg3, dg4]
    for dg in depgraphs:
        print(FStructure.read_depgraph(dg))
if __name__ == '__main__':
    demo_read_depgraph()