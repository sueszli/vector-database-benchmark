"""
Some categories of objects in a PDF file can be referred to by name rather than by object reference. The
correspondence between names and objects is established by the document’s name dictionary (PDF 1.2),
located by means of the Names entry in the document’s catalog (see 7.7.2, "Document Catalog"). Each entry in
this dictionary designates the root of a name tree (see 7.9.6, "Name Trees") defining names for a particular
category of objects.

A name tree serves a similar purpose to a dictionary—associating keys and values—but by different means.
A name tree differs from a dictionary in the following important ways:
- Unlike the keys in a dictionary, which are name objects, those in a name tree are strings.
- The keys are ordered.
- The values associated with the keys may be objects of any type. Stream objects shall be specified by
indirect object references (7.3.8, "Stream Objects"). The dictionary, array, and string objects should be
specified by indirect object references, and other PDF objects (nulls, numbers, booleans, and names)
should be specified as direct objects.
- The data structure can represent an arbitrarily large collection of key-value pairs, which can be looked up
efficiently without requiring the entire data structure to be read from the PDF file. (In contrast, a dictionary
can be subject to an implementation limit on the number of entries it can contain.)
"""
import typing
from borb.io.read.types import Dictionary
from borb.io.read.types import List
from borb.io.read.types import Name
from borb.io.read.types import String

class NameTree:
    """
    A name tree is similar to a dictionary that associates keys and values but the keys in a name tree are strings and are ordered
    """

    def __init__(self, document: Dictionary, name: Name):
        if False:
            for i in range(10):
                print('nop')
        self._document: Dictionary = document
        self._name: Name = name

    def __len__(self):
        if False:
            return 10
        return len(self._get_root_or_empty())

    def _get_root_or_empty(self):
        if False:
            while True:
                i = 10
        assert 'XRef' in self._document, 'No XREF found in this PDF'
        assert 'Trailer' in self._document['XRef'], 'No /Trailer dictionary found in the XREF'
        assert 'Root' in self._document['XRef']['Trailer'], 'No /Root dictionary found in the /Trailer'
        root = self._document['XRef']['Trailer']['Root']
        return root.get(Name('Names'), Dictionary())

    def _put_existing(self, parent: Dictionary, key: str, value: typing.Any):
        if False:
            for i in range(10):
                print('nop')
        pass

    def _put_new(self, parent: Dictionary, key: str, value: typing.Any):
        if False:
            i = 10
            return i + 15
        kid = Dictionary()
        kid[Name('F')] = String(key)
        kid[Name('Limits')] = List()
        for _ in range(0, 2):
            kid['Limits'].append(String(key))
        kid[Name('Names')] = List()
        kid[Name('Names')].append(String(key))
        if self._name == 'EmbeddedFiles':
            kid[Name('Names')].append(value)
            kid[Name('Type')] = Name('EF')
        if self._name == 'JavaScript':
            kid[Name('Names')].append(value)
        parent['Kids'].append(kid)

    def items(self) -> typing.Iterable[typing.Tuple[String, typing.Any]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        This function returns all key/value pairs in this NameTree\n        :return:    all key/value pairs in this NameTree\n        '
        assert 'XRef' in self._document, 'No XREF found in this PDF'
        assert 'Trailer' in self._document['XRef'], 'No /Trailer dictionary found in the XREF'
        assert 'Root' in self._document['XRef']['Trailer'], 'No /Root dictionary found in the /Trailer'
        root = self._document['XRef']['Trailer']['Root']
        if 'Names' not in root:
            root[Name('Names')] = Dictionary()
        names = root['Names']
        nodes_to_visit = [names[self._name]]
        keys = []
        values = []
        while len(nodes_to_visit) > 0:
            n = nodes_to_visit[0]
            nodes_to_visit.pop(0)
            if 'Kids' in n:
                for k in n['Kids']:
                    nodes_to_visit.append(k)
            if 'Limits' in n:
                lower_limit = str(n['Limits'][0])
                upper_limit = str(n['Limits'][1])
                if upper_limit == lower_limit:
                    keys.append(n['Limits'][1])
                    values.append(n['Names'][1])
        return zip(keys, values)

    def keys(self) -> typing.List[String]:
        if False:
            while True:
                i = 10
        '\n        This function returns the keys in this NameTree\n        :return:    the keys in this NameTree\n        '
        return [k for (k, v) in self.items()]

    def put(self, key: str, value: typing.Any) -> 'NameTree':
        if False:
            return 10
        '\n        This function adds a key/value pair in this NameTree\n        :param key:     the key\n        :param value:   the value\n        :return:        self\n        '
        assert 'XRef' in self._document, 'No XREF found in this PDF'
        assert 'Trailer' in self._document['XRef'], 'No /Trailer dictionary found in the XREF'
        assert 'Root' in self._document['XRef']['Trailer'], 'No /Root dictionary found in the /Trailer'
        root = self._document['XRef']['Trailer']['Root']
        if 'Names' not in root:
            root[Name('Names')] = Dictionary()
        names = root['Names']
        if self._name not in names:
            names[self._name] = Dictionary()
            names[self._name][Name('Kids')] = List()
        parent = names[self._name]
        while 'Kids' in parent:
            for k in parent['Kids']:
                lower_limit = str(k['Limits'][0])
                upper_limit = str(k['Limits'][1])
                if lower_limit == upper_limit:
                    continue
                if lower_limit < key < upper_limit:
                    parent = k
                    break
            break
        if len([x for x in parent['Kids'] if x['Limits'][0] == x['Limits'][1] == key]) == 0:
            self._put_new(parent, key, value)
        else:
            self._put_existing(parent, key, value)
        return self

    def values(self) -> typing.List[typing.Any]:
        if False:
            return 10
        '\n        This function returns the values in this NameTree\n        :return:    the values in this NameTree\n        '
        return [v for (k, v) in self.items()]