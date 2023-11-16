__license__ = 'GPL v3'
__copyright__ = '2013, Kovid Goyal <kovid at kovidgoyal.net>'
from collections import OrderedDict
from polyglot.builtins import iteritems

class Note:

    def __init__(self, namespace, parent, rels):
        if False:
            while True:
                i = 10
        self.type = namespace.get(parent, 'w:type', 'normal')
        self.parent = parent
        self.rels = rels
        self.namespace = namespace

    def __iter__(self):
        if False:
            while True:
                i = 10
        yield from self.namespace.descendants(self.parent, 'w:p', 'w:tbl')

class Footnotes:

    def __init__(self, namespace):
        if False:
            i = 10
            return i + 15
        self.namespace = namespace
        self.footnotes = {}
        self.endnotes = {}
        self.counter = 0
        self.notes = OrderedDict()

    def __call__(self, footnotes, footnotes_rels, endnotes, endnotes_rels):
        if False:
            i = 10
            return i + 15
        (XPath, get) = (self.namespace.XPath, self.namespace.get)
        if footnotes is not None:
            for footnote in XPath('./w:footnote[@w:id]')(footnotes):
                fid = get(footnote, 'w:id')
                if fid:
                    self.footnotes[fid] = Note(self.namespace, footnote, footnotes_rels)
        if endnotes is not None:
            for endnote in XPath('./w:endnote[@w:id]')(endnotes):
                fid = get(endnote, 'w:id')
                if fid:
                    self.endnotes[fid] = Note(self.namespace, endnote, endnotes_rels)

    def get_ref(self, ref):
        if False:
            for i in range(10):
                print('nop')
        fid = self.namespace.get(ref, 'w:id')
        notes = self.footnotes if ref.tag.endswith('}footnoteReference') else self.endnotes
        note = notes.get(fid, None)
        if note is not None and note.type == 'normal':
            self.counter += 1
            anchor = 'note_%d' % self.counter
            self.notes[anchor] = (str(self.counter), note)
            return (anchor, str(self.counter))
        return (None, None)

    def __iter__(self):
        if False:
            return 10
        for (anchor, (counter, note)) in iteritems(self.notes):
            yield (anchor, counter, note)

    @property
    def has_notes(self):
        if False:
            for i in range(10):
                print('nop')
        return bool(self.notes)