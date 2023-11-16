__author__ = 'Paul McCann <polm@dampfkraft.com>'
from visidata import vd, VisiData, TableSheet, ItemColumn

@VisiData.api
def open_conll(vd, p):
    if False:
        i = 10
        return i + 15
    return ConllSheet(p.name, source=p)

@VisiData.api
def open_conllu(vd, p):
    if False:
        for i in range(10):
            print('nop')
    return ConllSheet(p.name, source=p)

class ConllSheet(TableSheet):
    rowtype = 'tokens'
    columns = [ItemColumn('sent_id', 0, type=str), ItemColumn('token_id', 1, type=int), ItemColumn('form', 2, type=str), ItemColumn('lemma', 3, type=str), ItemColumn('upos', 4, type=str), ItemColumn('xpos', 5, type=str), ItemColumn('feats', 6, type=dict), ItemColumn('head', 7, type=int), ItemColumn('deprel', 8, type=str), ItemColumn('deps', 9), ItemColumn('misc', 10, type=dict)]

    def iterload(self):
        if False:
            print('Hello World!')
        pyconll = vd.importExternal('pyconll')
        self.setKeys([self.columns[0], self.columns[1]])
        with self.source.open(encoding='utf-8') as fp:
            for sent in pyconll.load.iter_sentences(fp):
                sent_id = sent.id
                for token in sent:
                    yield [sent_id, token.id, token._form, token.lemma, token.upos, token.xpos, token.feats, token.head, token.deprel, token.deps, token.misc]