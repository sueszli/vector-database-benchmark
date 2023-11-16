import random
from contextlib import contextmanager
import pytest
from spacy.lang.en import English
from spacy.pipeline._parser_internals.nonproj import contains_cycle
from spacy.tokens import Doc, DocBin, Span
from spacy.training import Corpus, Example
from spacy.training.augment import create_lower_casing_augmenter, create_orth_variants_augmenter, make_whitespace_variant
from ..util import make_tempdir

@contextmanager
def make_docbin(docs, name='roundtrip.spacy'):
    if False:
        i = 10
        return i + 15
    with make_tempdir() as tmpdir:
        output_file = tmpdir / name
        DocBin(docs=docs).to_disk(output_file)
        yield output_file

@pytest.fixture
def nlp():
    if False:
        i = 10
        return i + 15
    return English()

@pytest.fixture
def doc(nlp):
    if False:
        print('Hello World!')
    words = ['Sarah', "'s", 'sister', 'flew', 'to', 'Silicon', 'Valley', 'via', 'London', '.']
    tags = ['NNP', 'POS', 'NN', 'VBD', 'IN', 'NNP', 'NNP', 'IN', 'NNP', '.']
    pos = ['PROPN', 'PART', 'NOUN', 'VERB', 'ADP', 'PROPN', 'PROPN', 'ADP', 'PROPN', 'PUNCT']
    ents = ['B-PERSON', 'I-PERSON', 'O', '', 'O', 'B-LOC', 'I-LOC', 'O', 'B-GPE', 'O']
    cats = {'TRAVEL': 1.0, 'BAKING': 0.0}
    doc = Doc(nlp.vocab, words=words, tags=tags, pos=pos, ents=ents)
    doc.cats = cats
    return doc

@pytest.mark.filterwarnings('ignore::UserWarning')
def test_make_orth_variants(nlp):
    if False:
        print('Hello World!')
    single = [{'tags': ['NFP'], 'variants': ['…', '...']}, {'tags': [':'], 'variants': ['-', '—', '–', '--', '---', '——']}]
    words = ['\n\n', 'A', '\t', 'B', 'a', 'b', '…', '...', '-', '—', '–', '--', '---', '——']
    tags = ['_SP', 'NN', '\t', 'NN', 'NN', 'NN', 'NFP', 'NFP', ':', ':', ':', ':', ':', ':']
    spaces = [True] * len(words)
    spaces[0] = False
    spaces[2] = False
    doc = Doc(nlp.vocab, words=words, spaces=spaces, tags=tags)
    augmenter = create_orth_variants_augmenter(level=0.2, lower=0.5, orth_variants={'single': single})
    with make_docbin([doc] * 10) as output_file:
        reader = Corpus(output_file, augmenter=augmenter)
        list(reader(nlp))
    augmenter = create_orth_variants_augmenter(level=1.0, lower=1.0, orth_variants={'single': single})
    with make_docbin([doc] * 10) as output_file:
        reader = Corpus(output_file, augmenter=augmenter)
        for example in reader(nlp):
            for token in example.reference:
                assert token.text == token.text.lower()
    doc = Doc(nlp.vocab, words=words, spaces=[True] * len(words))
    augmenter = create_orth_variants_augmenter(level=1.0, lower=1.0, orth_variants={'single': single})
    with make_docbin([doc] * 10) as output_file:
        reader = Corpus(output_file, augmenter=augmenter)
        for example in reader(nlp):
            for (ex_token, doc_token) in zip(example.reference, doc):
                assert ex_token.text == doc_token.text.lower()
    doc = Doc(nlp.vocab, words=words, spaces=[True] * len(words))
    augmenter = create_orth_variants_augmenter(level=1.0, lower=0.0, orth_variants={'single': single})
    with make_docbin([doc] * 10) as output_file:
        reader = Corpus(output_file, augmenter=augmenter)
        for example in reader(nlp):
            for (ex_token, doc_token) in zip(example.reference, doc):
                assert ex_token.text == doc_token.text

def test_lowercase_augmenter(nlp, doc):
    if False:
        i = 10
        return i + 15
    augmenter = create_lower_casing_augmenter(level=1.0)
    with make_docbin([doc]) as output_file:
        reader = Corpus(output_file, augmenter=augmenter)
        corpus = list(reader(nlp))
    eg = corpus[0]
    assert eg.reference.text == doc.text.lower()
    assert eg.predicted.text == doc.text.lower()
    ents = [(e.start, e.end, e.label) for e in doc.ents]
    assert [(e.start, e.end, e.label) for e in eg.reference.ents] == ents
    for (ref_ent, orig_ent) in zip(eg.reference.ents, doc.ents):
        assert ref_ent.text == orig_ent.text.lower()
    assert [t.ent_iob for t in doc] == [t.ent_iob for t in eg.reference]
    assert [t.pos_ for t in eg.reference] == [t.pos_ for t in doc]
    words = ['A', 'B', 'CCC.']
    doc = Doc(nlp.vocab, words=words)
    with make_docbin([doc]) as output_file:
        reader = Corpus(output_file, augmenter=augmenter)
        corpus = list(reader(nlp))
    eg = corpus[0]
    assert eg.reference.text == doc.text.lower()
    assert eg.predicted.text == doc.text.lower()
    assert [t.text for t in eg.reference] == [t.lower() for t in words]
    assert [t.text for t in eg.predicted] == [t.text for t in nlp.make_doc(doc.text.lower())]

@pytest.mark.filterwarnings('ignore::UserWarning')
def test_custom_data_augmentation(nlp, doc):
    if False:
        while True:
            i = 10

    def create_spongebob_augmenter(randomize: bool=False):
        if False:
            print('Hello World!')

        def augment(nlp, example):
            if False:
                i = 10
                return i + 15
            text = example.text
            if randomize:
                ch = [c.lower() if random.random() < 0.5 else c.upper() for c in text]
            else:
                ch = [c.lower() if i % 2 else c.upper() for (i, c) in enumerate(text)]
            example_dict = example.to_dict()
            doc = nlp.make_doc(''.join(ch))
            example_dict['token_annotation']['ORTH'] = [t.text for t in doc]
            yield example
            yield example.from_dict(doc, example_dict)
        return augment
    with make_docbin([doc]) as output_file:
        reader = Corpus(output_file, augmenter=create_spongebob_augmenter())
        corpus = list(reader(nlp))
    orig_text = "Sarah 's sister flew to Silicon Valley via London . "
    augmented = "SaRaH 's sIsTeR FlEw tO SiLiCoN VaLlEy vIa lOnDoN . "
    assert corpus[0].text == orig_text
    assert corpus[0].reference.text == orig_text
    assert corpus[0].predicted.text == orig_text
    assert corpus[1].text == augmented
    assert corpus[1].reference.text == augmented
    assert corpus[1].predicted.text == augmented
    ents = [(e.start, e.end, e.label) for e in doc.ents]
    assert [(e.start, e.end, e.label) for e in corpus[0].reference.ents] == ents
    assert [(e.start, e.end, e.label) for e in corpus[1].reference.ents] == ents

def test_make_whitespace_variant(nlp):
    if False:
        i = 10
        return i + 15
    text = 'They flew to New York City.\nThen they drove to Washington, D.C.'
    words = ['They', 'flew', 'to', 'New', 'York', 'City', '.', '\n', 'Then', 'they', 'drove', 'to', 'Washington', ',', 'D.C.']
    spaces = [True, True, True, True, True, False, False, False, True, True, True, True, False, True, False]
    tags = ['PRP', 'VBD', 'IN', 'NNP', 'NNP', 'NNP', '.', '_SP', 'RB', 'PRP', 'VBD', 'IN', 'NNP', ',', 'NNP']
    lemmas = ['they', 'fly', 'to', 'New', 'York', 'City', '.', '\n', 'then', 'they', 'drive', 'to', 'Washington', ',', 'D.C.']
    heads = [1, 1, 1, 4, 5, 2, 1, 10, 10, 10, 10, 10, 11, 12, 12]
    deps = ['nsubj', 'ROOT', 'prep', 'compound', 'compound', 'pobj', 'punct', 'dep', 'advmod', 'nsubj', 'ROOT', 'prep', 'pobj', 'punct', 'appos']
    ents = ['O', '', 'O', 'B-GPE', 'I-GPE', 'I-GPE', 'O', 'O', 'O', 'O', 'O', 'O', 'B-GPE', 'O', 'B-GPE']
    doc = Doc(nlp.vocab, words=words, spaces=spaces, tags=tags, lemmas=lemmas, heads=heads, deps=deps, ents=ents)
    assert doc.text == text
    example = Example(nlp.make_doc(text), doc)
    mod_ex = make_whitespace_variant(nlp, example, ' ', 3)
    assert mod_ex.reference.ents[0].text == 'New York City'
    mod_ex = make_whitespace_variant(nlp, example, ' ', 4)
    assert mod_ex.reference.ents[0].text == 'New  York City'
    mod_ex = make_whitespace_variant(nlp, example, ' ', 5)
    assert mod_ex.reference.ents[0].text == 'New York  City'
    mod_ex = make_whitespace_variant(nlp, example, ' ', 6)
    assert mod_ex.reference.ents[0].text == 'New York City'
    for i in range(len(doc) + 1):
        mod_ex = make_whitespace_variant(nlp, example, ' ', i)
        assert mod_ex.reference[i].is_space
        assert [t.tag_ for t in mod_ex.reference] == tags[:i] + ['_SP'] + tags[i:]
        assert [t.lemma_ for t in mod_ex.reference] == lemmas[:i] + [' '] + lemmas[i:]
        assert [t.dep_ for t in mod_ex.reference] == deps[:i] + ['dep'] + deps[i:]
        assert not mod_ex.reference.has_annotation('POS')
        assert not mod_ex.reference.has_annotation('MORPH')
        assert not contains_cycle([t.head.i for t in mod_ex.reference])
        assert len(list(doc.sents)) == 2
        if i == 0:
            assert mod_ex.reference[i].head.i == 1
        else:
            assert mod_ex.reference[i].head.i == i - 1
        for j in (3, 8, 10):
            mod_ex2 = make_whitespace_variant(nlp, mod_ex, '\t\t\n', j)
            assert not contains_cycle([t.head.i for t in mod_ex2.reference])
            assert len(list(doc.sents)) == 2
            assert mod_ex2.reference[j].head.i == j - 1
        assert len(doc.ents) == len(mod_ex.reference.ents)
        assert any((t.ent_iob == 0 for t in mod_ex.reference))
        for ent in mod_ex.reference.ents:
            assert not ent[0].is_space
            assert not ent[-1].is_space
    example.reference[0].dep_ = ''
    mod_ex = make_whitespace_variant(nlp, example, ' ', 5)
    assert mod_ex.text == example.reference.text
    example.reference[0].dep_ = 'nsubj'
    example.reference.spans['spans'] = [example.reference[0:5]]
    mod_ex = make_whitespace_variant(nlp, example, ' ', 5)
    assert mod_ex.text == example.reference.text
    del example.reference.spans['spans']
    example.reference.ents = [Span(doc, 0, 2, label='ENT', kb_id='Q123')]
    mod_ex = make_whitespace_variant(nlp, example, ' ', 5)
    assert mod_ex.text == example.reference.text