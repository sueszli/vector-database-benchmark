import pickle
import hypothesis.strategies as st
import pytest
from hypothesis import given
from spacy import util
from spacy.lang.en import English
from spacy.language import Language
from spacy.pipeline._edit_tree_internals.edit_trees import EditTrees
from spacy.strings import StringStore
from spacy.training import Example
from spacy.util import make_tempdir
TRAIN_DATA = [('She likes green eggs', {'lemmas': ['she', 'like', 'green', 'egg']}), ('Eat blue ham', {'lemmas': ['eat', 'blue', 'ham']})]
PARTIAL_DATA = [('She likes green eggs', {'lemmas': ['', 'like', 'green', '']}), ('He hates green eggs', {'words': ['He', 'hat', 'es', 'green', 'eggs'], 'lemmas': ['', 'hat', 'e', 'green', '']})]

def test_initialize_examples():
    if False:
        return 10
    nlp = Language()
    lemmatizer = nlp.add_pipe('trainable_lemmatizer')
    train_examples = []
    for t in TRAIN_DATA:
        train_examples.append(Example.from_dict(nlp.make_doc(t[0]), t[1]))
    nlp.initialize(get_examples=lambda : train_examples)
    with pytest.raises(TypeError):
        nlp.initialize(get_examples=lambda : None)
    with pytest.raises(TypeError):
        nlp.initialize(get_examples=lambda : train_examples[0])
    with pytest.raises(TypeError):
        nlp.initialize(get_examples=lambda : [])
    with pytest.raises(TypeError):
        nlp.initialize(get_examples=train_examples)

def test_initialize_from_labels():
    if False:
        for i in range(10):
            print('nop')
    nlp = Language()
    lemmatizer = nlp.add_pipe('trainable_lemmatizer')
    lemmatizer.min_tree_freq = 1
    train_examples = []
    for t in TRAIN_DATA:
        train_examples.append(Example.from_dict(nlp.make_doc(t[0]), t[1]))
    nlp.initialize(get_examples=lambda : train_examples)
    nlp2 = Language()
    lemmatizer2 = nlp2.add_pipe('trainable_lemmatizer')
    lemmatizer2.initialize(get_examples=lambda : train_examples[:1], labels=lemmatizer.label_data)
    assert lemmatizer2.tree2label == {1: 0, 3: 1, 4: 2, 6: 3}
    assert lemmatizer2.label_data == {'trees': [{'orig': 'S', 'subst': 's'}, {'prefix_len': 1, 'suffix_len': 0, 'prefix_tree': 0, 'suffix_tree': 4294967295}, {'orig': 's', 'subst': ''}, {'prefix_len': 0, 'suffix_len': 1, 'prefix_tree': 4294967295, 'suffix_tree': 2}, {'prefix_len': 0, 'suffix_len': 0, 'prefix_tree': 4294967295, 'suffix_tree': 4294967295}, {'orig': 'E', 'subst': 'e'}, {'prefix_len': 1, 'suffix_len': 0, 'prefix_tree': 5, 'suffix_tree': 4294967295}], 'labels': (1, 3, 4, 6)}

@pytest.mark.parametrize('top_k', (1, 5, 30))
def test_no_data(top_k):
    if False:
        print('Hello World!')
    TEXTCAT_DATA = [("I'm so happy.", {'cats': {'POSITIVE': 1.0, 'NEGATIVE': 0.0}}), ("I'm so angry", {'cats': {'POSITIVE': 0.0, 'NEGATIVE': 1.0}})]
    nlp = English()
    nlp.add_pipe('trainable_lemmatizer', config={'top_k': top_k})
    nlp.add_pipe('textcat')
    train_examples = []
    for t in TEXTCAT_DATA:
        train_examples.append(Example.from_dict(nlp.make_doc(t[0]), t[1]))
    with pytest.raises(ValueError):
        nlp.initialize(get_examples=lambda : train_examples)

@pytest.mark.parametrize('top_k', (1, 5, 30))
def test_incomplete_data(top_k):
    if False:
        i = 10
        return i + 15
    nlp = English()
    lemmatizer = nlp.add_pipe('trainable_lemmatizer', config={'top_k': top_k})
    lemmatizer.min_tree_freq = 1
    train_examples = []
    for t in PARTIAL_DATA:
        train_examples.append(Example.from_dict(nlp.make_doc(t[0]), t[1]))
    optimizer = nlp.initialize(get_examples=lambda : train_examples)
    for i in range(50):
        losses = {}
        nlp.update(train_examples, sgd=optimizer, losses=losses)
    assert losses['trainable_lemmatizer'] < 1e-05
    test_text = 'She likes blue eggs'
    doc = nlp(test_text)
    assert doc[1].lemma_ == 'like'
    assert doc[2].lemma_ == 'blue'
    (scores, _) = lemmatizer.model([eg.predicted for eg in train_examples], is_train=True)
    (_, dX) = lemmatizer.get_loss(train_examples, scores)
    xp = lemmatizer.model.ops.xp
    assert xp.count_nonzero(dX[0][0]) == 0
    assert xp.count_nonzero(dX[0][3]) == 0
    assert xp.count_nonzero(dX[1][0]) == 0
    assert xp.count_nonzero(dX[1][3]) == 0
    assert xp.count_nonzero(dX[1][1]) == 0

@pytest.mark.parametrize('top_k', (1, 5, 30))
def test_overfitting_IO(top_k):
    if False:
        for i in range(10):
            print('nop')
    nlp = English()
    lemmatizer = nlp.add_pipe('trainable_lemmatizer', config={'top_k': top_k})
    lemmatizer.min_tree_freq = 1
    train_examples = []
    for t in TRAIN_DATA:
        train_examples.append(Example.from_dict(nlp.make_doc(t[0]), t[1]))
    optimizer = nlp.initialize(get_examples=lambda : train_examples)
    for i in range(50):
        losses = {}
        nlp.update(train_examples, sgd=optimizer, losses=losses)
    assert losses['trainable_lemmatizer'] < 1e-05
    test_text = 'She likes blue eggs'
    doc = nlp(test_text)
    assert doc[0].lemma_ == 'she'
    assert doc[1].lemma_ == 'like'
    assert doc[2].lemma_ == 'blue'
    assert doc[3].lemma_ == 'egg'
    with util.make_tempdir() as tmp_dir:
        nlp.to_disk(tmp_dir)
        nlp2 = util.load_model_from_path(tmp_dir)
        doc2 = nlp2(test_text)
        assert doc2[0].lemma_ == 'she'
        assert doc2[1].lemma_ == 'like'
        assert doc2[2].lemma_ == 'blue'
        assert doc2[3].lemma_ == 'egg'
    nlp_bytes = nlp.to_bytes()
    nlp3 = English()
    nlp3.add_pipe('trainable_lemmatizer', config={'top_k': top_k})
    nlp3.from_bytes(nlp_bytes)
    doc3 = nlp3(test_text)
    assert doc3[0].lemma_ == 'she'
    assert doc3[1].lemma_ == 'like'
    assert doc3[2].lemma_ == 'blue'
    assert doc3[3].lemma_ == 'egg'
    nlp_bytes = pickle.dumps(nlp)
    nlp4 = pickle.loads(nlp_bytes)
    doc4 = nlp4(test_text)
    assert doc4[0].lemma_ == 'she'
    assert doc4[1].lemma_ == 'like'
    assert doc4[2].lemma_ == 'blue'
    assert doc4[3].lemma_ == 'egg'

def test_lemmatizer_requires_labels():
    if False:
        i = 10
        return i + 15
    nlp = English()
    nlp.add_pipe('trainable_lemmatizer')
    with pytest.raises(ValueError):
        nlp.initialize()

def test_lemmatizer_label_data():
    if False:
        while True:
            i = 10
    nlp = English()
    lemmatizer = nlp.add_pipe('trainable_lemmatizer')
    lemmatizer.min_tree_freq = 1
    train_examples = []
    for t in TRAIN_DATA:
        train_examples.append(Example.from_dict(nlp.make_doc(t[0]), t[1]))
    nlp.initialize(get_examples=lambda : train_examples)
    nlp2 = English()
    lemmatizer2 = nlp2.add_pipe('trainable_lemmatizer')
    lemmatizer2.initialize(get_examples=lambda : train_examples, labels=lemmatizer.label_data)
    assert lemmatizer.labels == lemmatizer2.labels
    assert lemmatizer.trees.to_bytes() == lemmatizer2.trees.to_bytes()

def test_dutch():
    if False:
        for i in range(10):
            print('nop')
    strings = StringStore()
    trees = EditTrees(strings)
    tree = trees.add('deelt', 'delen')
    assert trees.tree_to_str(tree) == "(m 0 3 () (m 0 2 (s '' 'l') (s 'lt' 'n')))"
    tree = trees.add('gedeeld', 'delen')
    assert trees.tree_to_str(tree) == "(m 2 3 (s 'ge' '') (m 0 2 (s '' 'l') (s 'ld' 'n')))"

def test_from_to_bytes():
    if False:
        return 10
    strings = StringStore()
    trees = EditTrees(strings)
    trees.add('deelt', 'delen')
    trees.add('gedeeld', 'delen')
    b = trees.to_bytes()
    trees2 = EditTrees(strings)
    trees2.from_bytes(b)
    assert len(trees) == len(trees2)
    for i in range(len(trees)):
        assert trees.tree_to_str(i) == trees2.tree_to_str(i)
    trees2.add('deelt', 'delen')
    trees2.add('gedeeld', 'delen')
    assert len(trees) == len(trees2)

def test_from_to_disk():
    if False:
        print('Hello World!')
    strings = StringStore()
    trees = EditTrees(strings)
    trees.add('deelt', 'delen')
    trees.add('gedeeld', 'delen')
    trees2 = EditTrees(strings)
    with make_tempdir() as temp_dir:
        trees_file = temp_dir / 'edit_trees.bin'
        trees.to_disk(trees_file)
        trees2 = trees2.from_disk(trees_file)
    assert len(trees) == len(trees2)
    for i in range(len(trees)):
        assert trees.tree_to_str(i) == trees2.tree_to_str(i)
    trees2.add('deelt', 'delen')
    trees2.add('gedeeld', 'delen')
    assert len(trees) == len(trees2)

@given(st.text(), st.text())
def test_roundtrip(form, lemma):
    if False:
        i = 10
        return i + 15
    strings = StringStore()
    trees = EditTrees(strings)
    tree = trees.add(form, lemma)
    assert trees.apply(tree, form) == lemma

@given(st.text(alphabet='ab'), st.text(alphabet='ab'))
def test_roundtrip_small_alphabet(form, lemma):
    if False:
        i = 10
        return i + 15
    strings = StringStore()
    trees = EditTrees(strings)
    tree = trees.add(form, lemma)
    assert trees.apply(tree, form) == lemma

def test_unapplicable_trees():
    if False:
        print('Hello World!')
    strings = StringStore()
    trees = EditTrees(strings)
    tree3 = trees.add('deelt', 'delen')
    assert trees.apply(tree3, 'deeld') == None
    assert trees.apply(tree3, 'de') == None

def test_empty_strings():
    if False:
        while True:
            i = 10
    strings = StringStore()
    trees = EditTrees(strings)
    no_change = trees.add('xyz', 'xyz')
    empty = trees.add('', '')
    assert no_change == empty