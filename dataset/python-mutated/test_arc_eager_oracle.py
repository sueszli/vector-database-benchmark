import pytest
from spacy import registry
from spacy.pipeline import DependencyParser
from spacy.pipeline._parser_internals.arc_eager import ArcEager
from spacy.pipeline._parser_internals.nonproj import projectivize
from spacy.pipeline.dep_parser import DEFAULT_PARSER_MODEL
from spacy.tokens import Doc
from spacy.training import Example
from spacy.vocab import Vocab

def get_sequence_costs(M, words, heads, deps, transitions):
    if False:
        while True:
            i = 10
    doc = Doc(Vocab(), words=words)
    example = Example.from_dict(doc, {'heads': heads, 'deps': deps})
    (states, golds, _) = M.init_gold_batch([example])
    state = states[0]
    gold = golds[0]
    cost_history = []
    for gold_action in transitions:
        gold.update(state)
        state_costs = {}
        for i in range(M.n_moves):
            name = M.class_name(i)
            state_costs[name] = M.get_cost(state, gold, i)
        M.transition(state, gold_action)
        cost_history.append(state_costs)
    return (state, cost_history)

@pytest.fixture
def vocab():
    if False:
        print('Hello World!')
    return Vocab()

@pytest.fixture
def arc_eager(vocab):
    if False:
        return 10
    moves = ArcEager(vocab.strings, ArcEager.get_actions())
    moves.add_action(2, 'left')
    moves.add_action(3, 'right')
    return moves

@pytest.mark.issue(7056)
def test_issue7056():
    if False:
        i = 10
        return i + 15
    "Test that the Unshift transition works properly, and doesn't cause\n    sentence segmentation errors."
    vocab = Vocab()
    ae = ArcEager(vocab.strings, ArcEager.get_actions(left_labels=['amod'], right_labels=['pobj']))
    doc = Doc(vocab, words='Severe pain , after trauma'.split())
    state = ae.init_batch([doc])[0]
    ae.apply_transition(state, 'S')
    ae.apply_transition(state, 'L-amod')
    ae.apply_transition(state, 'S')
    ae.apply_transition(state, 'S')
    ae.apply_transition(state, 'S')
    ae.apply_transition(state, 'R-pobj')
    ae.apply_transition(state, 'D')
    ae.apply_transition(state, 'D')
    ae.apply_transition(state, 'D')
    assert not state.eol()

def test_oracle_four_words(arc_eager, vocab):
    if False:
        print('Hello World!')
    words = ['a', 'b', 'c', 'd']
    heads = [1, 1, 3, 3]
    deps = ['left', 'ROOT', 'left', 'ROOT']
    for dep in deps:
        arc_eager.add_action(2, dep)
        arc_eager.add_action(3, dep)
    actions = ['S', 'L-left', 'B-ROOT', 'S', 'D', 'S', 'L-left', 'S', 'D']
    (state, cost_history) = get_sequence_costs(arc_eager, words, heads, deps, actions)
    expected_gold = [['S'], ['B-ROOT', 'L-left'], ['B-ROOT'], ['S'], ['D'], ['S'], ['L-left'], ['S'], ['D']]
    assert state.is_final()
    for (i, state_costs) in enumerate(cost_history):
        golds = [act for (act, cost) in state_costs.items() if cost < 1]
        assert golds == expected_gold[i], (i, golds, expected_gold[i])
annot_tuples = [(0, 'When', 'WRB', 11, 'advmod', 'O'), (1, 'Walter', 'NNP', 2, 'compound', 'B-PERSON'), (2, 'Rodgers', 'NNP', 11, 'nsubj', 'L-PERSON'), (3, ',', ',', 2, 'punct', 'O'), (4, 'our', 'PRP$', 6, 'poss', 'O'), (5, 'embedded', 'VBN', 6, 'amod', 'O'), (6, 'reporter', 'NN', 2, 'appos', 'O'), (7, 'with', 'IN', 6, 'prep', 'O'), (8, 'the', 'DT', 10, 'det', 'B-ORG'), (9, '3rd', 'NNP', 10, 'compound', 'I-ORG'), (10, 'Cavalry', 'NNP', 7, 'pobj', 'L-ORG'), (11, 'says', 'VBZ', 44, 'advcl', 'O'), (12, 'three', 'CD', 13, 'nummod', 'U-CARDINAL'), (13, 'battalions', 'NNS', 16, 'nsubj', 'O'), (14, 'of', 'IN', 13, 'prep', 'O'), (15, 'troops', 'NNS', 14, 'pobj', 'O'), (16, 'are', 'VBP', 11, 'ccomp', 'O'), (17, 'on', 'IN', 16, 'prep', 'O'), (18, 'the', 'DT', 19, 'det', 'O'), (19, 'ground', 'NN', 17, 'pobj', 'O'), (20, ',', ',', 17, 'punct', 'O'), (21, 'inside', 'IN', 17, 'prep', 'O'), (22, 'Baghdad', 'NNP', 21, 'pobj', 'U-GPE'), (23, 'itself', 'PRP', 22, 'appos', 'O'), (24, ',', ',', 16, 'punct', 'O'), (25, 'have', 'VBP', 26, 'aux', 'O'), (26, 'taken', 'VBN', 16, 'dep', 'O'), (27, 'up', 'RP', 26, 'prt', 'O'), (28, 'positions', 'NNS', 26, 'dobj', 'O'), (29, 'they', 'PRP', 31, 'nsubj', 'O'), (30, "'re", 'VBP', 31, 'aux', 'O'), (31, 'going', 'VBG', 26, 'parataxis', 'O'), (32, 'to', 'TO', 33, 'aux', 'O'), (33, 'spend', 'VB', 31, 'xcomp', 'O'), (34, 'the', 'DT', 35, 'det', 'B-TIME'), (35, 'night', 'NN', 33, 'dobj', 'L-TIME'), (36, 'there', 'RB', 33, 'advmod', 'O'), (37, 'presumably', 'RB', 33, 'advmod', 'O'), (38, ',', ',', 44, 'punct', 'O'), (39, 'how', 'WRB', 40, 'advmod', 'O'), (40, 'many', 'JJ', 41, 'amod', 'O'), (41, 'soldiers', 'NNS', 44, 'pobj', 'O'), (42, 'are', 'VBP', 44, 'aux', 'O'), (43, 'we', 'PRP', 44, 'nsubj', 'O'), (44, 'talking', 'VBG', 44, 'ROOT', 'O'), (45, 'about', 'IN', 44, 'prep', 'O'), (46, 'right', 'RB', 47, 'advmod', 'O'), (47, 'now', 'RB', 44, 'advmod', 'O'), (48, '?', '.', 44, 'punct', 'O')]

def test_get_oracle_actions():
    if False:
        while True:
            i = 10
    (ids, words, tags, heads, deps, ents) = ([], [], [], [], [], [])
    for (id_, word, tag, head, dep, ent) in annot_tuples:
        ids.append(id_)
        words.append(word)
        tags.append(tag)
        heads.append(head)
        deps.append(dep)
        ents.append(ent)
    doc = Doc(Vocab(), words=[t[1] for t in annot_tuples])
    cfg = {'model': DEFAULT_PARSER_MODEL}
    model = registry.resolve(cfg, validate=True)['model']
    parser = DependencyParser(doc.vocab, model)
    parser.moves.add_action(0, '')
    parser.moves.add_action(1, '')
    parser.moves.add_action(1, '')
    parser.moves.add_action(4, 'ROOT')
    (heads, deps) = projectivize(heads, deps)
    for (i, (head, dep)) in enumerate(zip(heads, deps)):
        if head > i:
            parser.moves.add_action(2, dep)
        elif head < i:
            parser.moves.add_action(3, dep)
    example = Example.from_dict(doc, {'words': words, 'tags': tags, 'heads': heads, 'deps': deps})
    parser.moves.get_oracle_sequence(example)

def test_oracle_dev_sentence(vocab, arc_eager):
    if False:
        while True:
            i = 10
    words_deps_heads = '\n        Rolls-Royce nn Inc.\n        Motor nn Inc.\n        Cars nn Inc.\n        Inc. nsubj said\n        said ROOT said\n        it nsubj expects\n        expects ccomp said\n        its poss sales\n        U.S. nn sales\n        sales nsubj steady\n        to aux steady\n        remain cop steady\n        steady xcomp expects\n        at prep steady\n        about quantmod 1,200\n        1,200 num cars\n        cars pobj at\n        in prep steady\n        1990 pobj in\n        . punct said\n    '
    expected_transitions = ['S', 'S', 'S', 'L-nn', 'L-nn', 'L-nn', 'S', 'L-nsubj', 'S', 'S', 'L-nsubj', 'R-ccomp', 'S', 'S', 'L-nn', 'L-poss', 'S', 'S', 'S', 'L-cop', 'L-aux', 'L-nsubj', 'R-xcomp', 'R-prep', 'S', 'L-quantmod', 'S', 'L-num', 'R-pobj', 'D', 'D', 'R-prep', 'R-pobj', 'D', 'D', 'D', 'D', 'R-punct', 'D', 'D']
    gold_words = []
    gold_deps = []
    gold_heads = []
    for line in words_deps_heads.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        (word, dep, head) = line.split()
        gold_words.append(word)
        gold_deps.append(dep)
        gold_heads.append(head)
    gold_heads = [gold_words.index(head) for head in gold_heads]
    for dep in gold_deps:
        arc_eager.add_action(2, dep)
        arc_eager.add_action(3, dep)
    doc = Doc(Vocab(), words=gold_words)
    example = Example.from_dict(doc, {'heads': gold_heads, 'deps': gold_deps})
    ae_oracle_actions = arc_eager.get_oracle_sequence(example, _debug=False)
    ae_oracle_actions = [arc_eager.get_class_name(i) for i in ae_oracle_actions]
    assert ae_oracle_actions == expected_transitions

def test_oracle_bad_tokenization(vocab, arc_eager):
    if False:
        return 10
    words_deps_heads = '\n        [catalase] dep is\n        : punct is\n        that nsubj is\n        is root is\n        bad comp is\n    '
    gold_words = []
    gold_deps = []
    gold_heads = []
    for line in words_deps_heads.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        (word, dep, head) = line.split()
        gold_words.append(word)
        gold_deps.append(dep)
        gold_heads.append(head)
    gold_heads = [gold_words.index(head) for head in gold_heads]
    for dep in gold_deps:
        arc_eager.add_action(2, dep)
        arc_eager.add_action(3, dep)
    reference = Doc(Vocab(), words=gold_words, deps=gold_deps, heads=gold_heads)
    predicted = Doc(reference.vocab, words=['[', 'catalase', ']', ':', 'that', 'is', 'bad'])
    example = Example(predicted=predicted, reference=reference)
    ae_oracle_actions = arc_eager.get_oracle_sequence(example, _debug=False)
    ae_oracle_actions = [arc_eager.get_class_name(i) for i in ae_oracle_actions]
    assert ae_oracle_actions