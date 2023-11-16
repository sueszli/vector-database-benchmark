"""
Test a couple different classes of trees to check the output of the Arboretum conversion

Note that the text has been removed
"""
import os
import tempfile
import pytest
from stanza.server import tsurgeon
from stanza.tests import TEST_WORKING_DIR
from stanza.utils.datasets.constituency import convert_arboretum
pytestmark = [pytest.mark.pipeline, pytest.mark.travis]
PROJ_EXAMPLE = '\n<s id="s2" ref="AACBPIGY" source="id=AACBPIGY" forest="1/1" text="A B C D E F G H.">\n\t<graph root="s2_500">\n\t\t<terminals>\n\t\t\t<t id="s2_1" word="A" lemma="A" pos="prop" morph="NOM" extra="PROP:A compound brand"/>\n\t\t\t<t id="s2_2" word="B" lemma="B" pos="v-fin" morph="PR AKT" extra="mv"/>\n\t\t\t<t id="s2_3" word="C" lemma="C" pos="pron-pers" morph="2S ACC" extra="--"/>\n\t\t\t<t id="s2_4" word="D" lemma="D" pos="adj" morph="UTR S IDF NOM" extra="F:u+afhængig"/>\n\t\t\t<t id="s2_5" word="E" lemma="E" pos="prp" morph="--" extra="--"/>\n\t\t\t<t id="s2_6" word="F" lemma="F" pos="art" morph="NEU S DEF" extra="--"/>\n\t\t\t<t id="s2_7" word="G" lemma="G" pos="adj" morph="nG S DEF NOM" extra="--"/>\n\t\t\t<t id="s2_8" word="H" lemma="H" pos="n" morph="NEU S IDF NOM" extra="N:lys+net"/>\n\t\t\t<t id="s2_9" word="." lemma="--" pos="pu" morph="--" extra="--"/>\n\t\t</terminals>\n\n\t\t<nonterminals>\n\t\t\t<nt id="s2_500" cat="s">\n\t\t\t\t<edge label="STA" idref="s2_501"/>\n\t\t\t</nt>\n\t\t\t<nt id="s2_501" cat="fcl">\n\t\t\t\t<edge label="S" idref="s2_1"/>\n\t\t\t\t<edge label="P" idref="s2_2"/>\n\t\t\t\t<edge label="Od" idref="s2_3"/>\n\t\t\t\t<edge label="Co" idref="s2_502"/>\n\t\t\t\t<edge label="PU" idref="s2_9"/>\n\t\t\t</nt>\n\t\t\t<nt id="s2_502" cat="adjp">\n\t\t\t\t<edge label="H" idref="s2_4"/>\n\t\t\t\t<edge label="DA" idref="s2_503"/>\n\t\t\t</nt>\n\t\t\t<nt id="s2_503" cat="pp">\n\t\t\t\t<edge label="H" idref="s2_5"/>\n\t\t\t\t<edge label="DP" idref="s2_504"/>\n\t\t\t</nt>\n\t\t\t<nt id="s2_504" cat="np">\n\t\t\t\t<edge label="DN" idref="s2_6"/>\n\t\t\t\t<edge label="DN" idref="s2_7"/>\n\t\t\t\t<edge label="H" idref="s2_8"/>\n\t\t\t</nt>\n\t\t</nonterminals>\n\t</graph>\n</s>\n'
NOT_FIX_NONPROJ_EXAMPLE = '\n<s id="s322" ref="EDGBITSZ" source="id=EDGBITSZ" forest="1/2" text="A B C D E, F G H I J.">\n        <graph root="s322_500">\n                <terminals>\n                        <t id="s322_1" word="A" lemma="A" pos="prop" morph="NOM" extra="hum fem"/>\n                        <t id="s322_2" word="B" lemma="B" pos="v-fin" morph="PR AKT" extra="mv"/>\n                        <t id="s322_3" word="C" lemma="C" pos="pron-dem" morph="UTR S" extra="dem"/>\n                        <t id="s322_4" word="D" lemma="D" pos="n" morph="UTR S IDF NOM" extra="--"/>\n                        <t id="s322_5" word="E" lemma="E" pos="adv" morph="--" extra="--"/>\n                        <t id="s322_6" word="," lemma="--" pos="pu" morph="--" extra="--"/>\n                        <t id="s322_7" word="F" lemma="F" pos="pron-rel" morph="--" extra="rel"/>\n                        <t id="s322_8" word="G" lemma="G" pos="prop" morph="NOM" extra="hum"/>\n                        <t id="s322_9" word="H" lemma="H" pos="v-fin" morph="IMPF AKT" extra="mv"/>\n                        <t id="s322_10" word="I" lemma="I" pos="prp" morph="--" extra="--"/>\n                        <t id="s322_11" word="J" lemma="J" pos="n" morph="UTR S DEF NOM" extra="F:ur+premiere"/>\n                        <t id="s322_12" word="." lemma="--" pos="pu" morph="--" extra="--"/>\n                </terminals>\n\n                <nonterminals>\n                        <nt id="s322_500" cat="s">\n                                <edge label="STA" idref="s322_501"/>\n                        </nt>\n                        <nt id="s322_501" cat="fcl">\n                                <edge label="S" idref="s322_1"/>\n                                <edge label="P" idref="s322_2"/>\n                                <edge label="Od" idref="s322_502"/>\n                                <edge label="Vpart" idref="s322_5"/>\n                                <edge label="PU" idref="s322_6"/>\n                                <edge label="PU" idref="s322_12"/>\n                        </nt>\n                        <nt id="s322_502" cat="np">\n                                <edge label="DN" idref="s322_3"/>\n                                <edge label="H" idref="s322_4"/>\n                                <edge label="DN" idref="s322_503"/>\n                        </nt>\n                        <nt id="s322_503" cat="fcl">\n                                <edge label="Od" idref="s322_7"/>\n                                <edge label="S" idref="s322_8"/>\n                                <edge label="P" idref="s322_9"/>\n                                <edge label="Ao" idref="s322_504"/>\n                        </nt>\n                        <nt id="s322_504" cat="pp">\n                                <edge label="H" idref="s322_10"/>\n                                <edge label="DP" idref="s322_11"/>\n                        </nt>\n                </nonterminals>\n        </graph>\n</s>\n'
NONPROJ_EXAMPLE = '\n<s id="s9" ref="AATCNKQZ" source="id=AATCNKQZ" forest="1/1" text="A B C D E F G H I.">\n        <graph root="s9_500">\n                <terminals>\n                        <t id="s9_1" word="A" lemma="A" pos="adv" morph="--" extra="--"/>\n                        <t id="s9_2" word="B" lemma="B" pos="adv" morph="--" extra="--"/>\n                        <t id="s9_3" word="C" lemma="C" pos="v-fin" morph="IMPF AKT" extra="aux"/>\n                        <t id="s9_4" word="D" lemma="D" pos="prop" morph="NOM" extra="hum"/>\n                        <t id="s9_5" word="E" lemma="E" pos="adv" morph="--" extra="--"/>\n                        <t id="s9_6" word="F" lemma="F" pos="v-pcp2" morph="PAS" extra="mv"/>\n                        <t id="s9_7" word="G" lemma="G" pos="prp" morph="--" extra="--"/>\n                        <t id="s9_8" word="H" lemma="H" pos="num" morph="--" extra="card"/>\n                        <t id="s9_9" word="I" lemma="I" pos="n" morph="UTR P IDF NOM" extra="N:patrulje+vogn"/>\n                        <t id="s9_10" word="." lemma="--" pos="pu" morph="--" extra="--"/>\n                </terminals>\n\n                <nonterminals>\n                        <nt id="s9_500" cat="s">\n                                <edge label="STA" idref="s9_501"/>\n                        </nt>\n                        <nt id="s9_501" cat="fcl">\n                                <edge label="fA" idref="s9_502"/>\n                                <edge label="P" idref="s9_503"/>\n                                <edge label="S" idref="s9_4"/>\n                                <edge label="fA" idref="s9_5"/>\n                                <edge label="fA" idref="s9_504"/>\n                                <edge label="PU" idref="s9_10"/>\n                        </nt>\n                        <nt id="s9_502" cat="advp">\n                                <edge label="DA" idref="s9_1"/>\n                                <edge label="H" idref="s9_2"/>\n                        </nt>\n                        <nt id="s9_503" cat="vp">\n                                <edge label="Vaux" idref="s9_3"/>\n                                <edge label="Vm" idref="s9_6"/>\n                        </nt>\n                        <nt id="s9_504" cat="pp">\n                                <edge label="H" idref="s9_7"/>\n                                <edge label="DP" idref="s9_505"/>\n                        </nt>\n                        <nt id="s9_505" cat="np">\n                                <edge label="DN" idref="s9_8"/>\n                                <edge label="H" idref="s9_9"/>\n                        </nt>\n                </nonterminals>\n        </graph>\n</s>\n'

def test_projective_example():
    if False:
        while True:
            i = 10
    '\n    Test reading a basic tree, along with some further manipulations from the conversion program\n    '
    with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as tempdir:
        test_name = os.path.join(tempdir, 'proj.xml')
        with open(test_name, 'w', encoding='utf-8') as fout:
            fout.write(PROJ_EXAMPLE)
        sentences = convert_arboretum.read_xml_file(test_name)
        assert len(sentences) == 1
    (tree, words) = convert_arboretum.process_tree(sentences[0])
    expected_tree = '(s (fcl (prop s2_1) (v-fin s2_2) (pron-pers s2_3) (adjp (adj s2_4) (pp (prp s2_5) (np (art s2_6) (adj s2_7) (n s2_8)))) (pu s2_9)))'
    assert str(tree) == expected_tree
    assert [w.word for w in words.values()] == ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', '.']
    assert not convert_arboretum.word_sequence_missing_words(tree)
    with tsurgeon.Tsurgeon() as tsurgeon_processor:
        assert tree == convert_arboretum.check_words(tree, tsurgeon_processor)
    replaced_tree = convert_arboretum.replace_words(tree, words)
    expected_tree = '(s (fcl (prop A) (v-fin B) (pron-pers C) (adjp (adj D) (pp (prp E) (np (art F) (adj G) (n H)))) (pu .)))'
    assert str(replaced_tree) == expected_tree
    assert convert_arboretum.split_underscores(replaced_tree) == replaced_tree
    words['s2_1'] = words['s2_1']._replace(word='foo_bar')
    replaced_tree = convert_arboretum.replace_words(tree, words)
    expected_tree = '(s (fcl (prop foo_bar) (v-fin B) (pron-pers C) (adjp (adj D) (pp (prp E) (np (art F) (adj G) (n H)))) (pu .)))'
    assert str(replaced_tree) == expected_tree
    expected_tree = '(s (fcl (np (prop foo) (prop bar)) (v-fin B) (pron-pers C) (adjp (adj D) (pp (prp E) (np (art F) (adj G) (n H)))) (pu .)))'
    assert str(convert_arboretum.split_underscores(replaced_tree)) == expected_tree

def test_not_fix_example():
    if False:
        for i in range(10):
            print('nop')
    "\n    Test that a non-projective tree which we don't have a heuristic for quietly fails\n    "
    with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as tempdir:
        test_name = os.path.join(tempdir, 'nofix.xml')
        with open(test_name, 'w', encoding='utf-8') as fout:
            fout.write(NOT_FIX_NONPROJ_EXAMPLE)
        sentences = convert_arboretum.read_xml_file(test_name)
        assert len(sentences) == 1
    (tree, words) = convert_arboretum.process_tree(sentences[0])
    assert not convert_arboretum.word_sequence_missing_words(tree)
    with tsurgeon.Tsurgeon() as tsurgeon_processor:
        assert convert_arboretum.check_words(tree, tsurgeon_processor) is None

def test_fix_proj_example():
    if False:
        print('Hello World!')
    '\n    Test that a non-projective tree can be rearranged as expected\n\n    Note that there are several other classes of non-proj tree we could test as well...\n    '
    with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as tempdir:
        test_name = os.path.join(tempdir, 'fix.xml')
        with open(test_name, 'w', encoding='utf-8') as fout:
            fout.write(NONPROJ_EXAMPLE)
        sentences = convert_arboretum.read_xml_file(test_name)
        assert len(sentences) == 1
    (tree, words) = convert_arboretum.process_tree(sentences[0])
    assert not convert_arboretum.word_sequence_missing_words(tree)
    expected_orig = '(s (fcl (advp (adv s9_1) (adv s9_2)) (vp (v-fin s9_3) (v-pcp2 s9_6)) (prop s9_4) (adv s9_5) (pp (prp s9_7) (np (num s9_8) (n s9_9))) (pu s9_10)))'
    expected_proj = '(s (fcl (advp (adv s9_1) (adv s9_2)) (vp (v-fin s9_3) (prop s9_4) (adv s9_5) (v-pcp2 s9_6)) (pp (prp s9_7) (np (num s9_8) (n s9_9))) (pu s9_10)))'
    assert str(tree) == expected_orig
    with tsurgeon.Tsurgeon() as tsurgeon_processor:
        assert str(convert_arboretum.check_words(tree, tsurgeon_processor)) == expected_proj