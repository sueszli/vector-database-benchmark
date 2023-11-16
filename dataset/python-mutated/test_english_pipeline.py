"""
Basic testing of the English pipeline
"""
import pytest
import stanza
from stanza.utils.conll import CoNLL
from stanza.models.common.doc import Document
from stanza.tests import *
from stanza.tests.pipeline.pipeline_device_tests import check_on_gpu, check_on_cpu
pytestmark = [pytest.mark.pipeline, pytest.mark.travis]
EN_DOC = 'Barack Obama was born in Hawaii.  He was elected president in 2008.  Obama attended Harvard.'
EN_DOCS = ['Barack Obama was born in Hawaii.', 'He was elected president in 2008.', 'Obama attended Harvard.']
EN_DOC_TOKENS_GOLD = '\n<Token id=1;words=[<Word id=1;text=Barack;lemma=Barack;upos=PROPN;xpos=NNP;feats=Number=Sing;head=4;deprel=nsubj:pass>]>\n<Token id=2;words=[<Word id=2;text=Obama;lemma=Obama;upos=PROPN;xpos=NNP;feats=Number=Sing;head=1;deprel=flat>]>\n<Token id=3;words=[<Word id=3;text=was;lemma=be;upos=AUX;xpos=VBD;feats=Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin;head=4;deprel=aux:pass>]>\n<Token id=4;words=[<Word id=4;text=born;lemma=bear;upos=VERB;xpos=VBN;feats=Tense=Past|VerbForm=Part|Voice=Pass;head=0;deprel=root>]>\n<Token id=5;words=[<Word id=5;text=in;lemma=in;upos=ADP;xpos=IN;head=6;deprel=case>]>\n<Token id=6;words=[<Word id=6;text=Hawaii;lemma=Hawaii;upos=PROPN;xpos=NNP;feats=Number=Sing;head=4;deprel=obl>]>\n<Token id=7;words=[<Word id=7;text=.;lemma=.;upos=PUNCT;xpos=.;head=4;deprel=punct>]>\n\n<Token id=1;words=[<Word id=1;text=He;lemma=he;upos=PRON;xpos=PRP;feats=Case=Nom|Gender=Masc|Number=Sing|Person=3|PronType=Prs;head=3;deprel=nsubj:pass>]>\n<Token id=2;words=[<Word id=2;text=was;lemma=be;upos=AUX;xpos=VBD;feats=Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin;head=3;deprel=aux:pass>]>\n<Token id=3;words=[<Word id=3;text=elected;lemma=elect;upos=VERB;xpos=VBN;feats=Tense=Past|VerbForm=Part|Voice=Pass;head=0;deprel=root>]>\n<Token id=4;words=[<Word id=4;text=president;lemma=president;upos=NOUN;xpos=NN;feats=Number=Sing;head=3;deprel=xcomp>]>\n<Token id=5;words=[<Word id=5;text=in;lemma=in;upos=ADP;xpos=IN;head=6;deprel=case>]>\n<Token id=6;words=[<Word id=6;text=2008;lemma=2008;upos=NUM;xpos=CD;feats=NumForm=Digit|NumType=Card;head=3;deprel=obl>]>\n<Token id=7;words=[<Word id=7;text=.;lemma=.;upos=PUNCT;xpos=.;head=3;deprel=punct>]>\n\n<Token id=1;words=[<Word id=1;text=Obama;lemma=Obama;upos=PROPN;xpos=NNP;feats=Number=Sing;head=2;deprel=nsubj>]>\n<Token id=2;words=[<Word id=2;text=attended;lemma=attend;upos=VERB;xpos=VBD;feats=Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin;head=0;deprel=root>]>\n<Token id=3;words=[<Word id=3;text=Harvard;lemma=Harvard;upos=PROPN;xpos=NNP;feats=Number=Sing;head=2;deprel=obj>]>\n<Token id=4;words=[<Word id=4;text=.;lemma=.;upos=PUNCT;xpos=.;head=2;deprel=punct>]>\n'.strip()
EN_DOC_WORDS_GOLD = '\n<Word id=1;text=Barack;lemma=Barack;upos=PROPN;xpos=NNP;feats=Number=Sing;head=4;deprel=nsubj:pass>\n<Word id=2;text=Obama;lemma=Obama;upos=PROPN;xpos=NNP;feats=Number=Sing;head=1;deprel=flat>\n<Word id=3;text=was;lemma=be;upos=AUX;xpos=VBD;feats=Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin;head=4;deprel=aux:pass>\n<Word id=4;text=born;lemma=bear;upos=VERB;xpos=VBN;feats=Tense=Past|VerbForm=Part|Voice=Pass;head=0;deprel=root>\n<Word id=5;text=in;lemma=in;upos=ADP;xpos=IN;head=6;deprel=case>\n<Word id=6;text=Hawaii;lemma=Hawaii;upos=PROPN;xpos=NNP;feats=Number=Sing;head=4;deprel=obl>\n<Word id=7;text=.;lemma=.;upos=PUNCT;xpos=.;head=4;deprel=punct>\n\n<Word id=1;text=He;lemma=he;upos=PRON;xpos=PRP;feats=Case=Nom|Gender=Masc|Number=Sing|Person=3|PronType=Prs;head=3;deprel=nsubj:pass>\n<Word id=2;text=was;lemma=be;upos=AUX;xpos=VBD;feats=Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin;head=3;deprel=aux:pass>\n<Word id=3;text=elected;lemma=elect;upos=VERB;xpos=VBN;feats=Tense=Past|VerbForm=Part|Voice=Pass;head=0;deprel=root>\n<Word id=4;text=president;lemma=president;upos=NOUN;xpos=NN;feats=Number=Sing;head=3;deprel=xcomp>\n<Word id=5;text=in;lemma=in;upos=ADP;xpos=IN;head=6;deprel=case>\n<Word id=6;text=2008;lemma=2008;upos=NUM;xpos=CD;feats=NumForm=Digit|NumType=Card;head=3;deprel=obl>\n<Word id=7;text=.;lemma=.;upos=PUNCT;xpos=.;head=3;deprel=punct>\n\n<Word id=1;text=Obama;lemma=Obama;upos=PROPN;xpos=NNP;feats=Number=Sing;head=2;deprel=nsubj>\n<Word id=2;text=attended;lemma=attend;upos=VERB;xpos=VBD;feats=Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin;head=0;deprel=root>\n<Word id=3;text=Harvard;lemma=Harvard;upos=PROPN;xpos=NNP;feats=Number=Sing;head=2;deprel=obj>\n<Word id=4;text=.;lemma=.;upos=PUNCT;xpos=.;head=2;deprel=punct>\n'.strip()
EN_DOC_DEPENDENCY_PARSES_GOLD = "\n('Barack', 4, 'nsubj:pass')\n('Obama', 1, 'flat')\n('was', 4, 'aux:pass')\n('born', 0, 'root')\n('in', 6, 'case')\n('Hawaii', 4, 'obl')\n('.', 4, 'punct')\n\n('He', 3, 'nsubj:pass')\n('was', 3, 'aux:pass')\n('elected', 0, 'root')\n('president', 3, 'xcomp')\n('in', 6, 'case')\n('2008', 3, 'obl')\n('.', 3, 'punct')\n\n('Obama', 2, 'nsubj')\n('attended', 0, 'root')\n('Harvard', 2, 'obj')\n('.', 2, 'punct')\n".strip()
EN_DOC_CONLLU_GOLD = '\n# text = Barack Obama was born in Hawaii.\n# sent_id = 0\n# constituency = (ROOT (S (NP (NNP Barack) (NNP Obama)) (VP (VBD was) (VP (VBN born) (PP (IN in) (NP (NNP Hawaii))))) (. .)))\n# sentiment = 1\n1\tBarack\tBarack\tPROPN\tNNP\tNumber=Sing\t4\tnsubj:pass\t_\tstart_char=0|end_char=6|ner=B-PERSON\n2\tObama\tObama\tPROPN\tNNP\tNumber=Sing\t1\tflat\t_\tstart_char=7|end_char=12|ner=E-PERSON\n3\twas\tbe\tAUX\tVBD\tMood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin\t4\taux:pass\t_\tstart_char=13|end_char=16|ner=O\n4\tborn\tbear\tVERB\tVBN\tTense=Past|VerbForm=Part|Voice=Pass\t0\troot\t_\tstart_char=17|end_char=21|ner=O\n5\tin\tin\tADP\tIN\t_\t6\tcase\t_\tstart_char=22|end_char=24|ner=O\n6\tHawaii\tHawaii\tPROPN\tNNP\tNumber=Sing\t4\tobl\t_\tstart_char=25|end_char=31|ner=S-GPE\n7\t.\t.\tPUNCT\t.\t_\t4\tpunct\t_\tstart_char=31|end_char=32|ner=O\n\n# text = He was elected president in 2008.\n# sent_id = 1\n# constituency = (ROOT (S (NP (PRP He)) (VP (VBD was) (VP (VBN elected) (S (NP (NN president))) (PP (IN in) (NP (CD 2008))))) (. .)))\n# sentiment = 1\n1\tHe\the\tPRON\tPRP\tCase=Nom|Gender=Masc|Number=Sing|Person=3|PronType=Prs\t3\tnsubj:pass\t_\tstart_char=34|end_char=36|ner=O\n2\twas\tbe\tAUX\tVBD\tMood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin\t3\taux:pass\t_\tstart_char=37|end_char=40|ner=O\n3\telected\telect\tVERB\tVBN\tTense=Past|VerbForm=Part|Voice=Pass\t0\troot\t_\tstart_char=41|end_char=48|ner=O\n4\tpresident\tpresident\tNOUN\tNN\tNumber=Sing\t3\txcomp\t_\tstart_char=49|end_char=58|ner=O\n5\tin\tin\tADP\tIN\t_\t6\tcase\t_\tstart_char=59|end_char=61|ner=O\n6\t2008\t2008\tNUM\tCD\tNumForm=Digit|NumType=Card\t3\tobl\t_\tstart_char=62|end_char=66|ner=S-DATE\n7\t.\t.\tPUNCT\t.\t_\t3\tpunct\t_\tstart_char=66|end_char=67|ner=O\n\n# text = Obama attended Harvard.\n# sent_id = 2\n# constituency = (ROOT (S (NP (NNP Obama)) (VP (VBD attended) (NP (NNP Harvard))) (. .)))\n# sentiment = 1\n1\tObama\tObama\tPROPN\tNNP\tNumber=Sing\t2\tnsubj\t_\tstart_char=69|end_char=74|ner=S-PERSON\n2\tattended\tattend\tVERB\tVBD\tMood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin\t0\troot\t_\tstart_char=75|end_char=83|ner=O\n3\tHarvard\tHarvard\tPROPN\tNNP\tNumber=Sing\t2\tobj\t_\tstart_char=84|end_char=91|ner=S-ORG\n4\t.\t.\tPUNCT\t.\t_\t2\tpunct\t_\tstart_char=91|end_char=92|ner=O\n'.strip()
EN_DOC_CONLLU_GOLD_MULTIDOC = '\n# text = Barack Obama was born in Hawaii.\n# sent_id = 0\n# constituency = (ROOT (S (NP (NNP Barack) (NNP Obama)) (VP (VBD was) (VP (VBN born) (PP (IN in) (NP (NNP Hawaii))))) (. .)))\n# sentiment = 1\n1\tBarack\tBarack\tPROPN\tNNP\tNumber=Sing\t4\tnsubj:pass\t_\tstart_char=0|end_char=6|ner=B-PERSON\n2\tObama\tObama\tPROPN\tNNP\tNumber=Sing\t1\tflat\t_\tstart_char=7|end_char=12|ner=E-PERSON\n3\twas\tbe\tAUX\tVBD\tMood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin\t4\taux:pass\t_\tstart_char=13|end_char=16|ner=O\n4\tborn\tbear\tVERB\tVBN\tTense=Past|VerbForm=Part|Voice=Pass\t0\troot\t_\tstart_char=17|end_char=21|ner=O\n5\tin\tin\tADP\tIN\t_\t6\tcase\t_\tstart_char=22|end_char=24|ner=O\n6\tHawaii\tHawaii\tPROPN\tNNP\tNumber=Sing\t4\tobl\t_\tstart_char=25|end_char=31|ner=S-GPE\n7\t.\t.\tPUNCT\t.\t_\t4\tpunct\t_\tstart_char=31|end_char=32|ner=O\n\n# text = He was elected president in 2008.\n# sent_id = 1\n# constituency = (ROOT (S (NP (PRP He)) (VP (VBD was) (VP (VBN elected) (S (NP (NN president))) (PP (IN in) (NP (CD 2008))))) (. .)))\n# sentiment = 1\n1\tHe\the\tPRON\tPRP\tCase=Nom|Gender=Masc|Number=Sing|Person=3|PronType=Prs\t3\tnsubj:pass\t_\tstart_char=0|end_char=2|ner=O\n2\twas\tbe\tAUX\tVBD\tMood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin\t3\taux:pass\t_\tstart_char=3|end_char=6|ner=O\n3\telected\telect\tVERB\tVBN\tTense=Past|VerbForm=Part|Voice=Pass\t0\troot\t_\tstart_char=7|end_char=14|ner=O\n4\tpresident\tpresident\tNOUN\tNN\tNumber=Sing\t3\txcomp\t_\tstart_char=15|end_char=24|ner=O\n5\tin\tin\tADP\tIN\t_\t6\tcase\t_\tstart_char=25|end_char=27|ner=O\n6\t2008\t2008\tNUM\tCD\tNumForm=Digit|NumType=Card\t3\tobl\t_\tstart_char=28|end_char=32|ner=S-DATE\n7\t.\t.\tPUNCT\t.\t_\t3\tpunct\t_\tstart_char=32|end_char=33|ner=O\n\n# text = Obama attended Harvard.\n# sent_id = 2\n# constituency = (ROOT (S (NP (NNP Obama)) (VP (VBD attended) (NP (NNP Harvard))) (. .)))\n# sentiment = 1\n1\tObama\tObama\tPROPN\tNNP\tNumber=Sing\t2\tnsubj\t_\tstart_char=0|end_char=5|ner=S-PERSON\n2\tattended\tattend\tVERB\tVBD\tMood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin\t0\troot\t_\tstart_char=6|end_char=14|ner=O\n3\tHarvard\tHarvard\tPROPN\tNNP\tNumber=Sing\t2\tobj\t_\tstart_char=15|end_char=22|ner=S-ORG\n4\t.\t.\tPUNCT\t.\t_\t2\tpunct\t_\tstart_char=22|end_char=23|ner=O\n'.strip()

class TestEnglishPipeline:

    @pytest.fixture(scope='class')
    def pipeline(self):
        if False:
            while True:
                i = 10
        return stanza.Pipeline(dir=TEST_MODELS_DIR)

    @pytest.fixture(scope='class')
    def processed_doc(self, pipeline):
        if False:
            return 10
        ' Document created by running full English pipeline on a few sentences '
        return pipeline(EN_DOC)

    def test_text(self, processed_doc):
        if False:
            print('Hello World!')
        assert processed_doc.text == EN_DOC

    def test_conllu(self, processed_doc):
        if False:
            print('Hello World!')
        assert '{:C}'.format(processed_doc) == EN_DOC_CONLLU_GOLD

    def test_tokens(self, processed_doc):
        if False:
            return 10
        assert '\n\n'.join([sent.tokens_string() for sent in processed_doc.sentences]) == EN_DOC_TOKENS_GOLD

    def test_words(self, processed_doc):
        if False:
            print('Hello World!')
        assert '\n\n'.join([sent.words_string() for sent in processed_doc.sentences]) == EN_DOC_WORDS_GOLD

    def test_dependency_parse(self, processed_doc):
        if False:
            print('Hello World!')
        assert '\n\n'.join([sent.dependencies_string() for sent in processed_doc.sentences]) == EN_DOC_DEPENDENCY_PARSES_GOLD

    def test_empty(self, pipeline):
        if False:
            print('Hello World!')
        pipeline('')
        pipeline('--')

    def test_bulk_process(self, pipeline):
        if False:
            return 10
        ' Double check that the bulk_process method in Pipeline converts documents as expected '
        processed = pipeline.bulk_process(EN_DOCS)
        assert '\n\n'.join(['{:C}'.format(doc) for doc in processed]) == EN_DOC_CONLLU_GOLD_MULTIDOC
        docs = [Document([], text=t) for t in EN_DOCS]
        processed = pipeline.bulk_process(docs)
        assert '\n\n'.join(['{:C}'.format(doc) for doc in processed]) == EN_DOC_CONLLU_GOLD_MULTIDOC

    def test_empty_bulk_process(self, pipeline):
        if False:
            i = 10
            return i + 15
        ' Previously we had a bug where an empty document list would cause a crash '
        processed = pipeline.bulk_process([])
        assert processed == []

    def test_stream(self, pipeline):
        if False:
            i = 10
            return i + 15
        ' Test the streaming interface to the Pipeline '
        processed = [doc for doc in pipeline.stream(EN_DOCS)]
        assert '\n\n'.join(['{:C}'.format(doc) for doc in processed]) == EN_DOC_CONLLU_GOLD_MULTIDOC
        processed = [doc for doc in pipeline.stream(iter(EN_DOCS))]
        assert '\n\n'.join(['{:C}'.format(doc) for doc in processed]) == EN_DOC_CONLLU_GOLD_MULTIDOC
        processed = [doc for doc in pipeline.stream(EN_DOCS, batch_size=1)]
        processed = ['{:C}'.format(doc) for doc in processed]
        assert '\n\n'.join(processed) == EN_DOC_CONLLU_GOLD_MULTIDOC

    @pytest.fixture(scope='class')
    def processed_multidoc(self, pipeline):
        if False:
            for i in range(10):
                print('nop')
        ' Document created by running full English pipeline on a few sentences '
        docs = [Document([], text=t) for t in EN_DOCS]
        return pipeline(docs)

    def test_conllu_multidoc(self, processed_multidoc):
        if False:
            return 10
        assert '\n\n'.join(['{:C}'.format(doc) for doc in processed_multidoc]) == EN_DOC_CONLLU_GOLD_MULTIDOC

    def test_tokens_multidoc(self, processed_multidoc):
        if False:
            while True:
                i = 10
        assert '\n\n'.join([sent.tokens_string() for processed_doc in processed_multidoc for sent in processed_doc.sentences]) == EN_DOC_TOKENS_GOLD

    def test_words_multidoc(self, processed_multidoc):
        if False:
            i = 10
            return i + 15
        assert '\n\n'.join([sent.words_string() for processed_doc in processed_multidoc for sent in processed_doc.sentences]) == EN_DOC_WORDS_GOLD

    def test_sentence_indices_multidoc(self, processed_multidoc):
        if False:
            for i in range(10):
                print('nop')
        sentences = [sent for doc in processed_multidoc for sent in doc.sentences]
        for (sent_idx, sentence) in enumerate(sentences):
            assert sent_idx == sentence.index

    def test_dependency_parse_multidoc(self, processed_multidoc):
        if False:
            for i in range(10):
                print('nop')
        assert '\n\n'.join([sent.dependencies_string() for processed_doc in processed_multidoc for sent in processed_doc.sentences]) == EN_DOC_DEPENDENCY_PARSES_GOLD

    @pytest.fixture(scope='class')
    def processed_multidoc_variant(self):
        if False:
            while True:
                i = 10
        ' Document created by running full English pipeline on a few sentences '
        docs = [Document([], text=t) for t in EN_DOCS]
        nlp = stanza.Pipeline(dir=TEST_MODELS_DIR, processors={'tokenize': 'spacy'})
        return nlp(docs)

    def test_dependency_parse_multidoc_variant(self, processed_multidoc_variant):
        if False:
            return 10
        assert '\n\n'.join([sent.dependencies_string() for processed_doc in processed_multidoc_variant for sent in processed_doc.sentences]) == EN_DOC_DEPENDENCY_PARSES_GOLD

    def test_constituency_parser(self):
        if False:
            while True:
                i = 10
        nlp = stanza.Pipeline(dir=TEST_MODELS_DIR, processors='tokenize,pos,constituency')
        doc = nlp('This is a test')
        assert str(doc.sentences[0].constituency) == '(ROOT (S (NP (DT This)) (VP (VBZ is) (NP (DT a) (NN test)))))'

    def test_on_gpu(self, pipeline):
        if False:
            i = 10
            return i + 15
        '\n        The default pipeline should have all the models on the GPU\n        '
        check_on_gpu(pipeline)

    def test_on_cpu(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a pipeline on the CPU, check that all the models on CPU\n        '
        pipeline = stanza.Pipeline('en', dir=TEST_MODELS_DIR, use_gpu=False)
        check_on_cpu(pipeline)