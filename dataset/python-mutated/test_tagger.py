"""
Run the tagger for a couple iterations on some fake data

Uses a couple sentences of UD_English-EWT as training/dev data
"""
import os
import pytest
from stanza.models import tagger
from stanza.models.common import pretrain
from stanza.models.pos.trainer import Trainer
from stanza.tests import TEST_WORKING_DIR, TEST_MODELS_DIR
from stanza.utils.training.common import choose_pos_charlm, build_charlm_args
pytestmark = [pytest.mark.pipeline, pytest.mark.travis]
TRAIN_DATA = '\n# sent_id = weblog-juancole.com_juancole_20051126063000_ENG_20051126_063000-0003\n# text = DPA: Iraqi authorities announced that they had busted up 3 terrorist cells operating in Baghdad.\n1\tDPA\tDPA\tPROPN\tNNP\tNumber=Sing\t0\troot\t0:root\tSpaceAfter=No\n2\t:\t:\tPUNCT\t:\t_\t1\tpunct\t1:punct\t_\n3\tIraqi\tIraqi\tADJ\tJJ\tDegree=Pos\t4\tamod\t4:amod\t_\n4\tauthorities\tauthority\tNOUN\tNNS\tNumber=Plur\t5\tnsubj\t5:nsubj\t_\n5\tannounced\tannounce\tVERB\tVBD\tMood=Ind|Number=Plur|Person=3|Tense=Past|VerbForm=Fin\t1\tparataxis\t1:parataxis\t_\n6\tthat\tthat\tSCONJ\tIN\t_\t9\tmark\t9:mark\t_\n7\tthey\tthey\tPRON\tPRP\tCase=Nom|Number=Plur|Person=3|PronType=Prs\t9\tnsubj\t9:nsubj\t_\n8\thad\thave\tAUX\tVBD\tMood=Ind|Number=Plur|Person=3|Tense=Past|VerbForm=Fin\t9\taux\t9:aux\t_\n9\tbusted\tbust\tVERB\tVBN\tTense=Past|VerbForm=Part\t5\tccomp\t5:ccomp\t_\n10\tup\tup\tADP\tRP\t_\t9\tcompound:prt\t9:compound:prt\t_\n11\t3\t3\tNUM\tCD\tNumForm=Digit|NumType=Card\t13\tnummod\t13:nummod\t_\n12\tterrorist\tterrorist\tADJ\tJJ\tDegree=Pos\t13\tamod\t13:amod\t_\n13\tcells\tcell\tNOUN\tNNS\tNumber=Plur\t9\tobj\t9:obj\t_\n14\toperating\toperate\tVERB\tVBG\tVerbForm=Ger\t13\tacl\t13:acl\t_\n15\tin\tin\tADP\tIN\t_\t16\tcase\t16:case\t_\n16\tBaghdad\tBaghdad\tPROPN\tNNP\tNumber=Sing\t14\tobl\t14:obl:in\tSpaceAfter=No\n17\t.\t.\tPUNCT\t.\t_\t1\tpunct\t1:punct\t_\n\n# sent_id = weblog-juancole.com_juancole_20051126063000_ENG_20051126_063000-0004\n# text = Two of them were being run by 2 officials of the Ministry of the Interior!\n1\tTwo\ttwo\tNUM\tCD\tNumForm=Word|NumType=Card\t6\tnsubj:pass\t6:nsubj:pass\t_\n2\tof\tof\tADP\tIN\t_\t3\tcase\t3:case\t_\n3\tthem\tthey\tPRON\tPRP\tCase=Acc|Number=Plur|Person=3|PronType=Prs\t1\tnmod\t1:nmod:of\t_\n4\twere\tbe\tAUX\tVBD\tMood=Ind|Number=Plur|Person=3|Tense=Past|VerbForm=Fin\t6\taux\t6:aux\t_\n5\tbeing\tbe\tAUX\tVBG\tVerbForm=Ger\t6\taux:pass\t6:aux:pass\t_\n6\trun\trun\tVERB\tVBN\tTense=Past|VerbForm=Part|Voice=Pass\t0\troot\t0:root\t_\n7\tby\tby\tADP\tIN\t_\t9\tcase\t9:case\t_\n8\t2\t2\tNUM\tCD\tNumForm=Digit|NumType=Card\t9\tnummod\t9:nummod\t_\n9\tofficials\tofficial\tNOUN\tNNS\tNumber=Plur\t6\tobl\t6:obl:by\t_\n10\tof\tof\tADP\tIN\t_\t12\tcase\t12:case\t_\n11\tthe\tthe\tDET\tDT\tDefinite=Def|PronType=Art\t12\tdet\t12:det\t_\n12\tMinistry\tMinistry\tPROPN\tNNP\tNumber=Sing\t9\tnmod\t9:nmod:of\t_\n13\tof\tof\tADP\tIN\t_\t15\tcase\t15:case\t_\n14\tthe\tthe\tDET\tDT\tDefinite=Def|PronType=Art\t15\tdet\t15:det\t_\n15\tInterior\tInterior\tPROPN\tNNP\tNumber=Sing\t12\tnmod\t12:nmod:of\tSpaceAfter=No\n16\t!\t!\tPUNCT\t.\t_\t6\tpunct\t6:punct\t_\n\n'.lstrip()
TRAIN_DATA_2 = "\n# sent_id = 11\n# text = It's all hers!\n# previous = Which person owns this?\n# comment = predeterminer modifier\n1\tIt\tit\tPRON\tPRP\tNumber=Sing|Person=3|PronType=Prs\t4\tnsubj\t_\tSpaceAfter=No\n2\t's\tbe\tAUX\tVBZ\tMood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin\t4\tcop\t_\t_\n3\tall\tall\tDET\tDT\tCase=Nom\t4\tdet:predet\t_\t_\n4\thers\thers\tPRON\tPRP\tGender=Fem|Number=Sing|Person=3|Poss=Yes|PronType=Prs\t0\troot\t_\tSpaceAfter=No\n5\t!\t!\tPUNCT\t.\t_\t4\tpunct\t_\t_\n\n".lstrip()
TRAIN_DATA_NO_UPOS = "\n# sent_id = 11\n# text = It's all hers!\n# previous = Which person owns this?\n# comment = predeterminer modifier\n1\tIt\tit\t_\tPRP\tNumber=Sing|Person=3|PronType=Prs\t4\tnsubj\t_\tSpaceAfter=No\n2\t's\tbe\t_\tVBZ\tMood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin\t4\tcop\t_\t_\n3\tall\tall\t_\tDT\tCase=Nom\t4\tdet:predet\t_\t_\n4\thers\thers\t_\tPRP\tGender=Fem|Number=Sing|Person=3|Poss=Yes|PronType=Prs\t0\troot\t_\tSpaceAfter=No\n5\t!\t!\t_\t.\t_\t4\tpunct\t_\t_\n\n".lstrip()
TRAIN_DATA_NO_XPOS = "\n# sent_id = 11\n# text = It's all hers!\n# previous = Which person owns this?\n# comment = predeterminer modifier\n1\tIt\tit\tPRON\t_\tNumber=Sing|Person=3|PronType=Prs\t4\tnsubj\t_\tSpaceAfter=No\n2\t's\tbe\tAUX\t_\tMood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin\t4\tcop\t_\t_\n3\tall\tall\tDET\t_\tCase=Nom\t4\tdet:predet\t_\t_\n4\thers\thers\tPRON\t_\tGender=Fem|Number=Sing|Person=3|Poss=Yes|PronType=Prs\t0\troot\t_\tSpaceAfter=No\n5\t!\t!\tPUNCT\t_\t_\t4\tpunct\t_\t_\n\n".lstrip()
TRAIN_DATA_NO_FEATS = "\n# sent_id = 11\n# text = It's all hers!\n# previous = Which person owns this?\n# comment = predeterminer modifier\n1\tIt\tit\tPRON\tPRP\t_\t4\tnsubj\t_\tSpaceAfter=No\n2\t's\tbe\tAUX\tVBZ\t_\t4\tcop\t_\t_\n3\tall\tall\tDET\tDT\t_\t4\tdet:predet\t_\t_\n4\thers\thers\tPRON\tPRP\t_\t0\troot\t_\tSpaceAfter=No\n5\t!\t!\tPUNCT\t.\t_\t4\tpunct\t_\t_\n\n".lstrip()
DEV_DATA = '\n1\tFrom\tfrom\tADP\tIN\t_\t3\tcase\t3:case\t_\n2\tthe\tthe\tDET\tDT\tDefinite=Def|PronType=Art\t3\tdet\t3:det\t_\n3\tAP\tAP\tPROPN\tNNP\tNumber=Sing\t4\tobl\t4:obl:from\t_\n4\tcomes\tcome\tVERB\tVBZ\tMood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin\t0\troot\t0:root\t_\n5\tthis\tthis\tDET\tDT\tNumber=Sing|PronType=Dem\t6\tdet\t6:det\t_\n6\tstory\tstory\tNOUN\tNN\tNumber=Sing\t4\tnsubj\t4:nsubj\t_\n7\t:\t:\tPUNCT\t:\t_\t4\tpunct\t4:punct\t_\n\n'.lstrip()

class TestTagger:

    @pytest.fixture(scope='class')
    def wordvec_pretrain_file(self):
        if False:
            i = 10
            return i + 15
        return f'{TEST_WORKING_DIR}/in/tiny_emb.pt'

    @pytest.fixture(scope='class')
    def charlm_args(self):
        if False:
            while True:
                i = 10
        charlm = choose_pos_charlm('en', 'test', 'default')
        charlm_args = build_charlm_args('en', charlm, model_dir=TEST_MODELS_DIR)
        return charlm_args

    def run_training(self, tmp_path, wordvec_pretrain_file, train_text, dev_text, augment_nopunct=False, extra_args=None):
        if False:
            return 10
        '\n        Run the training for a few iterations, load & return the model\n        '
        dev_file = str(tmp_path / 'dev.conllu')
        pred_file = str(tmp_path / 'pred.conllu')
        save_name = 'test_tagger.pt'
        save_file = str(tmp_path / save_name)
        if isinstance(train_text, str):
            train_text = [train_text]
        train_files = []
        for (idx, train_blob) in enumerate(train_text):
            train_file = str(tmp_path / ('train_%d.conllu' % idx))
            with open(train_file, 'w', encoding='utf-8') as fout:
                fout.write(train_blob)
            train_files.append(train_file)
        train_file = ';'.join(train_files)
        with open(dev_file, 'w', encoding='utf-8') as fout:
            fout.write(dev_text)
        args = ['--wordvec_pretrain_file', wordvec_pretrain_file, '--train_file', train_file, '--eval_file', dev_file, '--output_file', pred_file, '--log_step', '10', '--eval_interval', '20', '--max_steps', '100', '--shorthand', 'en_test', '--save_dir', str(tmp_path), '--save_name', save_name, '--lang', 'en']
        if not augment_nopunct:
            args.extend(['--augment_nopunct', '0.0'])
        if extra_args is not None:
            args = args + extra_args
        tagger.main(args)
        assert os.path.exists(save_file)
        pt = pretrain.Pretrain(wordvec_pretrain_file)
        saved_model = Trainer(pretrain=pt, model_file=save_file)
        return saved_model

    def test_train(self, tmp_path, wordvec_pretrain_file, augment_nopunct=True):
        if False:
            return 10
        "\n        Simple test of a few 'epochs' of tagger training\n        "
        self.run_training(tmp_path, wordvec_pretrain_file, TRAIN_DATA, DEV_DATA)

    def test_vocab_cutoff(self, tmp_path, wordvec_pretrain_file):
        if False:
            return 10
        '\n        Test that the vocab cutoff leaves words we expect in the vocab, but not rare words\n        '
        trainer = self.run_training(tmp_path, wordvec_pretrain_file, TRAIN_DATA, DEV_DATA, extra_args=['--word_cutoff', '3'])
        word_vocab = trainer.vocab['word']
        assert 'of' in word_vocab
        assert 'officials' in TRAIN_DATA
        assert 'officials' not in word_vocab

    def test_multiple_files(self, tmp_path, wordvec_pretrain_file):
        if False:
            return 10
        '\n        Test that multiple train files works\n\n        Checks for evidence of it working by looking for words from the second file in the vocab\n        '
        trainer = self.run_training(tmp_path, wordvec_pretrain_file, [TRAIN_DATA, TRAIN_DATA_2 * 3], DEV_DATA, extra_args=['--word_cutoff', '3'])
        word_vocab = trainer.vocab['word']
        assert 'of' in word_vocab
        assert 'officials' in TRAIN_DATA
        assert 'officials' not in word_vocab
        assert '\thers\t' not in TRAIN_DATA
        assert '\thers\t' in TRAIN_DATA_2
        assert 'hers' in word_vocab

    def test_train_charlm(self, tmp_path, wordvec_pretrain_file, charlm_args):
        if False:
            print('Hello World!')
        trainer = self.run_training(tmp_path, wordvec_pretrain_file, TRAIN_DATA, DEV_DATA, extra_args=charlm_args)

    def test_train_charlm_projection(self, tmp_path, wordvec_pretrain_file, charlm_args):
        if False:
            for i in range(10):
                print('nop')
        extra_args = charlm_args + ['--charlm_transform_dim', '100']
        trainer = self.run_training(tmp_path, wordvec_pretrain_file, TRAIN_DATA, DEV_DATA, extra_args=extra_args)

    def test_missing_column(self, tmp_path, wordvec_pretrain_file):
        if False:
            while True:
                i = 10
        '\n        Test that using train files with missing columns works\n\n        TODO: we should find some evidence that it is successfully training the upos & xpos\n        '
        trainer = self.run_training(tmp_path, wordvec_pretrain_file, [TRAIN_DATA_NO_UPOS, TRAIN_DATA_NO_XPOS, TRAIN_DATA_NO_FEATS], DEV_DATA)

    def test_with_bert(self, tmp_path, wordvec_pretrain_file):
        if False:
            i = 10
            return i + 15
        self.run_training(tmp_path, wordvec_pretrain_file, TRAIN_DATA, DEV_DATA, extra_args=['--bert_model', 'hf-internal-testing/tiny-bert'])

    def test_with_bert_nlayers(self, tmp_path, wordvec_pretrain_file):
        if False:
            i = 10
            return i + 15
        self.run_training(tmp_path, wordvec_pretrain_file, TRAIN_DATA, DEV_DATA, extra_args=['--bert_model', 'hf-internal-testing/tiny-bert', '--bert_hidden_layers', '2'])