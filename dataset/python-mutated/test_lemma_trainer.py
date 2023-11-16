"""
Test a couple basic functions - load & save an existing model
"""
import pytest
import glob
import os
import tempfile
import torch
from stanza.models import lemmatizer
from stanza.models.lemma import trainer
from stanza.tests import *
from stanza.utils.training.common import choose_lemma_charlm, build_charlm_args
pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

@pytest.fixture(scope='module')
def english_model():
    if False:
        return 10
    models_path = os.path.join(TEST_MODELS_DIR, 'en', 'lemma', '*')
    models = glob.glob(models_path)
    assert len(models) >= 1
    model_file = models[0]
    return trainer.Trainer(model_file=model_file)

def test_load_model(english_model):
    if False:
        return 10
    '\n    Does nothing, just tests that loading works\n    '

def test_save_load_model(english_model):
    if False:
        print('Hello World!')
    '\n    Load, save, and load again\n    '
    with tempfile.TemporaryDirectory() as tempdir:
        save_file = os.path.join(tempdir, 'resaved', 'lemma.pt')
        english_model.save(save_file)
        reloaded = trainer.Trainer(model_file=save_file)
TRAIN_DATA = '\n# sent_id = weblog-juancole.com_juancole_20051126063000_ENG_20051126_063000-0003\n# text = DPA: Iraqi authorities announced that they had busted up 3 terrorist cells operating in Baghdad.\n1\tDPA\tDPA\tPROPN\tNNP\tNumber=Sing\t0\troot\t0:root\tSpaceAfter=No\n2\t:\t:\tPUNCT\t:\t_\t1\tpunct\t1:punct\t_\n3\tIraqi\tIraqi\tADJ\tJJ\tDegree=Pos\t4\tamod\t4:amod\t_\n4\tauthorities\tauthority\tNOUN\tNNS\tNumber=Plur\t5\tnsubj\t5:nsubj\t_\n5\tannounced\tannounce\tVERB\tVBD\tMood=Ind|Number=Plur|Person=3|Tense=Past|VerbForm=Fin\t1\tparataxis\t1:parataxis\t_\n6\tthat\tthat\tSCONJ\tIN\t_\t9\tmark\t9:mark\t_\n7\tthey\tthey\tPRON\tPRP\tCase=Nom|Number=Plur|Person=3|PronType=Prs\t9\tnsubj\t9:nsubj\t_\n8\thad\thave\tAUX\tVBD\tMood=Ind|Number=Plur|Person=3|Tense=Past|VerbForm=Fin\t9\taux\t9:aux\t_\n9\tbusted\tbust\tVERB\tVBN\tTense=Past|VerbForm=Part\t5\tccomp\t5:ccomp\t_\n10\tup\tup\tADP\tRP\t_\t9\tcompound:prt\t9:compound:prt\t_\n11\t3\t3\tNUM\tCD\tNumForm=Digit|NumType=Card\t13\tnummod\t13:nummod\t_\n12\tterrorist\tterrorist\tADJ\tJJ\tDegree=Pos\t13\tamod\t13:amod\t_\n13\tcells\tcell\tNOUN\tNNS\tNumber=Plur\t9\tobj\t9:obj\t_\n14\toperating\toperate\tVERB\tVBG\tVerbForm=Ger\t13\tacl\t13:acl\t_\n15\tin\tin\tADP\tIN\t_\t16\tcase\t16:case\t_\n16\tBaghdad\tBaghdad\tPROPN\tNNP\tNumber=Sing\t14\tobl\t14:obl:in\tSpaceAfter=No\n17\t.\t.\tPUNCT\t.\t_\t1\tpunct\t1:punct\t_\n\n# sent_id = weblog-juancole.com_juancole_20051126063000_ENG_20051126_063000-0004\n# text = Two of them were being run by 2 officials of the Ministry of the Interior!\n1\tTwo\ttwo\tNUM\tCD\tNumForm=Word|NumType=Card\t6\tnsubj:pass\t6:nsubj:pass\t_\n2\tof\tof\tADP\tIN\t_\t3\tcase\t3:case\t_\n3\tthem\tthey\tPRON\tPRP\tCase=Acc|Number=Plur|Person=3|PronType=Prs\t1\tnmod\t1:nmod:of\t_\n4\twere\tbe\tAUX\tVBD\tMood=Ind|Number=Plur|Person=3|Tense=Past|VerbForm=Fin\t6\taux\t6:aux\t_\n5\tbeing\tbe\tAUX\tVBG\tVerbForm=Ger\t6\taux:pass\t6:aux:pass\t_\n6\trun\trun\tVERB\tVBN\tTense=Past|VerbForm=Part|Voice=Pass\t0\troot\t0:root\t_\n7\tby\tby\tADP\tIN\t_\t9\tcase\t9:case\t_\n8\t2\t2\tNUM\tCD\tNumForm=Digit|NumType=Card\t9\tnummod\t9:nummod\t_\n9\tofficials\tofficial\tNOUN\tNNS\tNumber=Plur\t6\tobl\t6:obl:by\t_\n10\tof\tof\tADP\tIN\t_\t12\tcase\t12:case\t_\n11\tthe\tthe\tDET\tDT\tDefinite=Def|PronType=Art\t12\tdet\t12:det\t_\n12\tMinistry\tMinistry\tPROPN\tNNP\tNumber=Sing\t9\tnmod\t9:nmod:of\t_\n13\tof\tof\tADP\tIN\t_\t15\tcase\t15:case\t_\n14\tthe\tthe\tDET\tDT\tDefinite=Def|PronType=Art\t15\tdet\t15:det\t_\n15\tInterior\tInterior\tPROPN\tNNP\tNumber=Sing\t12\tnmod\t12:nmod:of\tSpaceAfter=No\n16\t!\t!\tPUNCT\t.\t_\t6\tpunct\t6:punct\t_\n\n'.lstrip()
DEV_DATA = '\n1\tFrom\tfrom\tADP\tIN\t_\t3\tcase\t3:case\t_\n2\tthe\tthe\tDET\tDT\tDefinite=Def|PronType=Art\t3\tdet\t3:det\t_\n3\tAP\tAP\tPROPN\tNNP\tNumber=Sing\t4\tobl\t4:obl:from\t_\n4\tcomes\tcome\tVERB\tVBZ\tMood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin\t0\troot\t0:root\t_\n5\tthis\tthis\tDET\tDT\tNumber=Sing|PronType=Dem\t6\tdet\t6:det\t_\n6\tstory\tstory\tNOUN\tNN\tNumber=Sing\t4\tnsubj\t4:nsubj\t_\n7\t:\t:\tPUNCT\t:\t_\t4\tpunct\t4:punct\t_\n\n'.lstrip()

class TestLemmatizer:

    @pytest.fixture(scope='class')
    def charlm_args(self):
        if False:
            for i in range(10):
                print('nop')
        charlm = choose_lemma_charlm('en', 'test', 'default')
        charlm_args = build_charlm_args('en', charlm, model_dir=TEST_MODELS_DIR)
        return charlm_args

    def run_training(self, tmp_path, train_text, dev_text, extra_args=None):
        if False:
            print('Hello World!')
        '\n        Run the training for a few iterations, load & return the model\n        '
        pred_file = str(tmp_path / 'pred.conllu')
        save_name = 'test_tagger.pt'
        save_file = str(tmp_path / save_name)
        train_file = str(tmp_path / 'train.conllu')
        with open(train_file, 'w', encoding='utf-8') as fout:
            fout.write(train_text)
        dev_file = str(tmp_path / 'dev.conllu')
        with open(dev_file, 'w', encoding='utf-8') as fout:
            fout.write(dev_text)
        args = ['--train_file', train_file, '--eval_file', dev_file, '--gold_file', dev_file, '--output_file', pred_file, '--num_epoch', '2', '--log_step', '10', '--save_dir', str(tmp_path), '--save_name', save_name, '--shorthand', 'en_test']
        if extra_args is not None:
            args = args + extra_args
        lemmatizer.main(args)
        assert os.path.exists(save_file)
        saved_model = trainer.Trainer(model_file=save_file)
        return saved_model

    def test_basic_train(self, tmp_path):
        if False:
            print('Hello World!')
        "\n        Simple test of a few 'epochs' of lemmatizer training\n        "
        self.run_training(tmp_path, TRAIN_DATA, DEV_DATA)

    def test_charlm_train(self, tmp_path, charlm_args):
        if False:
            while True:
                i = 10
        "\n        Simple test of a few 'epochs' of lemmatizer training\n        "
        saved_model = self.run_training(tmp_path, TRAIN_DATA, DEV_DATA, extra_args=charlm_args)
        args = saved_model.args
        save_name = os.path.join(args['save_dir'], args['save_name'])
        checkpoint = torch.load(save_name, lambda storage, loc: storage)
        assert not any((x.startswith('contextual_embedding') for x in checkpoint['model'].keys()))