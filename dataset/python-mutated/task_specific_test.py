import copy
import keras_tuner
import autokeras as ak
from autokeras.tuners import task_specific

def test_img_clf_init_hp0_equals_hp_of_a_model(tmp_path):
    if False:
        i = 10
        return i + 15
    clf = ak.ImageClassifier(directory=tmp_path)
    clf.inputs[0].shape = (32, 32, 3)
    clf.outputs[0].in_blocks[0].shape = (10,)
    init_hp = task_specific.IMAGE_CLASSIFIER[0]
    hp = keras_tuner.HyperParameters()
    hp.values = copy.copy(init_hp)
    clf.tuner.hypermodel.build(hp)
    assert set(init_hp.keys()) == set(hp._hps.keys())

def test_img_clf_init_hp1_equals_hp_of_a_model(tmp_path):
    if False:
        print('Hello World!')
    clf = ak.ImageClassifier(directory=tmp_path)
    clf.inputs[0].shape = (32, 32, 3)
    clf.outputs[0].in_blocks[0].shape = (10,)
    init_hp = task_specific.IMAGE_CLASSIFIER[1]
    hp = keras_tuner.HyperParameters()
    hp.values = copy.copy(init_hp)
    clf.tuner.hypermodel.build(hp)
    assert set(init_hp.keys()) == set(hp._hps.keys())

def test_img_clf_init_hp2_equals_hp_of_a_model(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    clf = ak.ImageClassifier(directory=tmp_path)
    clf.inputs[0].shape = (32, 32, 3)
    clf.outputs[0].in_blocks[0].shape = (10,)
    init_hp = task_specific.IMAGE_CLASSIFIER[2]
    hp = keras_tuner.HyperParameters()
    hp.values = copy.copy(init_hp)
    clf.tuner.hypermodel.build(hp)
    assert set(init_hp.keys()) == set(hp._hps.keys())

def test_txt_clf_init_hp2_equals_hp_of_a_model(tmp_path):
    if False:
        print('Hello World!')
    clf = ak.TextClassifier(directory=tmp_path)
    clf.inputs[0].shape = (1,)
    clf.inputs[0].batch_size = 6
    clf.inputs[0].num_samples = 1000
    clf.outputs[0].in_blocks[0].shape = (10,)
    clf.tuner.hypermodel.epochs = 1000
    clf.tuner.hypermodel.num_samples = 20000
    init_hp = task_specific.TEXT_CLASSIFIER[2]
    hp = keras_tuner.HyperParameters()
    hp.values = copy.copy(init_hp)
    clf.tuner.hypermodel.build(hp)
    assert set(init_hp.keys()) == set(hp._hps.keys())

def test_txt_clf_init_hp1_equals_hp_of_a_model(tmp_path):
    if False:
        while True:
            i = 10
    clf = ak.TextClassifier(directory=tmp_path)
    clf.inputs[0].shape = (1,)
    clf.outputs[0].in_blocks[0].shape = (10,)
    init_hp = task_specific.TEXT_CLASSIFIER[1]
    hp = keras_tuner.HyperParameters()
    hp.values = copy.copy(init_hp)
    clf.tuner.hypermodel.build(hp)
    assert set(init_hp.keys()) == set(hp._hps.keys())

def test_txt_clf_init_hp0_equals_hp_of_a_model(tmp_path):
    if False:
        print('Hello World!')
    clf = ak.TextClassifier(directory=tmp_path)
    clf.inputs[0].shape = (1,)
    clf.outputs[0].in_blocks[0].shape = (10,)
    init_hp = task_specific.TEXT_CLASSIFIER[0]
    hp = keras_tuner.HyperParameters()
    hp.values = copy.copy(init_hp)
    clf.tuner.hypermodel.build(hp)
    assert set(init_hp.keys()) == set(hp._hps.keys())

def test_sd_clf_init_hp0_equals_hp_of_a_model(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    clf = ak.StructuredDataClassifier(directory=tmp_path, column_names=['a', 'b'], column_types={'a': 'numerical', 'b': 'numerical'})
    clf.inputs[0].shape = (2,)
    clf.outputs[0].in_blocks[0].shape = (10,)
    init_hp = task_specific.STRUCTURED_DATA_CLASSIFIER[0]
    hp = keras_tuner.HyperParameters()
    hp.values = copy.copy(init_hp)
    clf.tuner.hypermodel.build(hp)
    assert set(init_hp.keys()) == set(hp._hps.keys())

def test_sd_reg_init_hp0_equals_hp_of_a_model(tmp_path):
    if False:
        print('Hello World!')
    clf = ak.StructuredDataRegressor(directory=tmp_path, column_names=['a', 'b'], column_types={'a': 'numerical', 'b': 'numerical'})
    clf.inputs[0].shape = (2,)
    clf.outputs[0].in_blocks[0].shape = (10,)
    init_hp = task_specific.STRUCTURED_DATA_REGRESSOR[0]
    hp = keras_tuner.HyperParameters()
    hp.values = copy.copy(init_hp)
    clf.tuner.hypermodel.build(hp)
    assert set(init_hp.keys()) == set(hp._hps.keys())