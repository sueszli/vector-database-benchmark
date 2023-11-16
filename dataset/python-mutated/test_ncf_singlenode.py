import os
import shutil
import pytest
import numpy as np
import pandas as pd
try:
    from recommenders.models.ncf.ncf_singlenode import NCF
    from recommenders.models.ncf.dataset import Dataset
    from recommenders.utils.constants import DEFAULT_USER_COL, DEFAULT_ITEM_COL, SEED
except ImportError:
    pass
N_NEG = 5
N_NEG_TEST = 10

@pytest.mark.gpu
@pytest.mark.parametrize('model_type, n_users, n_items', [('NeuMF', 1, 1), ('GMF', 10, 10), ('MLP', 4, 8)])
def test_init(model_type, n_users, n_items):
    if False:
        for i in range(10):
            print('nop')
    model = NCF(n_users=n_users, n_items=n_items, model_type=model_type, n_epochs=1, seed=SEED)
    assert model.model_type == model_type.lower()
    assert model.n_users == n_users
    assert model.n_items == n_items
    assert model.embedding_gmf_P.shape == [n_users, model.n_factors]
    assert model.embedding_gmf_Q.shape == [n_items, model.n_factors]
    assert model.embedding_mlp_P.shape == [n_users, model.n_factors]
    assert model.embedding_mlp_Q.shape == [n_items, model.n_factors]

@pytest.mark.gpu
@pytest.mark.parametrize('model_type, n_users, n_items', [('NeuMF', 5, 5), ('GMF', 5, 5), ('MLP', 5, 5)])
def test_regular_save_load(model_type, n_users, n_items):
    if False:
        while True:
            i = 10
    ckpt = '.%s' % model_type
    if os.path.exists(ckpt):
        shutil.rmtree(ckpt)
    model = NCF(n_users=n_users, n_items=n_items, model_type=model_type, n_epochs=1, seed=SEED)
    model.save(ckpt)
    if model.model_type == 'neumf':
        P = model.sess.run(model.embedding_gmf_P)
        Q = model.sess.run(model.embedding_mlp_Q)
    elif model.model_type == 'gmf':
        P = model.sess.run(model.embedding_gmf_P)
        Q = model.sess.run(model.embedding_gmf_Q)
    elif model.model_type == 'mlp':
        P = model.sess.run(model.embedding_mlp_P)
        Q = model.sess.run(model.embedding_mlp_Q)
    del model
    model = NCF(n_users=n_users, n_items=n_items, model_type=model_type, n_epochs=1, seed=SEED)
    if model.model_type == 'neumf':
        model.load(neumf_dir=ckpt)
        P_ = model.sess.run(model.embedding_gmf_P)
        Q_ = model.sess.run(model.embedding_mlp_Q)
    elif model.model_type == 'gmf':
        model.load(gmf_dir=ckpt)
        P_ = model.sess.run(model.embedding_gmf_P)
        Q_ = model.sess.run(model.embedding_gmf_Q)
    elif model.model_type == 'mlp':
        model.load(mlp_dir=ckpt)
        P_ = model.sess.run(model.embedding_mlp_P)
        Q_ = model.sess.run(model.embedding_mlp_Q)
    assert np.array_equal(P, P_)
    assert np.array_equal(Q, Q_)
    if os.path.exists(ckpt):
        shutil.rmtree(ckpt)

@pytest.mark.gpu
@pytest.mark.parametrize('n_users, n_items', [(5, 5), (4, 8)])
def test_neumf_save_load(n_users, n_items):
    if False:
        while True:
            i = 10
    model_type = 'gmf'
    ckpt_gmf = '.%s' % model_type
    if os.path.exists(ckpt_gmf):
        shutil.rmtree(ckpt_gmf)
    model = NCF(n_users=n_users, n_items=n_items, model_type=model_type, n_epochs=1)
    model.save(ckpt_gmf)
    P_gmf = model.sess.run(model.embedding_gmf_P)
    Q_gmf = model.sess.run(model.embedding_gmf_Q)
    del model
    model_type = 'mlp'
    ckpt_mlp = '.%s' % model_type
    if os.path.exists(ckpt_mlp):
        shutil.rmtree(ckpt_mlp)
    model = NCF(n_users=n_users, n_items=n_items, model_type=model_type, n_epochs=1)
    model.save('.%s' % model_type)
    P_mlp = model.sess.run(model.embedding_mlp_P)
    Q_mlp = model.sess.run(model.embedding_mlp_Q)
    del model
    model_type = 'neumf'
    model = NCF(n_users=n_users, n_items=n_items, model_type=model_type, n_epochs=1)
    model.load(gmf_dir=ckpt_gmf, mlp_dir=ckpt_mlp)
    P_gmf_ = model.sess.run(model.embedding_gmf_P)
    Q_gmf_ = model.sess.run(model.embedding_gmf_Q)
    P_mlp_ = model.sess.run(model.embedding_mlp_P)
    Q_mlp_ = model.sess.run(model.embedding_mlp_Q)
    assert np.array_equal(P_gmf, P_gmf_)
    assert np.array_equal(Q_gmf, Q_gmf_)
    assert np.array_equal(P_mlp, P_mlp_)
    assert np.array_equal(Q_mlp, Q_mlp_)
    if os.path.exists(ckpt_gmf):
        shutil.rmtree(ckpt_gmf)
    if os.path.exists(ckpt_mlp):
        shutil.rmtree(ckpt_mlp)

@pytest.mark.gpu
@pytest.mark.parametrize('model_type', ['NeuMF', 'GMF', 'MLP'])
def test_fit(dataset_ncf_files_sorted, model_type):
    if False:
        i = 10
        return i + 15
    (train_path, test_path, _) = dataset_ncf_files_sorted
    data = Dataset(train_file=train_path, test_file=test_path, n_neg=N_NEG, n_neg_test=N_NEG_TEST)
    model = NCF(n_users=data.n_users, n_items=data.n_items, model_type=model_type, n_epochs=1)
    model.fit(data)

@pytest.mark.gpu
@pytest.mark.parametrize('model_type', ['NeuMF', 'GMF', 'MLP'])
def test_predict(dataset_ncf_files_sorted, model_type):
    if False:
        print('Hello World!')
    (train_path, test_path, _) = dataset_ncf_files_sorted
    test = pd.read_csv(test_path)
    data = Dataset(train_file=train_path, test_file=test_path, n_neg=N_NEG, n_neg_test=N_NEG_TEST)
    model = NCF(n_users=data.n_users, n_items=data.n_items, model_type=model_type, n_epochs=1)
    model.fit(data)
    (test_users, test_items) = (list(test[DEFAULT_USER_COL]), list(test[DEFAULT_ITEM_COL]))
    assert type(model.predict(test_users[0], test_items[0])) == float
    res = model.predict(test_users, test_items, is_list=True)
    assert type(res) == list
    assert len(res) == len(test)