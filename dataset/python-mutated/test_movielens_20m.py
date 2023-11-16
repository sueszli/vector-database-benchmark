import os
import shutil
import pytest
from catalyst.settings import SETTINGS
from tests import MOVIELENS20M_ROOT
if SETTINGS.ml_required and SETTINGS.is_torch_1_7_0:
    from catalyst.contrib.datasets import MovieLens20M
minversion = pytest.mark.skipif(not SETTINGS.is_torch_1_7_0, reason='No catalyst[ml] required or torch version ')

def setup_module():
    if False:
        while True:
            i = 10
    '\n    Remove the temp folder if exists\n    '
    data_path = MOVIELENS20M_ROOT
    try:
        shutil.rmtree(data_path)
    except Exception as e:
        print('Error! Code: {c}, Message, {m}'.format(c=type(e).__name__, m=str(e)))

@minversion
@pytest.mark.skipif(not SETTINGS.ml_required, reason='No catalyst[ml] required')
def test_download_split_by_user():
    if False:
        i = 10
        return i + 15
    '\n    Test movielense download\n    '
    MovieLens20M(MOVIELENS20M_ROOT, download=True, sample=True)
    filename = 'ml-20m'
    assert os.path.isdir(MOVIELENS20M_ROOT) is True
    assert os.path.isdir(f'{MOVIELENS20M_ROOT}/MovieLens20M') is True
    assert os.path.isdir(f'{MOVIELENS20M_ROOT}/MovieLens20M/raw') is True
    assert os.path.isdir(f'{MOVIELENS20M_ROOT}/MovieLens20M/processed') is True
    assert os.path.isfile(f'{MOVIELENS20M_ROOT}/MovieLens20M/raw/{filename}/genome-scores.csv') is True
    assert os.path.getsize(f'{MOVIELENS20M_ROOT}/MovieLens20M/raw/{filename}/genome-scores.csv') > 0

@minversion
@pytest.mark.skipif(not SETTINGS.ml_required, reason='No catalyst[ml] required')
def test_download_split_by_ts():
    if False:
        i = 10
        return i + 15
    '\n    Test movielense download\n    '
    MovieLens20M(MOVIELENS20M_ROOT, download=True, split='ts', sample=True)
    filename = 'ml-20m'
    assert os.path.isdir(MOVIELENS20M_ROOT) is True
    assert os.path.isdir(f'{MOVIELENS20M_ROOT}/MovieLens20M') is True
    assert os.path.isdir(f'{MOVIELENS20M_ROOT}/MovieLens20M/raw') is True
    assert os.path.isdir(f'{MOVIELENS20M_ROOT}/MovieLens20M/processed') is True
    assert os.path.isfile(f'{MOVIELENS20M_ROOT}/MovieLens20M/raw/{filename}/genome-scores.csv') is True
    assert os.path.getsize(f'{MOVIELENS20M_ROOT}/MovieLens20M/raw/{filename}/genome-scores.csv') > 0

@minversion
@pytest.mark.skipif(not SETTINGS.ml_required, reason='No catalyst[ml] required')
def test_minimal_ranking():
    if False:
        for i in range(10):
            print('nop')
    '\n    Tets retrieveing the minimal ranking\n    '
    movielens_20m_min_two = MovieLens20M(MOVIELENS20M_ROOT, download=True, min_rating=2.0, sample=True, n_rows=1000000)
    assert 1 not in movielens_20m_min_two[1]._values().unique()
    assert 1 not in movielens_20m_min_two[3]._values().unique()
    assert 2 in movielens_20m_min_two[1]._values().unique() or 3 in movielens_20m_min_two[1]._values().unique() or 4 in movielens_20m_min_two[1]._values().unique() or (5 in movielens_20m_min_two[1]._values().unique()) or (len(movielens_20m_min_two[1]._values().unique()) == 0)
    assert 2 in movielens_20m_min_two[7]._values().unique() or 3 in movielens_20m_min_two[1]._values().unique() or 4 in movielens_20m_min_two[7]._values().unique() or (5 in movielens_20m_min_two[7]._values().unique()) or (len(movielens_20m_min_two[1]._values().unique()) == 0)
    assert 3 in movielens_20m_min_two[3]._values().unique() or 4 in movielens_20m_min_two[3]._values().unique() or 5 in movielens_20m_min_two[3]._values().unique() or (len(movielens_20m_min_two[1]._values().unique()) == 0)

@minversion
@pytest.mark.skipif(not SETTINGS.ml_required, reason='No catalyst[ml] required')
def test_users_per_item_filtering():
    if False:
        print('Hello World!')
    '\n    Tets retrieveing the minimal ranking\n    '
    min_users_per_item = 2.0
    movielens_20m_min_users = MovieLens20M(MOVIELENS20M_ROOT, download=True, min_users_per_item=min_users_per_item, sample=True, n_rows=1000000)
    assert (movielens_20m_min_users.users_activity['user_cnt'] >= min_users_per_item).any()

@minversion
@pytest.mark.skipif(not SETTINGS.ml_required, reason='No catalyst[ml] required')
def test_items_per_user_filtering():
    if False:
        return 10
    '\n    Tets retrieveing the minimal ranking\n    '
    min_items_per_user = 2.0
    min_users_per_item = 1.0
    movielens_20m_min_users = MovieLens20M(MOVIELENS20M_ROOT, download=True, min_items_per_user=min_items_per_user, min_users_per_item=min_users_per_item, sample=True, n_rows=1000000)
    assert (movielens_20m_min_users.items_activity['item_cnt'] >= min_items_per_user).any()

def teardown_module():
    if False:
        for i in range(10):
            print('nop')
    '\n    Remove tempoary files after test execution\n    '
    data_path = MOVIELENS20M_ROOT
    try:
        shutil.rmtree(data_path)
    except Exception as e:
        print('Error! Code: {c}, Message, {m}'.format(c=type(e).__name__, m=str(e)))