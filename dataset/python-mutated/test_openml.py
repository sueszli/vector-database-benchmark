"""Test the openml loader."""
import gzip
import json
import os
import re
from functools import partial
from io import BytesIO
from urllib.error import HTTPError
import numpy as np
import pytest
import scipy.sparse
import sklearn
from sklearn import config_context
from sklearn.datasets import fetch_openml as fetch_openml_orig
from sklearn.datasets._openml import _OPENML_PREFIX, _get_local_path, _open_openml_url, _retry_with_clean_cache
from sklearn.utils import Bunch, check_pandas_support
from sklearn.utils._testing import SkipTest, assert_allclose, assert_array_equal, fails_if_pypy
from sklearn.utils.fixes import _open_binary
OPENML_TEST_DATA_MODULE = 'sklearn.datasets.tests.data.openml'
test_offline = True

class _MockHTTPResponse:

    def __init__(self, data, is_gzip):
        if False:
            return 10
        self.data = data
        self.is_gzip = is_gzip

    def read(self, amt=-1):
        if False:
            return 10
        return self.data.read(amt)

    def close(self):
        if False:
            return 10
        self.data.close()

    def info(self):
        if False:
            print('Hello World!')
        if self.is_gzip:
            return {'Content-Encoding': 'gzip'}
        return {}

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        return iter(self.data)

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            for i in range(10):
                print('nop')
        return False
fetch_openml = partial(fetch_openml_orig, data_home=None)

def _monkey_patch_webbased_functions(context, data_id, gzip_response):
    if False:
        while True:
            i = 10
    url_prefix_data_description = 'https://api.openml.org/api/v1/json/data/'
    url_prefix_data_features = 'https://api.openml.org/api/v1/json/data/features/'
    url_prefix_download_data = 'https://api.openml.org/data/v1/'
    url_prefix_data_list = 'https://api.openml.org/api/v1/json/data/list/'
    path_suffix = '.gz'
    read_fn = gzip.open
    data_module = OPENML_TEST_DATA_MODULE + '.' + f'id_{data_id}'

    def _file_name(url, suffix):
        if False:
            while True:
                i = 10
        output = re.sub('\\W', '-', url[len('https://api.openml.org/'):]) + suffix + path_suffix
        return output.replace('-json-data-list', '-jdl').replace('-json-data-features', '-jdf').replace('-json-data-qualities', '-jdq').replace('-json-data', '-jd').replace('-data_name', '-dn').replace('-download', '-dl').replace('-limit', '-l').replace('-data_version', '-dv').replace('-status', '-s').replace('-deactivated', '-dact').replace('-active', '-act')

    def _mock_urlopen_shared(url, has_gzip_header, expected_prefix, suffix):
        if False:
            return 10
        assert url.startswith(expected_prefix)
        data_file_name = _file_name(url, suffix)
        with _open_binary(data_module, data_file_name) as f:
            if has_gzip_header and gzip_response:
                fp = BytesIO(f.read())
                return _MockHTTPResponse(fp, True)
            else:
                decompressed_f = read_fn(f, 'rb')
                fp = BytesIO(decompressed_f.read())
                return _MockHTTPResponse(fp, False)

    def _mock_urlopen_data_description(url, has_gzip_header):
        if False:
            while True:
                i = 10
        return _mock_urlopen_shared(url=url, has_gzip_header=has_gzip_header, expected_prefix=url_prefix_data_description, suffix='.json')

    def _mock_urlopen_data_features(url, has_gzip_header):
        if False:
            while True:
                i = 10
        return _mock_urlopen_shared(url=url, has_gzip_header=has_gzip_header, expected_prefix=url_prefix_data_features, suffix='.json')

    def _mock_urlopen_download_data(url, has_gzip_header):
        if False:
            print('Hello World!')
        return _mock_urlopen_shared(url=url, has_gzip_header=has_gzip_header, expected_prefix=url_prefix_download_data, suffix='.arff')

    def _mock_urlopen_data_list(url, has_gzip_header):
        if False:
            for i in range(10):
                print('nop')
        assert url.startswith(url_prefix_data_list)
        data_file_name = _file_name(url, '.json')
        with _open_binary(data_module, data_file_name) as f:
            decompressed_f = read_fn(f, 'rb')
            decoded_s = decompressed_f.read().decode('utf-8')
            json_data = json.loads(decoded_s)
        if 'error' in json_data:
            raise HTTPError(url=None, code=412, msg='Simulated mock error', hdrs=None, fp=None)
        with _open_binary(data_module, data_file_name) as f:
            if has_gzip_header:
                fp = BytesIO(f.read())
                return _MockHTTPResponse(fp, True)
            else:
                decompressed_f = read_fn(f, 'rb')
                fp = BytesIO(decompressed_f.read())
                return _MockHTTPResponse(fp, False)

    def _mock_urlopen(request, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        url = request.get_full_url()
        has_gzip_header = request.get_header('Accept-encoding') == 'gzip'
        if url.startswith(url_prefix_data_list):
            return _mock_urlopen_data_list(url, has_gzip_header)
        elif url.startswith(url_prefix_data_features):
            return _mock_urlopen_data_features(url, has_gzip_header)
        elif url.startswith(url_prefix_download_data):
            return _mock_urlopen_download_data(url, has_gzip_header)
        elif url.startswith(url_prefix_data_description):
            return _mock_urlopen_data_description(url, has_gzip_header)
        else:
            raise ValueError('Unknown mocking URL pattern: %s' % url)
    if test_offline:
        context.setattr(sklearn.datasets._openml, 'urlopen', _mock_urlopen)

@fails_if_pypy
@pytest.mark.parametrize('data_id, dataset_params, n_samples, n_features, n_targets', [(61, {'data_id': 61}, 150, 4, 1), (61, {'name': 'iris', 'version': 1}, 150, 4, 1), (2, {'data_id': 2}, 11, 38, 1), (2, {'name': 'anneal', 'version': 1}, 11, 38, 1), (561, {'data_id': 561}, 209, 7, 1), (561, {'name': 'cpu', 'version': 1}, 209, 7, 1), (40589, {'data_id': 40589}, 13, 72, 6), (1119, {'data_id': 1119}, 10, 14, 1), (1119, {'name': 'adult-census'}, 10, 14, 1), (40966, {'data_id': 40966}, 7, 77, 1), (40966, {'name': 'MiceProtein'}, 7, 77, 1), (40945, {'data_id': 40945}, 1309, 13, 1)])
@pytest.mark.parametrize('parser', ['liac-arff', 'pandas'])
@pytest.mark.parametrize('gzip_response', [True, False])
def test_fetch_openml_as_frame_true(monkeypatch, data_id, dataset_params, n_samples, n_features, n_targets, parser, gzip_response):
    if False:
        while True:
            i = 10
    'Check the behaviour of `fetch_openml` with `as_frame=True`.\n\n    Fetch by ID and/or name (depending if the file was previously cached).\n    '
    pd = pytest.importorskip('pandas')
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response=gzip_response)
    bunch = fetch_openml(as_frame=True, cache=False, parser=parser, **dataset_params)
    assert int(bunch.details['id']) == data_id
    assert isinstance(bunch, Bunch)
    assert isinstance(bunch.frame, pd.DataFrame)
    assert bunch.frame.shape == (n_samples, n_features + n_targets)
    assert isinstance(bunch.data, pd.DataFrame)
    assert bunch.data.shape == (n_samples, n_features)
    if n_targets == 1:
        assert isinstance(bunch.target, pd.Series)
        assert bunch.target.shape == (n_samples,)
    else:
        assert isinstance(bunch.target, pd.DataFrame)
        assert bunch.target.shape == (n_samples, n_targets)
    assert bunch.categories is None

@fails_if_pypy
@pytest.mark.parametrize('data_id, dataset_params, n_samples, n_features, n_targets', [(61, {'data_id': 61}, 150, 4, 1), (61, {'name': 'iris', 'version': 1}, 150, 4, 1), (2, {'data_id': 2}, 11, 38, 1), (2, {'name': 'anneal', 'version': 1}, 11, 38, 1), (561, {'data_id': 561}, 209, 7, 1), (561, {'name': 'cpu', 'version': 1}, 209, 7, 1), (40589, {'data_id': 40589}, 13, 72, 6), (1119, {'data_id': 1119}, 10, 14, 1), (1119, {'name': 'adult-census'}, 10, 14, 1), (40966, {'data_id': 40966}, 7, 77, 1), (40966, {'name': 'MiceProtein'}, 7, 77, 1)])
@pytest.mark.parametrize('parser', ['liac-arff', 'pandas'])
def test_fetch_openml_as_frame_false(monkeypatch, data_id, dataset_params, n_samples, n_features, n_targets, parser):
    if False:
        i = 10
        return i + 15
    'Check the behaviour of `fetch_openml` with `as_frame=False`.\n\n    Fetch both by ID and/or name + version.\n    '
    pytest.importorskip('pandas')
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response=True)
    bunch = fetch_openml(as_frame=False, cache=False, parser=parser, **dataset_params)
    assert int(bunch.details['id']) == data_id
    assert isinstance(bunch, Bunch)
    assert bunch.frame is None
    assert isinstance(bunch.data, np.ndarray)
    assert bunch.data.shape == (n_samples, n_features)
    assert isinstance(bunch.target, np.ndarray)
    if n_targets == 1:
        assert bunch.target.shape == (n_samples,)
    else:
        assert bunch.target.shape == (n_samples, n_targets)
    assert isinstance(bunch.categories, dict)

@fails_if_pypy
@pytest.mark.parametrize('data_id', [61, 1119, 40945])
def test_fetch_openml_consistency_parser(monkeypatch, data_id):
    if False:
        while True:
            i = 10
    'Check the consistency of the LIAC-ARFF and pandas parsers.'
    pd = pytest.importorskip('pandas')
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response=True)
    bunch_liac = fetch_openml(data_id=data_id, as_frame=True, cache=False, parser='liac-arff')
    bunch_pandas = fetch_openml(data_id=data_id, as_frame=True, cache=False, parser='pandas')
    (data_liac, data_pandas) = (bunch_liac.data, bunch_pandas.data)

    def convert_numerical_dtypes(series):
        if False:
            print('Hello World!')
        pandas_series = data_pandas[series.name]
        if pd.api.types.is_numeric_dtype(pandas_series):
            return series.astype(pandas_series.dtype)
        else:
            return series
    data_liac_with_fixed_dtypes = data_liac.apply(convert_numerical_dtypes)
    pd.testing.assert_frame_equal(data_liac_with_fixed_dtypes, data_pandas)
    (frame_liac, frame_pandas) = (bunch_liac.frame, bunch_pandas.frame)
    pd.testing.assert_frame_equal(frame_pandas[bunch_pandas.feature_names], data_pandas)

    def convert_numerical_and_categorical_dtypes(series):
        if False:
            print('Hello World!')
        pandas_series = frame_pandas[series.name]
        if pd.api.types.is_numeric_dtype(pandas_series):
            return series.astype(pandas_series.dtype)
        elif isinstance(pandas_series.dtype, pd.CategoricalDtype):
            return series.cat.rename_categories(pandas_series.cat.categories)
        else:
            return series
    frame_liac_with_fixed_dtypes = frame_liac.apply(convert_numerical_and_categorical_dtypes)
    pd.testing.assert_frame_equal(frame_liac_with_fixed_dtypes, frame_pandas)

@fails_if_pypy
@pytest.mark.parametrize('parser', ['liac-arff', 'pandas'])
def test_fetch_openml_equivalence_array_dataframe(monkeypatch, parser):
    if False:
        i = 10
        return i + 15
    'Check the equivalence of the dataset when using `as_frame=False` and\n    `as_frame=True`.\n    '
    pytest.importorskip('pandas')
    data_id = 61
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response=True)
    bunch_as_frame_true = fetch_openml(data_id=data_id, as_frame=True, cache=False, parser=parser)
    bunch_as_frame_false = fetch_openml(data_id=data_id, as_frame=False, cache=False, parser=parser)
    assert_allclose(bunch_as_frame_false.data, bunch_as_frame_true.data)
    assert_array_equal(bunch_as_frame_false.target, bunch_as_frame_true.target)

@fails_if_pypy
@pytest.mark.parametrize('parser', ['liac-arff', 'pandas'])
def test_fetch_openml_iris_pandas(monkeypatch, parser):
    if False:
        i = 10
        return i + 15
    'Check fetching on a numerical only dataset with string labels.'
    pd = pytest.importorskip('pandas')
    CategoricalDtype = pd.api.types.CategoricalDtype
    data_id = 61
    data_shape = (150, 4)
    target_shape = (150,)
    frame_shape = (150, 5)
    target_dtype = CategoricalDtype(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
    data_dtypes = [np.float64] * 4
    data_names = ['sepallength', 'sepalwidth', 'petallength', 'petalwidth']
    target_name = 'class'
    _monkey_patch_webbased_functions(monkeypatch, data_id, True)
    bunch = fetch_openml(data_id=data_id, as_frame=True, cache=False, parser=parser)
    data = bunch.data
    target = bunch.target
    frame = bunch.frame
    assert isinstance(data, pd.DataFrame)
    assert np.all(data.dtypes == data_dtypes)
    assert data.shape == data_shape
    assert np.all(data.columns == data_names)
    assert np.all(bunch.feature_names == data_names)
    assert bunch.target_names == [target_name]
    assert isinstance(target, pd.Series)
    assert target.dtype == target_dtype
    assert target.shape == target_shape
    assert target.name == target_name
    assert target.index.is_unique
    assert isinstance(frame, pd.DataFrame)
    assert frame.shape == frame_shape
    assert np.all(frame.dtypes == data_dtypes + [target_dtype])
    assert frame.index.is_unique

@fails_if_pypy
@pytest.mark.parametrize('parser', ['liac-arff', 'pandas'])
@pytest.mark.parametrize('target_column', ['petalwidth', ['petalwidth', 'petallength']])
def test_fetch_openml_forcing_targets(monkeypatch, parser, target_column):
    if False:
        print('Hello World!')
    'Check that we can force the target to not be the default target.'
    pd = pytest.importorskip('pandas')
    data_id = 61
    _monkey_patch_webbased_functions(monkeypatch, data_id, True)
    bunch_forcing_target = fetch_openml(data_id=data_id, as_frame=True, cache=False, target_column=target_column, parser=parser)
    bunch_default = fetch_openml(data_id=data_id, as_frame=True, cache=False, parser=parser)
    pd.testing.assert_frame_equal(bunch_forcing_target.frame, bunch_default.frame)
    if isinstance(target_column, list):
        pd.testing.assert_index_equal(bunch_forcing_target.target.columns, pd.Index(target_column))
        assert bunch_forcing_target.data.shape == (150, 3)
    else:
        assert bunch_forcing_target.target.name == target_column
        assert bunch_forcing_target.data.shape == (150, 4)

@fails_if_pypy
@pytest.mark.parametrize('data_id', [61, 2, 561, 40589, 1119])
@pytest.mark.parametrize('parser', ['liac-arff', 'pandas'])
def test_fetch_openml_equivalence_frame_return_X_y(monkeypatch, data_id, parser):
    if False:
        return 10
    'Check the behaviour of `return_X_y=True` when `as_frame=True`.'
    pd = pytest.importorskip('pandas')
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response=True)
    bunch = fetch_openml(data_id=data_id, as_frame=True, cache=False, return_X_y=False, parser=parser)
    (X, y) = fetch_openml(data_id=data_id, as_frame=True, cache=False, return_X_y=True, parser=parser)
    pd.testing.assert_frame_equal(bunch.data, X)
    if isinstance(y, pd.Series):
        pd.testing.assert_series_equal(bunch.target, y)
    else:
        pd.testing.assert_frame_equal(bunch.target, y)

@fails_if_pypy
@pytest.mark.parametrize('data_id', [61, 561, 40589, 1119])
@pytest.mark.parametrize('parser', ['liac-arff', 'pandas'])
def test_fetch_openml_equivalence_array_return_X_y(monkeypatch, data_id, parser):
    if False:
        i = 10
        return i + 15
    'Check the behaviour of `return_X_y=True` when `as_frame=False`.'
    pytest.importorskip('pandas')
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response=True)
    bunch = fetch_openml(data_id=data_id, as_frame=False, cache=False, return_X_y=False, parser=parser)
    (X, y) = fetch_openml(data_id=data_id, as_frame=False, cache=False, return_X_y=True, parser=parser)
    assert_array_equal(bunch.data, X)
    assert_array_equal(bunch.target, y)

@fails_if_pypy
def test_fetch_openml_difference_parsers(monkeypatch):
    if False:
        i = 10
        return i + 15
    'Check the difference between liac-arff and pandas parser.'
    pytest.importorskip('pandas')
    data_id = 1119
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response=True)
    as_frame = False
    bunch_liac_arff = fetch_openml(data_id=data_id, as_frame=as_frame, cache=False, parser='liac-arff')
    bunch_pandas = fetch_openml(data_id=data_id, as_frame=as_frame, cache=False, parser='pandas')
    assert bunch_liac_arff.data.dtype.kind == 'f'
    assert bunch_pandas.data.dtype == 'O'

@pytest.fixture(scope='module')
def datasets_column_names():
    if False:
        print('Hello World!')
    'Returns the columns names for each dataset.'
    return {61: ['sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'class'], 2: ['family', 'product-type', 'steel', 'carbon', 'hardness', 'temper_rolling', 'condition', 'formability', 'strength', 'non-ageing', 'surface-finish', 'surface-quality', 'enamelability', 'bc', 'bf', 'bt', 'bw%2Fme', 'bl', 'm', 'chrom', 'phos', 'cbond', 'marvi', 'exptl', 'ferro', 'corr', 'blue%2Fbright%2Fvarn%2Fclean', 'lustre', 'jurofm', 's', 'p', 'shape', 'thick', 'width', 'len', 'oil', 'bore', 'packing', 'class'], 561: ['vendor', 'MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'class'], 40589: ['Mean_Acc1298_Mean_Mem40_Centroid', 'Mean_Acc1298_Mean_Mem40_Rolloff', 'Mean_Acc1298_Mean_Mem40_Flux', 'Mean_Acc1298_Mean_Mem40_MFCC_0', 'Mean_Acc1298_Mean_Mem40_MFCC_1', 'Mean_Acc1298_Mean_Mem40_MFCC_2', 'Mean_Acc1298_Mean_Mem40_MFCC_3', 'Mean_Acc1298_Mean_Mem40_MFCC_4', 'Mean_Acc1298_Mean_Mem40_MFCC_5', 'Mean_Acc1298_Mean_Mem40_MFCC_6', 'Mean_Acc1298_Mean_Mem40_MFCC_7', 'Mean_Acc1298_Mean_Mem40_MFCC_8', 'Mean_Acc1298_Mean_Mem40_MFCC_9', 'Mean_Acc1298_Mean_Mem40_MFCC_10', 'Mean_Acc1298_Mean_Mem40_MFCC_11', 'Mean_Acc1298_Mean_Mem40_MFCC_12', 'Mean_Acc1298_Std_Mem40_Centroid', 'Mean_Acc1298_Std_Mem40_Rolloff', 'Mean_Acc1298_Std_Mem40_Flux', 'Mean_Acc1298_Std_Mem40_MFCC_0', 'Mean_Acc1298_Std_Mem40_MFCC_1', 'Mean_Acc1298_Std_Mem40_MFCC_2', 'Mean_Acc1298_Std_Mem40_MFCC_3', 'Mean_Acc1298_Std_Mem40_MFCC_4', 'Mean_Acc1298_Std_Mem40_MFCC_5', 'Mean_Acc1298_Std_Mem40_MFCC_6', 'Mean_Acc1298_Std_Mem40_MFCC_7', 'Mean_Acc1298_Std_Mem40_MFCC_8', 'Mean_Acc1298_Std_Mem40_MFCC_9', 'Mean_Acc1298_Std_Mem40_MFCC_10', 'Mean_Acc1298_Std_Mem40_MFCC_11', 'Mean_Acc1298_Std_Mem40_MFCC_12', 'Std_Acc1298_Mean_Mem40_Centroid', 'Std_Acc1298_Mean_Mem40_Rolloff', 'Std_Acc1298_Mean_Mem40_Flux', 'Std_Acc1298_Mean_Mem40_MFCC_0', 'Std_Acc1298_Mean_Mem40_MFCC_1', 'Std_Acc1298_Mean_Mem40_MFCC_2', 'Std_Acc1298_Mean_Mem40_MFCC_3', 'Std_Acc1298_Mean_Mem40_MFCC_4', 'Std_Acc1298_Mean_Mem40_MFCC_5', 'Std_Acc1298_Mean_Mem40_MFCC_6', 'Std_Acc1298_Mean_Mem40_MFCC_7', 'Std_Acc1298_Mean_Mem40_MFCC_8', 'Std_Acc1298_Mean_Mem40_MFCC_9', 'Std_Acc1298_Mean_Mem40_MFCC_10', 'Std_Acc1298_Mean_Mem40_MFCC_11', 'Std_Acc1298_Mean_Mem40_MFCC_12', 'Std_Acc1298_Std_Mem40_Centroid', 'Std_Acc1298_Std_Mem40_Rolloff', 'Std_Acc1298_Std_Mem40_Flux', 'Std_Acc1298_Std_Mem40_MFCC_0', 'Std_Acc1298_Std_Mem40_MFCC_1', 'Std_Acc1298_Std_Mem40_MFCC_2', 'Std_Acc1298_Std_Mem40_MFCC_3', 'Std_Acc1298_Std_Mem40_MFCC_4', 'Std_Acc1298_Std_Mem40_MFCC_5', 'Std_Acc1298_Std_Mem40_MFCC_6', 'Std_Acc1298_Std_Mem40_MFCC_7', 'Std_Acc1298_Std_Mem40_MFCC_8', 'Std_Acc1298_Std_Mem40_MFCC_9', 'Std_Acc1298_Std_Mem40_MFCC_10', 'Std_Acc1298_Std_Mem40_MFCC_11', 'Std_Acc1298_Std_Mem40_MFCC_12', 'BH_LowPeakAmp', 'BH_LowPeakBPM', 'BH_HighPeakAmp', 'BH_HighPeakBPM', 'BH_HighLowRatio', 'BHSUM1', 'BHSUM2', 'BHSUM3', 'amazed.suprised', 'happy.pleased', 'relaxing.calm', 'quiet.still', 'sad.lonely', 'angry.aggresive'], 1119: ['age', 'workclass', 'fnlwgt:', 'education:', 'education-num:', 'marital-status:', 'occupation:', 'relationship:', 'race:', 'sex:', 'capital-gain:', 'capital-loss:', 'hours-per-week:', 'native-country:', 'class'], 40966: ['DYRK1A_N', 'ITSN1_N', 'BDNF_N', 'NR1_N', 'NR2A_N', 'pAKT_N', 'pBRAF_N', 'pCAMKII_N', 'pCREB_N', 'pELK_N', 'pERK_N', 'pJNK_N', 'PKCA_N', 'pMEK_N', 'pNR1_N', 'pNR2A_N', 'pNR2B_N', 'pPKCAB_N', 'pRSK_N', 'AKT_N', 'BRAF_N', 'CAMKII_N', 'CREB_N', 'ELK_N', 'ERK_N', 'GSK3B_N', 'JNK_N', 'MEK_N', 'TRKA_N', 'RSK_N', 'APP_N', 'Bcatenin_N', 'SOD1_N', 'MTOR_N', 'P38_N', 'pMTOR_N', 'DSCR1_N', 'AMPKA_N', 'NR2B_N', 'pNUMB_N', 'RAPTOR_N', 'TIAM1_N', 'pP70S6_N', 'NUMB_N', 'P70S6_N', 'pGSK3B_N', 'pPKCG_N', 'CDK5_N', 'S6_N', 'ADARB1_N', 'AcetylH3K9_N', 'RRP1_N', 'BAX_N', 'ARC_N', 'ERBB4_N', 'nNOS_N', 'Tau_N', 'GFAP_N', 'GluR3_N', 'GluR4_N', 'IL1B_N', 'P3525_N', 'pCASP9_N', 'PSD95_N', 'SNCA_N', 'Ubiquitin_N', 'pGSK3B_Tyr216_N', 'SHH_N', 'BAD_N', 'BCL2_N', 'pS6_N', 'pCFOS_N', 'SYP_N', 'H3AcK18_N', 'EGR1_N', 'H3MeK4_N', 'CaNA_N', 'class'], 40945: ['pclass', 'survived', 'name', 'sex', 'age', 'sibsp', 'parch', 'ticket', 'fare', 'cabin', 'embarked', 'boat', 'body', 'home.dest']}

@pytest.fixture(scope='module')
def datasets_missing_values():
    if False:
        while True:
            i = 10
    return {61: {}, 2: {'family': 11, 'temper_rolling': 9, 'condition': 2, 'formability': 4, 'non-ageing': 10, 'surface-finish': 11, 'enamelability': 11, 'bc': 11, 'bf': 10, 'bt': 11, 'bw%2Fme': 8, 'bl': 9, 'm': 11, 'chrom': 11, 'phos': 11, 'cbond': 10, 'marvi': 11, 'exptl': 11, 'ferro': 11, 'corr': 11, 'blue%2Fbright%2Fvarn%2Fclean': 11, 'lustre': 8, 'jurofm': 11, 's': 11, 'p': 11, 'oil': 10, 'packing': 11}, 561: {}, 40589: {}, 1119: {}, 40966: {'BCL2_N': 7}, 40945: {'age': 263, 'fare': 1, 'cabin': 1014, 'embarked': 2, 'boat': 823, 'body': 1188, 'home.dest': 564}}

@fails_if_pypy
@pytest.mark.parametrize('data_id, parser, expected_n_categories, expected_n_floats, expected_n_ints', [(61, 'liac-arff', 1, 4, 0), (61, 'pandas', 1, 4, 0), (2, 'liac-arff', 33, 6, 0), (2, 'pandas', 33, 2, 4), (561, 'liac-arff', 1, 7, 0), (561, 'pandas', 1, 0, 7), (40589, 'liac-arff', 6, 72, 0), (40589, 'pandas', 6, 69, 3), (1119, 'liac-arff', 9, 6, 0), (1119, 'pandas', 9, 0, 6), (40966, 'liac-arff', 1, 77, 0), (40966, 'pandas', 1, 77, 0), (40945, 'liac-arff', 3, 6, 0), (40945, 'pandas', 3, 3, 3)])
@pytest.mark.parametrize('gzip_response', [True, False])
def test_fetch_openml_types_inference(monkeypatch, data_id, parser, expected_n_categories, expected_n_floats, expected_n_ints, gzip_response, datasets_column_names, datasets_missing_values):
    if False:
        while True:
            i = 10
    'Check that `fetch_openml` infer the right number of categories, integers, and\n    floats.'
    pd = pytest.importorskip('pandas')
    CategoricalDtype = pd.api.types.CategoricalDtype
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response=gzip_response)
    bunch = fetch_openml(data_id=data_id, as_frame=True, cache=False, parser=parser)
    frame = bunch.frame
    n_categories = len([dtype for dtype in frame.dtypes if isinstance(dtype, CategoricalDtype)])
    n_floats = len([dtype for dtype in frame.dtypes if dtype.kind == 'f'])
    n_ints = len([dtype for dtype in frame.dtypes if dtype.kind == 'i'])
    assert n_categories == expected_n_categories
    assert n_floats == expected_n_floats
    assert n_ints == expected_n_ints
    assert frame.columns.tolist() == datasets_column_names[data_id]
    frame_feature_to_n_nan = frame.isna().sum().to_dict()
    for (name, n_missing) in frame_feature_to_n_nan.items():
        expected_missing = datasets_missing_values[data_id].get(name, 0)
        assert n_missing == expected_missing

@pytest.mark.filterwarnings('ignore:The default value of `parser` will change')
@pytest.mark.parametrize('params, err_msg', [({'parser': 'unknown'}, "The 'parser' parameter of fetch_openml must be a str among"), ({'as_frame': 'unknown'}, "The 'as_frame' parameter of fetch_openml must be an instance")])
def test_fetch_openml_validation_parameter(monkeypatch, params, err_msg):
    if False:
        print('Hello World!')
    data_id = 1119
    _monkey_patch_webbased_functions(monkeypatch, data_id, True)
    with pytest.raises(ValueError, match=err_msg):
        fetch_openml(data_id=data_id, **params)

@pytest.mark.parametrize('params', [{'as_frame': True, 'parser': 'auto'}, {'as_frame': 'auto', 'parser': 'auto'}, {'as_frame': False, 'parser': 'pandas'}])
def test_fetch_openml_requires_pandas_error(monkeypatch, params):
    if False:
        for i in range(10):
            print('nop')
    'Check that we raise the proper errors when we require pandas.'
    data_id = 1119
    try:
        check_pandas_support('test_fetch_openml_requires_pandas')
    except ImportError:
        _monkey_patch_webbased_functions(monkeypatch, data_id, True)
        err_msg = 'requires pandas to be installed. Alternatively, explicitly'
        with pytest.raises(ImportError, match=err_msg):
            fetch_openml(data_id=data_id, **params)
    else:
        raise SkipTest('This test requires pandas to not be installed.')

def test_fetch_openml_requires_pandas_in_future(monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    'Check that we raise a warning that pandas will be required in the future.'
    params = {'as_frame': False, 'parser': 'auto'}
    data_id = 1119
    try:
        check_pandas_support('test_fetch_openml_requires_pandas')
    except ImportError:
        _monkey_patch_webbased_functions(monkeypatch, data_id, True)
        warn_msg = "From version 1.4, `parser='auto'` with `as_frame=False` will use pandas"
        with pytest.warns(FutureWarning, match=warn_msg):
            fetch_openml(data_id=data_id, **params)
    else:
        raise SkipTest('This test requires pandas to not be installed.')

@pytest.mark.filterwarnings('ignore:Version 1 of dataset Australian is inactive')
@pytest.mark.filterwarnings('ignore:The default value of `parser` will change')
@pytest.mark.parametrize('params, err_msg', [({'parser': 'pandas'}, "Sparse ARFF datasets cannot be loaded with parser='pandas'"), ({'as_frame': True}, 'Sparse ARFF datasets cannot be loaded with as_frame=True.'), ({'parser': 'pandas', 'as_frame': True}, 'Sparse ARFF datasets cannot be loaded with as_frame=True.')])
def test_fetch_openml_sparse_arff_error(monkeypatch, params, err_msg):
    if False:
        return 10
    'Check that we raise the expected error for sparse ARFF datasets and\n    a wrong set of incompatible parameters.\n    '
    pytest.importorskip('pandas')
    data_id = 292
    _monkey_patch_webbased_functions(monkeypatch, data_id, True)
    with pytest.raises(ValueError, match=err_msg):
        fetch_openml(data_id=data_id, cache=False, **params)

@fails_if_pypy
@pytest.mark.filterwarnings('ignore:Version 1 of dataset Australian is inactive')
@pytest.mark.parametrize('data_id, data_type', [(61, 'dataframe'), (292, 'sparse')])
def test_fetch_openml_auto_mode(monkeypatch, data_id, data_type):
    if False:
        i = 10
        return i + 15
    'Check the auto mode of `fetch_openml`.'
    pd = pytest.importorskip('pandas')
    _monkey_patch_webbased_functions(monkeypatch, data_id, True)
    data = fetch_openml(data_id=data_id, as_frame='auto', parser='auto', cache=False)
    klass = pd.DataFrame if data_type == 'dataframe' else scipy.sparse.csr_matrix
    assert isinstance(data.data, klass)

@fails_if_pypy
def test_convert_arff_data_dataframe_warning_low_memory_pandas(monkeypatch):
    if False:
        while True:
            i = 10
    'Check that we raise a warning regarding the working memory when using\n    LIAC-ARFF parser.'
    pytest.importorskip('pandas')
    data_id = 1119
    _monkey_patch_webbased_functions(monkeypatch, data_id, True)
    msg = 'Could not adhere to working_memory config.'
    with pytest.warns(UserWarning, match=msg):
        with config_context(working_memory=1e-06):
            fetch_openml(data_id=data_id, as_frame=True, cache=False, parser='liac-arff')

@pytest.mark.parametrize('gzip_response', [True, False])
def test_fetch_openml_iris_warn_multiple_version(monkeypatch, gzip_response):
    if False:
        return 10
    'Check that a warning is raised when multiple versions exist and no version is\n    requested.'
    data_id = 61
    data_name = 'iris'
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response)
    msg = 'Multiple active versions of the dataset matching the name iris exist. Versions may be fundamentally different, returning version 1.'
    with pytest.warns(UserWarning, match=msg):
        fetch_openml(name=data_name, as_frame=False, cache=False, parser='liac-arff')

@pytest.mark.parametrize('gzip_response', [True, False])
def test_fetch_openml_no_target(monkeypatch, gzip_response):
    if False:
        print('Hello World!')
    'Check that we can get a dataset without target.'
    data_id = 61
    target_column = None
    expected_observations = 150
    expected_features = 5
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response)
    data = fetch_openml(data_id=data_id, target_column=target_column, cache=False, as_frame=False, parser='liac-arff')
    assert data.data.shape == (expected_observations, expected_features)
    assert data.target is None

@pytest.mark.parametrize('gzip_response', [True, False])
@pytest.mark.parametrize('parser', ['liac-arff', 'pandas'])
def test_missing_values_pandas(monkeypatch, gzip_response, parser):
    if False:
        while True:
            i = 10
    'check that missing values in categories are compatible with pandas\n    categorical'
    pytest.importorskip('pandas')
    data_id = 42585
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response=gzip_response)
    penguins = fetch_openml(data_id=data_id, cache=False, as_frame=True, parser=parser)
    cat_dtype = penguins.data.dtypes['sex']
    assert penguins.data['sex'].isna().any()
    assert_array_equal(cat_dtype.categories, ['FEMALE', 'MALE', '_'])

@pytest.mark.parametrize('gzip_response', [True, False])
@pytest.mark.parametrize('dataset_params', [{'data_id': 40675}, {'data_id': None, 'name': 'glass2', 'version': 1}])
def test_fetch_openml_inactive(monkeypatch, gzip_response, dataset_params):
    if False:
        return 10
    'Check that we raise a warning when the dataset is inactive.'
    data_id = 40675
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response)
    msg = 'Version 1 of dataset glass2 is inactive,'
    with pytest.warns(UserWarning, match=msg):
        glass2 = fetch_openml(cache=False, as_frame=False, parser='liac-arff', **dataset_params)
    assert glass2.data.shape == (163, 9)
    assert glass2.details['id'] == '40675'

@pytest.mark.parametrize('gzip_response', [True, False])
@pytest.mark.parametrize('data_id, params, err_type, err_msg', [(40675, {'name': 'glass2'}, ValueError, 'No active dataset glass2 found'), (61, {'data_id': 61, 'target_column': ['sepalwidth', 'class']}, ValueError, 'Can only handle homogeneous multi-target datasets'), (40945, {'data_id': 40945, 'as_frame': False}, ValueError, 'STRING attributes are not supported for array representation. Try as_frame=True'), (2, {'data_id': 2, 'target_column': 'family', 'as_frame': True}, ValueError, "Target column 'family'"), (2, {'data_id': 2, 'target_column': 'family', 'as_frame': False}, ValueError, "Target column 'family'"), (61, {'data_id': 61, 'target_column': 'undefined'}, KeyError, "Could not find target_column='undefined'"), (61, {'data_id': 61, 'target_column': ['undefined', 'class']}, KeyError, "Could not find target_column='undefined'")])
@pytest.mark.parametrize('parser', ['liac-arff', 'pandas'])
def test_fetch_openml_error(monkeypatch, gzip_response, data_id, params, err_type, err_msg, parser):
    if False:
        print('Hello World!')
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response)
    if params.get('as_frame', True) or parser == 'pandas':
        pytest.importorskip('pandas')
    with pytest.raises(err_type, match=err_msg):
        fetch_openml(cache=False, parser=parser, **params)

@pytest.mark.parametrize('params, err_type, err_msg', [({'data_id': -1, 'name': None, 'version': 'version'}, ValueError, "The 'version' parameter of fetch_openml must be an int in the range"), ({'data_id': -1, 'name': 'nAmE'}, ValueError, "The 'data_id' parameter of fetch_openml must be an int in the range"), ({'data_id': -1, 'name': 'nAmE', 'version': 'version'}, ValueError, "The 'version' parameter of fetch_openml must be an int"), ({}, ValueError, 'Neither name nor data_id are provided. Please provide name or data_id.')])
def test_fetch_openml_raises_illegal_argument(params, err_type, err_msg):
    if False:
        return 10
    with pytest.raises(err_type, match=err_msg):
        fetch_openml(**params)

@pytest.mark.parametrize('gzip_response', [True, False])
def test_warn_ignore_attribute(monkeypatch, gzip_response):
    if False:
        return 10
    data_id = 40966
    expected_row_id_msg = "target_column='{}' has flag is_row_identifier."
    expected_ignore_msg = "target_column='{}' has flag is_ignore."
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response)
    target_col = 'MouseID'
    msg = expected_row_id_msg.format(target_col)
    with pytest.warns(UserWarning, match=msg):
        fetch_openml(data_id=data_id, target_column=target_col, cache=False, as_frame=False, parser='liac-arff')
    target_col = 'Genotype'
    msg = expected_ignore_msg.format(target_col)
    with pytest.warns(UserWarning, match=msg):
        fetch_openml(data_id=data_id, target_column=target_col, cache=False, as_frame=False, parser='liac-arff')
    target_col = 'MouseID'
    msg = expected_row_id_msg.format(target_col)
    with pytest.warns(UserWarning, match=msg):
        fetch_openml(data_id=data_id, target_column=[target_col, 'class'], cache=False, as_frame=False, parser='liac-arff')
    target_col = 'Genotype'
    msg = expected_ignore_msg.format(target_col)
    with pytest.warns(UserWarning, match=msg):
        fetch_openml(data_id=data_id, target_column=[target_col, 'class'], cache=False, as_frame=False, parser='liac-arff')

@pytest.mark.parametrize('gzip_response', [True, False])
def test_dataset_with_openml_error(monkeypatch, gzip_response):
    if False:
        return 10
    data_id = 1
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response)
    msg = 'OpenML registered a problem with the dataset. It might be unusable. Error:'
    with pytest.warns(UserWarning, match=msg):
        fetch_openml(data_id=data_id, cache=False, as_frame=False, parser='liac-arff')

@pytest.mark.parametrize('gzip_response', [True, False])
def test_dataset_with_openml_warning(monkeypatch, gzip_response):
    if False:
        i = 10
        return i + 15
    data_id = 3
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response)
    msg = 'OpenML raised a warning on the dataset. It might be unusable. Warning:'
    with pytest.warns(UserWarning, match=msg):
        fetch_openml(data_id=data_id, cache=False, as_frame=False, parser='liac-arff')

def test_fetch_openml_overwrite_default_params_read_csv(monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    'Check that we can overwrite the default parameters of `read_csv`.'
    pytest.importorskip('pandas')
    data_id = 1590
    _monkey_patch_webbased_functions(monkeypatch, data_id=data_id, gzip_response=False)
    common_params = {'data_id': data_id, 'as_frame': True, 'cache': False, 'parser': 'pandas'}
    adult_without_spaces = fetch_openml(**common_params)
    adult_with_spaces = fetch_openml(**common_params, read_csv_kwargs={'skipinitialspace': False})
    assert all((cat.startswith(' ') for cat in adult_with_spaces.frame['class'].cat.categories))
    assert not any((cat.startswith(' ') for cat in adult_without_spaces.frame['class'].cat.categories))

@pytest.mark.parametrize('gzip_response', [True, False])
def test_open_openml_url_cache(monkeypatch, gzip_response, tmpdir):
    if False:
        i = 10
        return i + 15
    data_id = 61
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response)
    openml_path = sklearn.datasets._openml._DATA_FILE.format(data_id)
    cache_directory = str(tmpdir.mkdir('scikit_learn_data'))
    response1 = _open_openml_url(openml_path, cache_directory)
    location = _get_local_path(openml_path, cache_directory)
    assert os.path.isfile(location)
    response2 = _open_openml_url(openml_path, cache_directory)
    assert response1.read() == response2.read()

@pytest.mark.parametrize('write_to_disk', [True, False])
def test_open_openml_url_unlinks_local_path(monkeypatch, tmpdir, write_to_disk):
    if False:
        for i in range(10):
            print('nop')
    data_id = 61
    openml_path = sklearn.datasets._openml._DATA_FILE.format(data_id)
    cache_directory = str(tmpdir.mkdir('scikit_learn_data'))
    location = _get_local_path(openml_path, cache_directory)

    def _mock_urlopen(request, *args, **kwargs):
        if False:
            return 10
        if write_to_disk:
            with open(location, 'w') as f:
                f.write('')
        raise ValueError('Invalid request')
    monkeypatch.setattr(sklearn.datasets._openml, 'urlopen', _mock_urlopen)
    with pytest.raises(ValueError, match='Invalid request'):
        _open_openml_url(openml_path, cache_directory)
    assert not os.path.exists(location)

def test_retry_with_clean_cache(tmpdir):
    if False:
        print('Hello World!')
    data_id = 61
    openml_path = sklearn.datasets._openml._DATA_FILE.format(data_id)
    cache_directory = str(tmpdir.mkdir('scikit_learn_data'))
    location = _get_local_path(openml_path, cache_directory)
    os.makedirs(os.path.dirname(location))
    with open(location, 'w') as f:
        f.write('')

    @_retry_with_clean_cache(openml_path, cache_directory)
    def _load_data():
        if False:
            i = 10
            return i + 15
        if os.path.exists(location):
            raise Exception('File exist!')
        return 1
    warn_msg = 'Invalid cache, redownloading file'
    with pytest.warns(RuntimeWarning, match=warn_msg):
        result = _load_data()
    assert result == 1

def test_retry_with_clean_cache_http_error(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    data_id = 61
    openml_path = sklearn.datasets._openml._DATA_FILE.format(data_id)
    cache_directory = str(tmpdir.mkdir('scikit_learn_data'))

    @_retry_with_clean_cache(openml_path, cache_directory)
    def _load_data():
        if False:
            while True:
                i = 10
        raise HTTPError(url=None, code=412, msg='Simulated mock error', hdrs=None, fp=None)
    error_msg = 'Simulated mock error'
    with pytest.raises(HTTPError, match=error_msg):
        _load_data()

@pytest.mark.parametrize('gzip_response', [True, False])
def test_fetch_openml_cache(monkeypatch, gzip_response, tmpdir):
    if False:
        print('Hello World!')

    def _mock_urlopen_raise(request, *args, **kwargs):
        if False:
            while True:
                i = 10
        raise ValueError('This mechanism intends to test correct cachehandling. As such, urlopen should never be accessed. URL: %s' % request.get_full_url())
    data_id = 61
    cache_directory = str(tmpdir.mkdir('scikit_learn_data'))
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response)
    (X_fetched, y_fetched) = fetch_openml(data_id=data_id, cache=True, data_home=cache_directory, return_X_y=True, as_frame=False, parser='liac-arff')
    monkeypatch.setattr(sklearn.datasets._openml, 'urlopen', _mock_urlopen_raise)
    (X_cached, y_cached) = fetch_openml(data_id=data_id, cache=True, data_home=cache_directory, return_X_y=True, as_frame=False, parser='liac-arff')
    np.testing.assert_array_equal(X_fetched, X_cached)
    np.testing.assert_array_equal(y_fetched, y_cached)

@fails_if_pypy
@pytest.mark.parametrize('as_frame, parser', [(True, 'liac-arff'), (False, 'liac-arff'), (True, 'pandas'), (False, 'pandas')])
def test_fetch_openml_verify_checksum(monkeypatch, as_frame, cache, tmpdir, parser):
    if False:
        print('Hello World!')
    'Check that the checksum is working as expected.'
    if as_frame or parser == 'pandas':
        pytest.importorskip('pandas')
    data_id = 2
    _monkey_patch_webbased_functions(monkeypatch, data_id, True)
    original_data_module = OPENML_TEST_DATA_MODULE + '.' + f'id_{data_id}'
    original_data_file_name = 'data-v1-dl-1666876.arff.gz'
    corrupt_copy_path = tmpdir / 'test_invalid_checksum.arff'
    with _open_binary(original_data_module, original_data_file_name) as orig_file:
        orig_gzip = gzip.open(orig_file, 'rb')
        data = bytearray(orig_gzip.read())
        data[len(data) - 1] = 37
    with gzip.GzipFile(corrupt_copy_path, 'wb') as modified_gzip:
        modified_gzip.write(data)
    mocked_openml_url = sklearn.datasets._openml.urlopen

    def swap_file_mock(request, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        url = request.get_full_url()
        if url.endswith('data/v1/download/1666876'):
            with open(corrupt_copy_path, 'rb') as f:
                corrupted_data = f.read()
            return _MockHTTPResponse(BytesIO(corrupted_data), is_gzip=True)
        else:
            return mocked_openml_url(request)
    monkeypatch.setattr(sklearn.datasets._openml, 'urlopen', swap_file_mock)
    with pytest.raises(ValueError) as exc:
        sklearn.datasets.fetch_openml(data_id=data_id, cache=False, as_frame=as_frame, parser=parser)
    assert exc.match('1666876')

def test_open_openml_url_retry_on_network_error(monkeypatch):
    if False:
        i = 10
        return i + 15

    def _mock_urlopen_network_error(request, *args, **kwargs):
        if False:
            print('Hello World!')
        raise HTTPError('', 404, 'Simulated network error', None, None)
    monkeypatch.setattr(sklearn.datasets._openml, 'urlopen', _mock_urlopen_network_error)
    invalid_openml_url = 'invalid-url'
    with pytest.warns(UserWarning, match=re.escape(f'A network error occurred while downloading {_OPENML_PREFIX + invalid_openml_url}. Retrying...')) as record:
        with pytest.raises(HTTPError, match='Simulated network error'):
            _open_openml_url(invalid_openml_url, None, delay=0)
        assert len(record) == 3

@pytest.mark.parametrize('gzip_response', [True, False])
@pytest.mark.parametrize('parser', ('liac-arff', 'pandas'))
def test_fetch_openml_with_ignored_feature(monkeypatch, gzip_response, parser):
    if False:
        for i in range(10):
            print('nop')
    'Check that we can load the "zoo" dataset.\n    Non-regression test for:\n    https://github.com/scikit-learn/scikit-learn/issues/14340\n    '
    if parser == 'pandas':
        pytest.importorskip('pandas')
    data_id = 62
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response)
    dataset = sklearn.datasets.fetch_openml(data_id=data_id, cache=False, as_frame=False, parser=parser)
    assert dataset is not None
    assert dataset['data'].shape == (101, 16)
    assert 'animal' not in dataset['feature_names']

def test_fetch_openml_strip_quotes(monkeypatch):
    if False:
        return 10
    'Check that we strip the single quotes when used as a string delimiter.\n\n    Non-regression test for:\n    https://github.com/scikit-learn/scikit-learn/issues/23381\n    '
    pd = pytest.importorskip('pandas')
    data_id = 40966
    _monkey_patch_webbased_functions(monkeypatch, data_id=data_id, gzip_response=False)
    common_params = {'as_frame': True, 'cache': False, 'data_id': data_id}
    mice_pandas = fetch_openml(parser='pandas', **common_params)
    mice_liac_arff = fetch_openml(parser='liac-arff', **common_params)
    pd.testing.assert_series_equal(mice_pandas.target, mice_liac_arff.target)
    assert not mice_pandas.target.str.startswith("'").any()
    assert not mice_pandas.target.str.endswith("'").any()
    mice_pandas = fetch_openml(parser='pandas', target_column='NUMB_N', **common_params)
    mice_liac_arff = fetch_openml(parser='liac-arff', target_column='NUMB_N', **common_params)
    pd.testing.assert_series_equal(mice_pandas.frame['class'], mice_liac_arff.frame['class'])
    assert not mice_pandas.frame['class'].str.startswith("'").any()
    assert not mice_pandas.frame['class'].str.endswith("'").any()

def test_fetch_openml_leading_whitespace(monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    'Check that we can strip leading whitespace in pandas parser.\n\n    Non-regression test for:\n    https://github.com/scikit-learn/scikit-learn/issues/25311\n    '
    pd = pytest.importorskip('pandas')
    data_id = 1590
    _monkey_patch_webbased_functions(monkeypatch, data_id=data_id, gzip_response=False)
    common_params = {'as_frame': True, 'cache': False, 'data_id': data_id}
    adult_pandas = fetch_openml(parser='pandas', **common_params)
    adult_liac_arff = fetch_openml(parser='liac-arff', **common_params)
    pd.testing.assert_series_equal(adult_pandas.frame['class'], adult_liac_arff.frame['class'])

def test_fetch_openml_quotechar_escapechar(monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    'Check that we can handle escapechar and single/double quotechar.\n\n    Non-regression test for:\n    https://github.com/scikit-learn/scikit-learn/issues/25478\n    '
    pd = pytest.importorskip('pandas')
    data_id = 42074
    _monkey_patch_webbased_functions(monkeypatch, data_id=data_id, gzip_response=False)
    common_params = {'as_frame': True, 'cache': False, 'data_id': data_id}
    adult_pandas = fetch_openml(parser='pandas', **common_params)
    adult_liac_arff = fetch_openml(parser='liac-arff', **common_params)
    pd.testing.assert_frame_equal(adult_pandas.frame, adult_liac_arff.frame)

def test_fetch_openml_deprecation_parser(monkeypatch):
    if False:
        print('Hello World!')
    'Check that we raise a deprecation warning for parser parameter.'
    pytest.importorskip('pandas')
    data_id = 61
    _monkey_patch_webbased_functions(monkeypatch, data_id=data_id, gzip_response=False)
    with pytest.warns(FutureWarning, match='The default value of `parser` will change'):
        sklearn.datasets.fetch_openml(data_id=data_id)