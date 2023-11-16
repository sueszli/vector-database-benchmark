import datetime
from pathlib import Path
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.util.version import Version
pyreadstat = pytest.importorskip('pyreadstat')

@pytest.mark.filterwarnings('ignore::pandas.errors.ChainedAssignmentError')
@pytest.mark.parametrize('path_klass', [lambda p: p, Path])
def test_spss_labelled_num(path_klass, datapath):
    if False:
        for i in range(10):
            print('nop')
    fname = path_klass(datapath('io', 'data', 'spss', 'labelled-num.sav'))
    df = pd.read_spss(fname, convert_categoricals=True)
    expected = pd.DataFrame({'VAR00002': 'This is one'}, index=[0])
    expected['VAR00002'] = pd.Categorical(expected['VAR00002'])
    tm.assert_frame_equal(df, expected)
    df = pd.read_spss(fname, convert_categoricals=False)
    expected = pd.DataFrame({'VAR00002': 1.0}, index=[0])
    tm.assert_frame_equal(df, expected)

@pytest.mark.filterwarnings('ignore::pandas.errors.ChainedAssignmentError')
def test_spss_labelled_num_na(datapath):
    if False:
        i = 10
        return i + 15
    fname = datapath('io', 'data', 'spss', 'labelled-num-na.sav')
    df = pd.read_spss(fname, convert_categoricals=True)
    expected = pd.DataFrame({'VAR00002': ['This is one', None]})
    expected['VAR00002'] = pd.Categorical(expected['VAR00002'])
    tm.assert_frame_equal(df, expected)
    df = pd.read_spss(fname, convert_categoricals=False)
    expected = pd.DataFrame({'VAR00002': [1.0, np.nan]})
    tm.assert_frame_equal(df, expected)

@pytest.mark.filterwarnings('ignore::pandas.errors.ChainedAssignmentError')
def test_spss_labelled_str(datapath):
    if False:
        while True:
            i = 10
    fname = datapath('io', 'data', 'spss', 'labelled-str.sav')
    df = pd.read_spss(fname, convert_categoricals=True)
    expected = pd.DataFrame({'gender': ['Male', 'Female']})
    expected['gender'] = pd.Categorical(expected['gender'])
    tm.assert_frame_equal(df, expected)
    df = pd.read_spss(fname, convert_categoricals=False)
    expected = pd.DataFrame({'gender': ['M', 'F']})
    tm.assert_frame_equal(df, expected)

@pytest.mark.filterwarnings('ignore::pandas.errors.ChainedAssignmentError')
def test_spss_umlauts(datapath):
    if False:
        return 10
    fname = datapath('io', 'data', 'spss', 'umlauts.sav')
    df = pd.read_spss(fname, convert_categoricals=True)
    expected = pd.DataFrame({'var1': ['the ä umlaut', 'the ü umlaut', 'the ä umlaut', 'the ö umlaut']})
    expected['var1'] = pd.Categorical(expected['var1'])
    tm.assert_frame_equal(df, expected)
    df = pd.read_spss(fname, convert_categoricals=False)
    expected = pd.DataFrame({'var1': [1.0, 2.0, 1.0, 3.0]})
    tm.assert_frame_equal(df, expected)

def test_spss_usecols(datapath):
    if False:
        while True:
            i = 10
    fname = datapath('io', 'data', 'spss', 'labelled-num.sav')
    with pytest.raises(TypeError, match='usecols must be list-like.'):
        pd.read_spss(fname, usecols='VAR00002')

def test_spss_umlauts_dtype_backend(datapath, dtype_backend):
    if False:
        i = 10
        return i + 15
    fname = datapath('io', 'data', 'spss', 'umlauts.sav')
    df = pd.read_spss(fname, convert_categoricals=False, dtype_backend=dtype_backend)
    expected = pd.DataFrame({'var1': [1.0, 2.0, 1.0, 3.0]}, dtype='Int64')
    if dtype_backend == 'pyarrow':
        pa = pytest.importorskip('pyarrow')
        from pandas.arrays import ArrowExtensionArray
        expected = pd.DataFrame({col: ArrowExtensionArray(pa.array(expected[col], from_pandas=True)) for col in expected.columns})
    tm.assert_frame_equal(df, expected)

def test_invalid_dtype_backend():
    if False:
        return 10
    msg = "dtype_backend numpy is invalid, only 'numpy_nullable' and 'pyarrow' are allowed."
    with pytest.raises(ValueError, match=msg):
        pd.read_spss('test', dtype_backend='numpy')

@pytest.mark.filterwarnings('ignore::pandas.errors.ChainedAssignmentError')
def test_spss_metadata(datapath):
    if False:
        print('Hello World!')
    fname = datapath('io', 'data', 'spss', 'labelled-num.sav')
    df = pd.read_spss(fname)
    metadata = {'column_names': ['VAR00002'], 'column_labels': [None], 'column_names_to_labels': {'VAR00002': None}, 'file_encoding': 'UTF-8', 'number_columns': 1, 'number_rows': 1, 'variable_value_labels': {'VAR00002': {1.0: 'This is one'}}, 'value_labels': {'labels0': {1.0: 'This is one'}}, 'variable_to_label': {'VAR00002': 'labels0'}, 'notes': [], 'original_variable_types': {'VAR00002': 'F8.0'}, 'readstat_variable_types': {'VAR00002': 'double'}, 'table_name': None, 'missing_ranges': {}, 'missing_user_values': {}, 'variable_storage_width': {'VAR00002': 8}, 'variable_display_width': {'VAR00002': 8}, 'variable_alignment': {'VAR00002': 'unknown'}, 'variable_measure': {'VAR00002': 'unknown'}, 'file_label': None, 'file_format': 'sav/zsav'}
    if Version(pyreadstat.__version__) >= Version('1.2.4'):
        metadata.update({'creation_time': datetime.datetime(2015, 2, 6, 14, 33, 36), 'modification_time': datetime.datetime(2015, 2, 6, 14, 33, 36)})
    assert df.attrs == metadata