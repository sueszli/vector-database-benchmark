import numpy as np
from numpy.testing import assert_equal
import numpy as np
import pytest
from numpy.testing import assert_equal

from arctic._util import FwPointersCfg
from arctic.store._ndarray_store import NdarrayStore, _APPEND_COUNT
from arctic.store.version_store import register_versioned_storage
from tests.integration.store.test_version_store import FwPointersCtx

register_versioned_storage(NdarrayStore)

@pytest.mark.parametrize('fw_pointers_cfg', [FwPointersCfg.DISABLED, FwPointersCfg.HYBRID, FwPointersCfg.ENABLED])
def test_append_simple_ndarray(library, fw_pointers_cfg):
    with FwPointersCtx(fw_pointers_cfg):
        ndarr = np.ones(1000, dtype='int64')
        library.write('MYARR', ndarr)
        library.append('MYARR', np.ones(1000, dtype='int64'))
        library.append('MYARR', np.ones(1000, dtype='int64'))
        library.append('MYARR', np.ones(2005, dtype='int64'))
        saved_arr = library.read('MYARR').data
        assert np.all(np.ones(5005, dtype='int64') == saved_arr)


@pytest.mark.parametrize('fw_pointers_cfg', [FwPointersCfg.DISABLED, FwPointersCfg.HYBRID, FwPointersCfg.ENABLED])
def test_append_simple_ndarray_promoting_types(library, fw_pointers_cfg):
    with FwPointersCtx(fw_pointers_cfg):
        ndarr = np.ones(100, dtype='int64')
        library.write('MYARR', ndarr)
        library.append('MYARR', np.ones(100, dtype='float64'))
        library.append('MYARR', np.ones(100, dtype='int64'))
        library.append('MYARR', np.ones(205, dtype='float64'))
        saved_arr = library.read('MYARR').data
        assert np.all(np.ones(505, dtype='float64') == saved_arr)


def test_promote_types(library):
    ndarr = np.empty(1000, dtype=[('abc', 'int64')])
    library.write('MYARR', ndarr[:800])
    library.append('MYARR', ndarr[-200:].astype([('abc', 'float64')]))
    saved_arr = library.read('MYARR').data
    assert np.all(ndarr.astype([('abc', 'float64')]) == saved_arr)


def test_promote_types2(library):
    ndarr = np.array(np.arange(1000), dtype=[('abc', 'float64')])
    library.write('MYARR', ndarr[:800])
    library.append('MYARR', ndarr[-200:].astype([('abc', 'int64')]))
    saved_arr = library.read('MYARR').data
    assert np.all(ndarr.astype([('abc', np.promote_types('float64', 'int64'))]) == saved_arr)


def test_promote_types_smaller_sizes(library):
    library.write('MYARR', np.ones(100, dtype='int64'))
    library.append('MYARR', np.ones(100, dtype='int32'))
    saved_arr = library.read('MYARR').data
    assert np.all(np.ones(200, dtype='int64') == saved_arr)


def test_promote_types_larger_sizes(library):
    library.write('MYARR', np.ones(100, dtype='int32'))
    library.append('MYARR', np.ones(100, dtype='int64'))
    saved_arr = library.read('MYARR').data
    assert np.all(np.ones(200, dtype='int64') == saved_arr)


def test_promote_field_types_smaller_sizes(library):
    arr = np.array([(3, 7)], dtype=[('a', '<i8'), ('b', '<i8')])
    library.write('MYARR', arr)
    arr = np.array([(9, 8)], dtype=[('a', '<i4'), ('b', '<i8')])
    library.append('MYARR', arr)
    saved_arr = library.read('MYARR').data
    expected = np.array([(3, 7), (9, 8)], dtype=[('a', '<i8'), ('b', '<i8')])
    assert np.all(saved_arr == expected)


def test_promote_field_types_larger_sizes(library):
    arr = np.array([(3, 7)], dtype=[('a', '<i4'), ('b', '<i8')])
    library.write('MYARR', arr)
    arr = np.array([(9, 8)], dtype=[('a', '<i8'), ('b', '<i8')])
    library.append('MYARR', arr)
    saved_arr = library.read('MYARR').data
    expected = np.array([(3, 7), (9, 8)], dtype=[('a', '<i8'), ('b', '<i8')])
    assert np.all(saved_arr == expected)


@pytest.mark.parametrize('fw_pointers_cfg', [FwPointersCfg.DISABLED, FwPointersCfg.HYBRID, FwPointersCfg.ENABLED])
def test_append_ndarray_with_field_shape(library, fw_pointers_cfg):
    with FwPointersCtx(fw_pointers_cfg):
        ndarr = np.empty(10, dtype=[('A', 'int64'), ('B', 'float64', (2,))])
        ndarr['A'] = 1
        ndarr['B'] = 2
        ndarr2 = np.empty(10, dtype=[('A', 'int64'), ('B', 'int64', (2,))])
        ndarr2['A'] = 1
        ndarr2['B'] = 2

        library.write('MYARR', ndarr)
        library.append('MYARR', ndarr2)
        saved_arr = library.read('MYARR').data
        ndarr3 = np.empty(20, dtype=[('A', 'int64'), ('B', 'float64', (2,))])
        ndarr3['A'] = 1
        ndarr3['B'] = 2
        assert np.all(ndarr3 == saved_arr)


@pytest.mark.parametrize('fw_pointers_cfg', [FwPointersCfg.DISABLED, FwPointersCfg.HYBRID, FwPointersCfg.ENABLED])
def test_append_read_large_ndarray(library, fw_pointers_cfg):
    with FwPointersCtx(fw_pointers_cfg):
        dtype = np.dtype([('abc', 'int64')])
        ndarr = np.arange(50 * 1024 * 1024 / dtype.itemsize).view(dtype=dtype)
        assert len(ndarr.tobytes()) > 16 * 1024 * 1024
        library.write('MYARR1', ndarr)
        # Exactly enough appends to trigger 2 re-compacts, so the result should be identical
        # to writing the whole array at once
        ndarr2 = np.arange(240).view(dtype=dtype)
        for n in np.split(ndarr2, 120):
            library.append('MYARR1', n)

        saved_arr = library.read('MYARR1').data
        assert np.all(np.concatenate([ndarr, ndarr2]) == saved_arr)

        library.write('MYARR2', np.concatenate([ndarr, ndarr2]))

        version1 = library._read_metadata('MYARR1')
        version2 = library._read_metadata('MYARR2')
        assert version1['append_count'] == version2['append_count']
        assert version1['append_size'] == version2['append_size']
        assert version1['segment_count'] == version2['segment_count']
        assert version1['up_to'] == version2['up_to']


@pytest.mark.parametrize('fw_pointers_cfg', [FwPointersCfg.DISABLED, FwPointersCfg.HYBRID, FwPointersCfg.ENABLED])
def test_save_append_read_ndarray(library, fw_pointers_cfg):
    with FwPointersCtx(fw_pointers_cfg):
        dtype = np.dtype([('abc', 'int64')])
        ndarr = np.arange(30 * 1024 * 1024 / dtype.itemsize).view(dtype=dtype)
        assert len(ndarr.tobytes()) > 16 * 1024 * 1024
        library.write('MYARR', ndarr)

        sliver = np.arange(30).view(dtype=dtype)
        library.append('MYARR', sliver)

        saved_arr = library.read('MYARR').data
        assert np.all(np.concatenate([ndarr, sliver]) == saved_arr)

        library.append('MYARR', sliver)
        saved_arr = library.read('MYARR').data
        assert np.all(np.concatenate([ndarr, sliver, sliver]) == saved_arr)


def test_save_append_read_1row_ndarray(library):
    dtype = np.dtype([('abc', 'int64')])
    ndarr = np.arange(30 * 1024 * 1024 / dtype.itemsize).view(dtype=dtype)
    assert len(ndarr.tobytes()) > 16 * 1024 * 1024
    library.write('MYARR', ndarr)

    sliver = np.arange(1).view(dtype=dtype)
    library.append('MYARR', sliver)

    saved_arr = library.read('MYARR').data
    assert np.all(np.concatenate([ndarr, sliver]) == saved_arr)

    library.append('MYARR', sliver)
    saved_arr = library.read('MYARR').data
    assert np.all(np.concatenate([ndarr, sliver, sliver]) == saved_arr)


def test_append_too_large_ndarray(library):
    dtype = np.dtype([('abc', 'int64')])
    ndarr = np.arange(30 * 1024 * 1024 / dtype.itemsize).view(dtype=dtype)
    assert len(ndarr.tobytes()) > 16 * 1024 * 1024
    library.write('MYARR', ndarr)
    library.append('MYARR', ndarr)
    saved_arr = library.read('MYARR').data
    assert np.all(np.concatenate([ndarr, ndarr]) == saved_arr)


def test_empty_field_append_keeps_all_columns(library):
    ndarr = np.array([(3, 5)], dtype=[('a', '<i'), ('b', '<i')])
    ndarr2 = np.array([], dtype=[('a', '<i')])
    library.write('MYARR', ndarr)
    library.append('MYARR', ndarr2)
    saved_arr = library.read('MYARR').data
    assert np.all(saved_arr == np.array([(3, 5)], dtype=[('a', '<i'), ('b', '<i')]))


@pytest.mark.parametrize('fw_pointers_cfg', [FwPointersCfg.DISABLED, FwPointersCfg.HYBRID, FwPointersCfg.ENABLED])
def test_empty_append_promotes_dtype(library, fw_pointers_cfg):
    with FwPointersCtx(fw_pointers_cfg):
        ndarr = np.array(["a", "b", "c"])
        ndarr2 = np.array([])
        library.write('MYARR', ndarr)
        library.append('MYARR', ndarr2)
        saved_arr = library.read('MYARR').data
        assert np.all(saved_arr == ndarr)


def test_empty_append_promotes_dtype2(library):
    ndarr = np.array([])
    ndarr2 = np.array(["a", "b", "c"])
    library.write('MYARR', ndarr)
    library.append('MYARR', ndarr2)
    saved_arr = library.read('MYARR').data
    assert np.all(saved_arr == ndarr2)


def test_empty_append_promotes_dtype3(library):
    ndarr = np.array([])
    ndarr2 = np.array(["a", "b", "c"])
    library.write('MYARR', ndarr)
    library.append('MYARR', ndarr2)
    library.append('MYARR', ndarr)
    library.append('MYARR', ndarr2)
    saved_arr = library.read('MYARR').data
    assert np.all(saved_arr == np.hstack((ndarr2, ndarr2)))


def test_convert_to_structured_array(library):
    arr = np.ones(100, dtype='int64')
    library.write('MYARR', arr)
    arr = np.array([(6,)], dtype=[('a', '<i8')])
    with pytest.raises(ValueError):
        library.append('MYARR', arr)


@pytest.mark.parametrize('fw_pointers_cfg', [FwPointersCfg.DISABLED, FwPointersCfg.HYBRID, FwPointersCfg.ENABLED])
def test_empty_append_concat_and_rewrite(library, fw_pointers_cfg):
    with FwPointersCtx(fw_pointers_cfg):
        ndarr = np.array([])
        ndarr2 = np.array(["a", "b", "c"])
        library.write('MYARR', ndarr)
        for _ in range(_APPEND_COUNT + 2):
            library.append('MYARR', ndarr)
        library.append('MYARR', ndarr2)
        saved_arr = library.read('MYARR').data
        assert np.all(saved_arr == ndarr2)


@pytest.mark.parametrize('fw_pointers_cfg', [FwPointersCfg.DISABLED, FwPointersCfg.HYBRID, FwPointersCfg.ENABLED])
def test_empty_append_concat_and_rewrite_2(library, fw_pointers_cfg):
    with FwPointersCtx(fw_pointers_cfg):
        ndarr2 = np.array(["a", "b", "c"])
        library.write('MYARR', ndarr2)
        for _ in range(_APPEND_COUNT + 1):
            library.append('MYARR', ndarr2)
        saved_arr = library.read('MYARR').data
        assert np.all(saved_arr == np.hstack([ndarr2] * (_APPEND_COUNT + 2)))


@pytest.mark.parametrize('fw_pointers_cfg', [FwPointersCfg.DISABLED, FwPointersCfg.HYBRID, FwPointersCfg.ENABLED])
def test_empty_append_concat_and_rewrite_3(library, fw_pointers_cfg):
    with FwPointersCtx(fw_pointers_cfg):
        ndarr = np.array([])
        ndarr2 = np.array(["a", "b", "c"])
        library.write('MYARR', ndarr2)
        for _ in range(_APPEND_COUNT + 1):
            library.append('MYARR', ndarr)
        saved_arr = library.read('MYARR').data
        assert np.all(saved_arr == ndarr2)


def test_append_with_extra_columns(library):
    ndarr = np.array([(2.1, 1, "a")], dtype=[('C', float), ('B', int), ('A', 'S1')])
    ndarr2 = np.array([("b", 2, 3.1, 'c', 4, 5.)], dtype=[('A', 'S1'), ('B', int), ('C', float),
                                                          ('D', 'S1'), ('E', int), ('F', float)])
    expected = np.array([("a", 1, 2.1, '', 0, np.nan),
                         ("b", 2, 3.1, 'c', 4, 5.)],
                        dtype=np.dtype([('A', 'S1'), ('B', int), ('C', float),
                                        ('D', 'S1'), ('E', int), ('F', float)]))
    library.write('MYARR', ndarr)
    library.append('MYARR', ndarr2)
    saved_arr = library.read('MYARR').data

    assert expected.dtype == saved_arr.dtype
    assert_equal(expected.tolist(), saved_arr.tolist())


@pytest.mark.parametrize('fw_pointers_cfg', [FwPointersCfg.DISABLED, FwPointersCfg.HYBRID, FwPointersCfg.ENABLED])
def test_save_append_delete_append(library, fw_pointers_cfg):
    with FwPointersCtx(fw_pointers_cfg):
        dtype = np.dtype([('abc', 'int64')])
        ndarr = np.arange(30 / dtype.itemsize).view(dtype=dtype)
        v1 = library.write('MYARR', ndarr)

        sliver = np.arange(30).view(dtype=dtype)
        v2 = library.append('MYARR', sliver)

        # intentionally leave an orphaned chunk lying around here
        library._delete_version('MYARR', v2.version, do_cleanup=False)

        sliver2 = np.arange(start=10, stop=40).view(dtype=dtype)
        # we can't append here, as the latest version is now out of sync with version_nums.
        # This gets translated to a do_append by the handler anyway.
        v3 = library.write('MYARR', np.concatenate([ndarr, sliver2]))

        assert np.all(ndarr == library.read('MYARR', as_of=v1.version).data)

        # Check that we don't get the orphaned chunk from v2 back again.
        assert np.all(np.concatenate([ndarr, sliver2]) == library.read('MYARR', as_of=v3.version).data)


@pytest.mark.parametrize('fw_pointers_cfg', [FwPointersCfg.DISABLED, FwPointersCfg.HYBRID, FwPointersCfg.ENABLED])
def test_append_after_failed_append(library, fw_pointers_cfg):
    with FwPointersCtx(fw_pointers_cfg):
        dtype = np.dtype([('abc', 'int64')])
        ndarr = np.arange(30 / dtype.itemsize).view(dtype=dtype)

        v1 = library.write('MYARR', ndarr)

        sliver = np.arange(3, 4).view(dtype=dtype)
        v2 = library.append('MYARR', sliver)

        # simulate a failed append - intentionally leave an orphaned chunk lying around here
        library._delete_version('MYARR', v2.version, do_cleanup=False)

        sliver2 = np.arange(3, 5).view(dtype=dtype)
        v3 = library.append('MYARR', sliver2)

        assert np.all(ndarr == library.read('MYARR', as_of=v1.version).data)
        assert np.all(np.concatenate([ndarr, sliver2]) == library.read('MYARR', as_of=v3.version).data)


def test_append_reorder_columns(library):
    foo = np.array([(1, 2)], dtype=np.dtype([('a', 'u1'), ('b', 'u1')]))
    library.write('MYARR', foo)
    foo = np.array([(1, 2)], dtype=np.dtype([('b', 'u1'), ('a', 'u1')]))
    library.append('MYARR', foo)

    assert np.all(library.read('MYARR').data == np.array([(2, 1), (1, 2)], dtype=[('b', 'u1'), ('a', 'u1')]))
