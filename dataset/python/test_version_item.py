import pandas as pd

from arctic.store.versioned_item import VersionedItem


def test_versioned_item_str():
    item = VersionedItem(symbol="sym",
                         library="ONEMINUTE",
                         data=pd.DataFrame(),
                         version=1.0,
                         host='myhost',
                         metadata={'metadata': 'foo'})

    expected = "VersionedItem(symbol=sym,library=ONEMINUTE," + \
               "data=<class 'pandas.core.frame.DataFrame'>,version=1.0,metadata={'metadata': 'foo'},host=myhost)"
    assert str(item) == expected
    assert repr(item) == expected


def test_versioned_item_default_host():
    item = VersionedItem(symbol="sym",
                         library="ONEMINUTE",
                         data=[1, 2, 3],
                         version=1.0,
                         metadata={'metadata': 'foo'})

    expected_item = VersionedItem(symbol="sym",
                                  library="ONEMINUTE",
                                  data=[1, 2, 3],
                                  version=1.0,
                                  host=None,
                                  metadata={'metadata': 'foo'})

    assert item == expected_item


def test_versioned_item_str_handles_none():
    item = VersionedItem(symbol=None,
                         library=None,
                         data=None,
                         version=None,
                         metadata=None,
                         host=None)

    assert str(item)


def test_versioned_item_metadata_dict():
    item = VersionedItem(symbol="test",
                         library="test_lib",
                         data=None,
                         version=1.2,
                         metadata=None,
                         host=None)
    assert(item.metadata_dict() == {'symbol': 'test', 'library': 'test_lib', 'version': 1.2})
