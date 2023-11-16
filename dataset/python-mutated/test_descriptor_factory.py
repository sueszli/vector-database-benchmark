from __future__ import annotations
import pytest
pytest
from tests.support.util.api import verify_all
import bokeh.core.property.descriptor_factory as bcpdf
ALL = ('PropertyDescriptorFactory',)

def test_make_descriptors_not_implemented() -> None:
    if False:
        for i in range(10):
            print('nop')
    obj = bcpdf.PropertyDescriptorFactory()
    with pytest.raises(NotImplementedError):
        obj.make_descriptors('foo')
Test___all__ = verify_all(bcpdf, ALL)