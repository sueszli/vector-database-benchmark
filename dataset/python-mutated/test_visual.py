from __future__ import annotations
import pytest
pytest
import base64
import datetime
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Literal
import numpy as np
import PIL.Image
from bokeh.core.enums import MarkerType, ToolIcon
from bokeh.core.has_props import HasProps
from tests.support.util.api import verify_all
from _util_property import _TestHasProps, _TestModel
import bokeh.core.property.visual as bcpv
ALL = ('DashPattern', 'FontSize', 'HatchPatternType', 'Image', 'MinMaxBounds', 'MarkerType')

class Foo(HasProps):
    pat = bcpv.DashPattern()

class TestDashPattern:

    def test_valid_named(self) -> None:
        if False:
            while True:
                i = 10
        f = Foo()
        assert f.pat == []
        f.pat = 'solid'
        assert f.pat == []
        f.pat = 'dashed'
        assert f.pat == [6]
        f.pat = 'dotted'
        assert f.pat == [2, 4]
        f.pat = 'dotdash'
        assert f.pat == [2, 4, 6, 4]
        f.pat = 'dashdot'
        assert f.pat == [6, 4, 2, 4]

    def test_valid_string(self) -> None:
        if False:
            i = 10
            return i + 15
        f = Foo()
        f.pat = ''
        assert f.pat == []
        f.pat = '2'
        assert f.pat == [2]
        f.pat = '2 4'
        assert f.pat == [2, 4]
        f.pat = '2 4 6'
        assert f.pat == [2, 4, 6]
        with pytest.raises(ValueError):
            f.pat = 'abc 6'

    def test_valid_list(self) -> None:
        if False:
            while True:
                i = 10
        f = Foo()
        f.pat = ()
        assert f.pat == ()
        f.pat = (2,)
        assert f.pat == (2,)
        f.pat = (2, 4)
        assert f.pat == (2, 4)
        f.pat = (2, 4, 6)
        assert f.pat == (2, 4, 6)
        with pytest.raises(ValueError):
            f.pat = (2, 4.2)
        with pytest.raises(ValueError):
            f.pat = (2, 'a')

    def test_valid(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        prop = bcpv.DashPattern()
        assert prop.is_valid('')
        assert prop.is_valid(())
        assert prop.is_valid([])
        assert prop.is_valid('solid')
        assert prop.is_valid('dashed')
        assert prop.is_valid('dotted')
        assert prop.is_valid('dotdash')
        assert prop.is_valid('dashdot')
        assert prop.is_valid([1, 2, 3])
        assert prop.is_valid('1 2 3')

    def test_invalid(self) -> None:
        if False:
            return 10
        prop = bcpv.DashPattern()
        assert not prop.is_valid(None)
        assert not prop.is_valid(False)
        assert not prop.is_valid(True)
        assert not prop.is_valid(0)
        assert not prop.is_valid(1)
        assert not prop.is_valid(0.0)
        assert not prop.is_valid(1.0)
        assert not prop.is_valid(1.0 + 1j)
        assert not prop.is_valid('foo')
        assert not prop.is_valid('DASHDOT')
        assert not prop.is_valid([1, 2, 3.0])
        assert not prop.is_valid('1 2 x')
        assert not prop.is_valid(_TestHasProps())
        assert not prop.is_valid(_TestModel())

    def test_has_ref(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        prop = bcpv.DashPattern()
        assert not prop.has_ref

    def test_str(self) -> None:
        if False:
            return 10
        prop = bcpv.DashPattern()
        assert str(prop) == 'DashPattern'
css_units = '%|em|ex|ch|ic|rem|vw|vh|vi|vb|vmin|vmax|cm|mm|q|in|pc|pt|px'

class Test_FontSize:

    def test_valid(self) -> None:
        if False:
            print('Hello World!')
        prop = bcpv.FontSize()
        for unit in css_units.split('|'):
            v = f'10{unit}'
            assert prop.is_valid(v)
            v = f'10.2{unit}'
            assert prop.is_valid(v)
        for unit in css_units.upper().split('|'):
            v = f'10{unit}'
            assert prop.is_valid(v)
            v = f'10.2{unit}'
            assert prop.is_valid(v)

    def test_invalid(self) -> None:
        if False:
            return 10
        prop = bcpv.FontSize()
        for unit in css_units.split('|'):
            v = f'_10{unit}'
            assert not prop.is_valid(v)
            v = f'_10.2{unit}'
            assert not prop.is_valid(v)
        for unit in css_units.upper().split('|'):
            v = f'_10{unit}'
            assert not prop.is_valid(v)
            v = f'_10.2{unit}'
            assert not prop.is_valid(v)
        assert not prop.is_valid(None)
        assert not prop.is_valid(False)
        assert not prop.is_valid(True)
        assert not prop.is_valid(0)
        assert not prop.is_valid(1)
        assert not prop.is_valid(0.0)
        assert not prop.is_valid(1.0)
        assert not prop.is_valid(1.0 + 1j)
        assert not prop.is_valid('')
        assert not prop.is_valid(())
        assert not prop.is_valid([])
        assert not prop.is_valid({})
        assert not prop.is_valid(_TestHasProps())
        assert not prop.is_valid(_TestModel())

    def test_has_ref(self) -> None:
        if False:
            print('Hello World!')
        prop = bcpv.FontSize()
        assert not prop.has_ref

    def test_str(self) -> None:
        if False:
            return 10
        prop = bcpv.FontSize()
        assert str(prop) == 'FontSize'

class Test_Image:

    def test_default_creation(self) -> None:
        if False:
            print('Hello World!')
        bcpv.Image()

    def test_validate_None(self) -> None:
        if False:
            while True:
                i = 10
        prop = bcpv.Image()
        assert not prop.is_valid(None)

    def test_validate_data_url(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        prop = bcpv.Image()
        assert prop.is_valid('data:image/png;base64,')

    def test_validate_Path(self) -> None:
        if False:
            while True:
                i = 10
        prop = bcpv.Image()
        assert prop.is_valid(Path('some/path'))

    def test_validate_raw_path(self) -> None:
        if False:
            return 10
        prop = bcpv.Image()
        assert prop.is_valid('some/path')

    @pytest.mark.parametrize('typ', ('png', 'gif', 'tiff'))
    def test_validate_PIL(self, typ) -> None:
        if False:
            i = 10
            return i + 15
        file = BytesIO()
        image = PIL.Image.new('RGBA', size=(50, 50), color=(155, 0, 0))
        image.save(file, typ)
        prop = bcpv.Image()
        assert prop.is_valid(image)

    def test_validate_numpy_RGB(self) -> None:
        if False:
            i = 10
            return i + 15
        data = np.zeros((50, 50, 3), dtype=np.uint8)
        data[:, 30:35] = [255, 0, 0]
        prop = bcpv.Image()
        assert prop.is_valid(data)

    def test_validate_numpy_RGBA(self) -> None:
        if False:
            print('Hello World!')
        data = np.zeros((50, 50, 4), dtype=np.uint8)
        data[:, 30:35] = [255, 0, 0, 255]
        prop = bcpv.Image()
        assert prop.is_valid(data)

    def test_validate_invalid(self) -> None:
        if False:
            return 10
        prop = bcpv.Image()
        assert not prop.is_valid(10)
        assert not prop.is_valid(True)
        assert not prop.is_valid(False)
        assert not prop.is_valid([])
        assert not prop.is_valid({})
        assert not prop.is_valid(set())
        data = np.zeros((50, 50, 2), dtype=np.uint8)
        assert not prop.is_valid(data)
        data = np.zeros((50, 50), dtype=np.uint8)
        assert not prop.is_valid(data)

    def test_transform_data_url(self) -> None:
        if False:
            while True:
                i = 10
        prop = bcpv.Image()
        image = 'data:image/png;base64,'
        assert prop.transform(image) == image

    def test_transform_path(self) -> None:
        if False:
            i = 10
            return i + 15
        with tempfile.TemporaryDirectory() as dir:
            path = Path(dir) / 'image.png'
            image = PIL.Image.new('RGBA', size=(50, 50), color=(155, 0, 0))
            image.save(path, 'png')
            prop = bcpv.Image()
            assert prop.transform(path).startswith('data:image/png')

    def test_transform_file(self) -> None:
        if False:
            return 10
        with tempfile.NamedTemporaryFile() as file:
            image = PIL.Image.new('RGBA', size=(50, 50), color=(155, 0, 0))
            image.save(file, 'png')
            prop = bcpv.Image()
            assert prop.transform(file).startswith('data:image/png')

    def test_transform_string(self) -> None:
        if False:
            return 10
        prop = bcpv.Image()
        image = ToolIcon.zoom_in
        assert prop.transform(image) == image

    def test_transform_numpy(self) -> None:
        if False:
            while True:
                i = 10
        data = np.zeros((50, 50, 3), dtype=np.uint8)
        data[:, 30:35] = [255, 0, 0]
        value = PIL.Image.fromarray(data)
        out = BytesIO()
        value.save(out, 'png')
        expected = 'data:image/png;base64,' + base64.b64encode(out.getvalue()).decode('ascii')
        prop = bcpv.Image()
        assert prop.transform(data) == expected

    @pytest.mark.parametrize('typ', ('png', 'gif', 'tiff'))
    def test_transform_PIL(self, typ: Literal['png', 'gif', 'tiff']) -> None:
        if False:
            for i in range(10):
                print('nop')
        image = PIL.Image.new('RGBA', size=(50, 50), color=(155, 0, 0))
        out0 = BytesIO()
        image.save(out0, typ)
        value = PIL.Image.open(out0)
        out1 = BytesIO()
        value.save(out1, typ)
        expected = 'data:image/%s;base64,' % typ + base64.b64encode(out1.getvalue()).decode('ascii')
        prop = bcpv.Image()
        assert prop.transform(value) == expected

    def test_transform_bad(self) -> None:
        if False:
            while True:
                i = 10
        prop = bcpv.Image()
        with pytest.raises(ValueError):
            assert prop.transform(10)

    def test_has_ref(self) -> None:
        if False:
            print('Hello World!')
        prop = bcpv.Image()
        assert not prop.has_ref

    def test_str(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        prop = bcpv.Image()
        assert str(prop) == 'Image'

class Test_MinMaxBounds:

    def test_valid_no_datetime(self) -> None:
        if False:
            while True:
                i = 10
        prop = bcpv.MinMaxBounds(accept_datetime=False)
        assert prop.is_valid('auto')
        assert prop.is_valid((12, 13))
        assert prop.is_valid((-32, -13))
        assert prop.is_valid((12.1, 13.1))
        assert prop.is_valid((None, 13.1))
        assert prop.is_valid((-22, None))

    def test_invalid_no_datetime(self) -> None:
        if False:
            print('Hello World!')
        prop = bcpv.MinMaxBounds(accept_datetime=False)
        assert not prop.is_valid(None)
        assert not prop.is_valid('string')
        assert not prop.is_valid(12)
        assert not prop.is_valid(('a', 'b'))
        assert not prop.is_valid((13, 12))
        assert not prop.is_valid((13.1, 12.2))
        assert not prop.is_valid((datetime.date(2012, 10, 1), datetime.date(2012, 12, 2)))

    def test_MinMaxBounds_with_datetime(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        prop = bcpv.MinMaxBounds(accept_datetime=True)
        assert prop.is_valid((datetime.datetime(2012, 10, 1), datetime.datetime(2012, 12, 2)))
        assert not prop.is_valid((datetime.datetime(2012, 10, 1), 22))

    def test_has_ref(self) -> None:
        if False:
            i = 10
            return i + 15
        prop = bcpv.MinMaxBounds()
        assert not prop.has_ref

    def test_str(self) -> None:
        if False:
            i = 10
            return i + 15
        prop = bcpv.MinMaxBounds()
        assert str(prop).startswith('MinMaxBounds(')

class Test_MarkerType:

    def test_valid(self) -> None:
        if False:
            return 10
        prop = bcpv.MarkerType()
        for typ in MarkerType:
            assert prop.is_valid(typ)

    def test_invalid(self) -> None:
        if False:
            return 10
        prop = bcpv.MarkerType()
        assert not prop.is_valid(None)
        assert not prop.is_valid(False)
        assert not prop.is_valid(True)
        assert not prop.is_valid(0)
        assert not prop.is_valid(1)
        assert not prop.is_valid(0.0)
        assert not prop.is_valid(1.0)
        assert not prop.is_valid(1.0 + 1j)
        assert not prop.is_valid('')
        assert not prop.is_valid(())
        assert not prop.is_valid([])
        assert not prop.is_valid({})
        assert not prop.is_valid(_TestHasProps())
        assert not prop.is_valid(_TestModel())
        assert not prop.is_valid('string')
        assert not prop.is_valid([1, 2, 3])
        assert not prop.is_valid([1, 2, 3.0])

    def test_has_ref(self) -> None:
        if False:
            i = 10
            return i + 15
        prop = bcpv.MarkerType()
        assert not prop.has_ref

    def test_str(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        prop = bcpv.MarkerType()
        assert str(prop).startswith('MarkerType(')
Test___all__ = verify_all(bcpv, ALL)