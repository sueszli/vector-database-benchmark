from unittest import TestCase
import plotly.graph_objs as go
import pytest

class TestPropertyValidation(TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.scatter = go.Scatter()
        self.scatter.name = 'Scatter 1'

    def test_validators_work_attr(self):
        if False:
            print('Hello World!')
        "\n        Note: all of the individual validators are tested in\n        `_plotly_utils/tests/validators`. Here we're just making sure that\n        datatypes make use of validators\n        "
        with pytest.raises(ValueError):
            self.scatter.name = [1, 2, 3]

    def test_validators_work_item(self):
        if False:
            return 10
        "\n        Note: all of the individual validators are tested in\n        `_plotly_utils/tests/validators`. Here we're just making sure that\n        datatypes make use of validators\n        "
        with pytest.raises(ValueError):
            self.scatter['name'] = [1, 2, 3]

    def test_invalid_attr_assignment(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(ValueError):
            self.scatter.bogus = 87

    def test_invalid_item_assignment(self):
        if False:
            while True:
                i = 10
        with pytest.raises(ValueError):
            self.scatter['bogus'] = 87

    def test_invalid_dot_assignment(self):
        if False:
            return 10
        with pytest.raises(ValueError):
            self.scatter['marker.bogus'] = 87

    def test_invalid_tuple_assignment(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(ValueError):
            self.scatter['marker', 'bogus'] = 87

    def test_invalid_constructor_kwarg(self):
        if False:
            while True:
                i = 10
        with pytest.raises(ValueError):
            go.Scatter(bogus=87)

class TestPropertyPresentation(TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.scatter = go.Scatter()
        self.scatter.name = 'Scatter 1'
        self.layout = go.Layout()

    def test_present_dataarray(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertIsNone(self.scatter.x)
        self.scatter.x = [1, 2, 3, 4]
        self.assertEqual(self.scatter.to_plotly_json()['x'], [1, 2, 3, 4])
        self.assertEqual(self.scatter.x, (1, 2, 3, 4))

    def test_present_compound_array(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.layout.images, ())
        self.layout.images = [go.layout.Image(layer='above'), go.layout.Image(layer='below')]
        self.assertEqual(self.layout.to_plotly_json()['images'], [{'layer': 'above'}, {'layer': 'below'}])
        self.assertEqual(self.layout.images, (go.layout.Image(layer='above'), go.layout.Image(layer='below')))

    def test_present_colorscale(self):
        if False:
            print('Hello World!')
        self.assertIsNone(self.scatter.marker.colorscale)
        self.scatter.marker.colorscale = [(0, 'red'), (1, 'green')]
        self.assertEqual(self.scatter.to_plotly_json()['marker']['colorscale'], [[0, 'red'], [1, 'green']])
        self.assertEqual(self.scatter.marker.colorscale, ((0, 'red'), (1, 'green')))
        self.scatter.marker.colorscale = 'viridis'
        colorscale = self.scatter.to_plotly_json()['marker']['colorscale']
        colorscale = [col[1] for col in colorscale]
        self.scatter.marker.colorscale = 'viridis_r'
        colorscale_r = self.scatter.to_plotly_json()['marker']['colorscale']
        colorscale_r = [col[1] for col in colorscale_r]
        self.assertEqual(colorscale[::-1], colorscale_r)

class TestPropertyIterContains(TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.parcoords = go.Parcoords()
        self.parcoords.name = 'Scatter 1'

    def test_contains(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue('name' in self.parcoords)
        self.assertTrue('line' in self.parcoords)
        self.assertTrue('type' in self.parcoords)
        self.assertTrue('dimensions' in self.parcoords)
        self.assertFalse('bogus' in self.parcoords)

    def test_iter(self):
        if False:
            print('Hello World!')
        parcoords_list = list(self.parcoords)
        self.assertTrue('name' in parcoords_list)
        self.assertTrue('line' in parcoords_list)
        self.assertTrue('type' in parcoords_list)
        self.assertTrue('dimensions' in parcoords_list)
        self.assertFalse('bogus' in parcoords_list)