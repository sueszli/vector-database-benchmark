"""Tests for `ee_tile_layers` package."""
import unittest
from unittest.mock import patch
import box
import ee
from geemap import ee_tile_layers
from tests import fake_ee

@patch.object(ee, 'Feature', fake_ee.Feature)
@patch.object(ee, 'FeatureCollection', fake_ee.FeatureCollection)
@patch.object(ee, 'Geometry', fake_ee.Geometry)
@patch.object(ee, 'Image', fake_ee.Image)
@patch.object(ee, 'ImageCollection', fake_ee.ImageCollection)
class TestEETileLayers(unittest.TestCase):
    """Tests for `ee_tile_layers` package."""

    def test_validate_vis_params_none(self):
        if False:
            for i in range(10):
                print('nop')
        vis_params = None
        self.assertEqual(ee_tile_layers._validate_vis_params(vis_params), {})

    def test_validate_vis_params_invalid_palette_type(self):
        if False:
            i = 10
            return i + 15
        vis_params = {'min': 0.1, 'palette': {}}
        with self.assertRaisesRegex(ValueError, 'palette must be'):
            ee_tile_layers._validate_vis_params(vis_params)

    def test_validate_vis_params_box_palette_no_default(self):
        if False:
            while True:
                i = 10
        vis_params = {'min': 0.1, 'palette': box.Box({'unknown-key': 2})}
        with self.assertRaisesRegex(ValueError, 'Box object is invalid'):
            ee_tile_layers._validate_vis_params(vis_params)

    def test_validate_vis_params_box_palette(self):
        if False:
            while True:
                i = 10
        vis_params = {'min': 0.1, 'palette': box.Box({'default': ['#00ff00']})}
        self.assertEqual(ee_tile_layers._validate_vis_params(vis_params), {'min': 0.1, 'palette': ['#00ff00']})

    def test_validate_vis_params_str_palette(self):
        if False:
            print('Hello World!')
        vis_params = {'min': 0.1, 'palette': '00FF00'}
        self.assertEqual(ee_tile_layers._validate_vis_params(vis_params), {'min': 0.1, 'palette': '#00ff00'})

    def test_validate_vis_params_list_palette(self):
        if False:
            print('Hello World!')
        vis_params = {'min': 0.1, 'palette': ['#00ff00']}
        self.assertEqual(ee_tile_layers._validate_vis_params(vis_params), {'min': 0.1, 'palette': ['#00ff00']})

    def test_get_tile_url_format_geometry(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(ee_tile_layers._get_tile_url_format(ee.Geometry(''), {}), 'url-format')
        self.assertEqual(ee_tile_layers._get_tile_url_format(ee.Feature(''), {}), 'url-format')
        self.assertEqual(ee_tile_layers._get_tile_url_format(ee.FeatureCollection([]), {}), 'url-format')

    def test_get_tile_url_format_image(self):
        if False:
            print('Hello World!')
        self.assertEqual(ee_tile_layers._get_tile_url_format(ee.Image(), {}), 'url-format')

    def test_get_tile_url_format_imagecollection(self):
        if False:
            print('Hello World!')
        self.assertEqual(ee_tile_layers._get_tile_url_format(ee.ImageCollection([]), {}), 'url-format')

    def test_get_tile_url_format_invalid_type(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(AttributeError, 'Cannot add an object of type str'):
            ee_tile_layers._get_tile_url_format('some-invalid-type', {})

    def test_ee_leaflet_tile_layer(self):
        if False:
            for i in range(10):
                print('nop')
        layer = ee_tile_layers.EELeafletTileLayer(ee_object=ee.Image(1), vis_params={'min': 42, 'palette': '012345'}, name='a-name', shown=False, opacity=0.5)
        self.assertEqual(layer.url_format, 'url-format')

    def test_ee_folium_tile_layer(self):
        if False:
            for i in range(10):
                print('nop')
        layer = ee_tile_layers.EEFoliumTileLayer(ee_object=ee.Image(1), vis_params={'min': 42, 'palette': '012345'}, name='a-name', shown=False, opacity=0.5)
        self.assertEqual(layer.url_format, 'url-format')