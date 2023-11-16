import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import KeyEvent
import astropy.units as u
from astropy.coordinates import FK5, SkyCoord, galactocentric_frame_defaults
from astropy.time import Time
from astropy.visualization.wcsaxes.core import WCSAxes
from astropy.wcs import WCS
from .test_images import BaseImageTests

class TestDisplayWorldCoordinate(BaseImageTests):

    def teardown_method(self, method):
        if False:
            i = 10
            return i + 15
        plt.close('all')

    def test_overlay_coords(self, ignore_matplotlibrc, tmp_path):
        if False:
            while True:
                i = 10
        wcs = WCS(self.msx_header)
        fig = plt.figure(figsize=(4, 4))
        canvas = fig.canvas
        ax = WCSAxes(fig, [0.1, 0.1, 0.8, 0.8], wcs=wcs)
        fig.add_axes(ax)
        fig.savefig(tmp_path / 'test1.png')
        string_world = ax._display_world_coords(0.523412, 0.518311)
        assert string_world == '0°29\'45" -0°29\'20" (world)'
        event1 = KeyEvent('test_pixel_coords', canvas, 'w')
        fig.canvas.callbacks.process('key_press_event', event1)
        string_pixel = ax._display_world_coords(0.523412, 0.523412)
        assert string_pixel == '0.523412 0.523412 (pixel)'
        event3 = KeyEvent('test_pixel_coords', canvas, 'w')
        fig.canvas.callbacks.process('key_press_event', event3)
        string_world2 = ax._display_world_coords(0.523412, 0.518311)
        assert string_world2 == '0°29\'45" -0°29\'20" (world)'
        overlay = ax.get_coords_overlay('fk5')
        overlay[0].set_major_formatter('d.ddd')
        fig.savefig(tmp_path / 'test2.png')
        event4 = KeyEvent('test_pixel_coords', canvas, 'w')
        fig.canvas.callbacks.process('key_press_event', event4)
        string_world3 = ax._display_world_coords(0.523412, 0.518311)
        assert string_world3 == '267.176° -28°45\'56" (world, overlay 1)'
        overlay = ax.get_coords_overlay(FK5())
        overlay[0].set_major_formatter('d.ddd')
        fig.savefig(tmp_path / 'test3.png')
        event5 = KeyEvent('test_pixel_coords', canvas, 'w')
        fig.canvas.callbacks.process('key_press_event', event5)
        string_world4 = ax._display_world_coords(0.523412, 0.518311)
        assert string_world4 == '267.176° -28°45\'56" (world, overlay 2)'
        overlay = ax.get_coords_overlay(FK5(equinox=Time('J2030')))
        overlay[0].set_major_formatter('d.ddd')
        fig.savefig(tmp_path / 'test4.png')
        event6 = KeyEvent('test_pixel_coords', canvas, 'w')
        fig.canvas.callbacks.process('key_press_event', event6)
        string_world5 = ax._display_world_coords(0.523412, 0.518311)
        assert string_world5 == '267.652° -28°46\'23" (world, overlay 3)'

    def test_cube_coords(self, ignore_matplotlibrc, tmp_path):
        if False:
            i = 10
            return i + 15
        wcs = WCS(self.cube_header)
        fig = plt.figure(figsize=(4, 4))
        canvas = fig.canvas
        ax = WCSAxes(fig, [0.1, 0.1, 0.8, 0.8], wcs=wcs, slices=('y', 50, 'x'))
        fig.add_axes(ax)
        fig.savefig(tmp_path / 'test.png')
        string_world = ax._display_world_coords(0.523412, 0.518311)
        assert string_world == '3h26m52.0s 30°37\'17" 2563 (world)'
        event1 = KeyEvent('test_pixel_coords', canvas, 'w')
        fig.canvas.callbacks.process('key_press_event', event1)
        string_pixel = ax._display_world_coords(0.523412, 0.523412)
        assert string_pixel == '0.523412 0.523412 (pixel)'

    def test_cube_coords_uncorr_slicing(self, ignore_matplotlibrc, tmp_path):
        if False:
            for i in range(10):
                print('nop')
        wcs = WCS(self.cube_header)
        fig = plt.figure(figsize=(4, 4))
        canvas = fig.canvas
        ax = WCSAxes(fig, [0.1, 0.1, 0.8, 0.8], wcs=wcs, slices=('x', 'y', 2))
        fig.add_axes(ax)
        fig.savefig(tmp_path / 'test.png')
        string_world = ax._display_world_coords(0.523412, 0.518311)
        assert string_world == '3h26m56.6s 30°18\'19" (world)'
        event1 = KeyEvent('test_pixel_coords', canvas, 'w')
        fig.canvas.callbacks.process('key_press_event', event1)
        string_pixel = ax._display_world_coords(0.523412, 0.523412)
        assert string_pixel == '0.523412 0.523412 (pixel)'

    def test_plot_coord_3d_transform(self):
        if False:
            print('Hello World!')
        wcs = WCS(self.msx_header)
        with galactocentric_frame_defaults.set('latest'):
            coord = SkyCoord(0 * u.kpc, 0 * u.kpc, 0 * u.kpc, frame='galactocentric')
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection=wcs)
        (point,) = ax.plot_coord(coord, 'ro')
        np.testing.assert_allclose(point.get_xydata()[0], [0, 0], atol=0.0001)