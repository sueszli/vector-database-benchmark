"""Scraper for sphinx-gallery.

This is used to collect screenshots from the examples when executed via
sphinx-gallery. This can be included in any project wanting to take
advantage of this by adding the following to your sphinx ``conf.py``:

.. code-block:: python

    sphinx_gallery_conf = {
        ...
        'image_scrapers': ('vispy',)
    }

The scraper is provided to sphinx-gallery via the
``vispy._get_sg_image_scraper()`` function.

"""
from __future__ import annotations
import os
import time
import shutil
from vispy.io import imsave
from vispy.gloo.util import _screenshot
from vispy.scene import SceneCanvas
from sphinx_gallery.scrapers import optipng, figure_rst

class VisPyGalleryScraper:
    """Custom sphinx-gallery scraper to save the current Canvas to an image."""

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return self.__class__.__name__

    def __call__(self, block, block_vars, gallery_conf):
        if False:
            return 10
        'Scrape VisPy Canvases and applications.\n\n        Parameters\n        ----------\n        block : tuple\n            A tuple containing the (label, content, line_number) of the block.\n        block_vars : dict\n            Dict of block variables.\n        gallery_conf : dict\n            Contains the configuration of Sphinx-Gallery\n\n        Returns\n        -------\n        rst : str\n            The ReSTructuredText that will be rendered to HTML containing\n            the images. This is often produced by\n            :func:`sphinx_gallery.scrapers.figure_rst`.\n\n        '
        example_fn = block_vars['src_file']
        frame_num_list = self._get_frame_list_from_source(example_fn)
        image_path_iterator = block_vars['image_path_iterator']
        canvas_or_widget = get_canvaslike_from_globals(block_vars['example_globals'])
        if not frame_num_list:
            image_paths = []
        elif isinstance(frame_num_list[0], str):
            image_paths = []
            for (frame_image, image_path) in zip(frame_num_list, image_path_iterator):
                image_path = os.path.splitext(image_path)[0] + os.path.splitext(frame_image)[1]
                shutil.move(frame_image, image_path)
                image_paths.append(image_path)
        else:
            image_paths = self._save_example_to_files(canvas_or_widget, frame_num_list, gallery_conf, image_path_iterator)
        fig_titles = ''
        return figure_rst(image_paths, gallery_conf['src_dir'], fig_titles)

    def _save_example_to_files(self, canvas_or_widget, frame_num_list, gallery_conf, image_path_iterator):
        if False:
            print('Hello World!')
        image_path = next(image_path_iterator)
        frame_grabber = FrameGrabber(canvas_or_widget, frame_num_list)
        frame_grabber.collect_frames()
        if len(frame_num_list) > 1:
            image_path = os.path.splitext(image_path)[0] + '.gif'
            frame_grabber.save_animation(image_path)
        else:
            frame_grabber.save_frame(image_path)
        frame_grabber.cleanup()
        if 'images' in gallery_conf['compress_images']:
            optipng(image_path, gallery_conf['compress_images_args'])
        return [image_path]

    def _get_frame_list_from_source(self, filename):
        if False:
            i = 10
            return i + 15
        lines = open(filename, 'rb').read().decode('utf-8').splitlines()
        for line in lines[:10]:
            if not line.startswith('# vispy:'):
                continue
            if 'gallery-exports' in line:
                _frames = line.split('gallery-exports')[1].split(',')[0].strip()
                frames = self._frame_exports_to_list(_frames)
                break
            if 'gallery ' in line:
                _frames = line.split('gallery')[1].split(',')[0].strip()
                frames = self._frame_specifier_to_list(_frames)
                break
        else:
            frames = []
        return frames

    def _frame_specifier_to_list(self, frame_specifier):
        if False:
            while True:
                i = 10
        _frames = frame_specifier or '0'
        frames = [int(i) for i in _frames.split(':')]
        if not frames:
            frames = [5]
        if len(frames) > 1:
            frames = list(range(*frames))
        return frames

    def _frame_exports_to_list(self, frame_specifier):
        if False:
            return 10
        frames = frame_specifier.split(' ')
        frame_paths = []
        for frame_fn in frames:
            if not os.path.isfile(frame_fn):
                raise FileNotFoundError('Example gallery frame specifier must be a frame number, frame range, or relative filename produced by the example.')
            frame_paths.append(frame_fn)
        return frame_paths

def get_canvaslike_from_globals(globals_dict):
    if False:
        for i in range(10):
            print('nop')
    qt_widget = _get_qt_top_parent(globals_dict)
    if qt_widget is not None:
        return qt_widget
    if 'canvas' in globals_dict:
        return globals_dict['canvas']
    if 'Canvas' in globals_dict:
        return globals_dict['Canvas']()
    if 'fig' in globals_dict:
        return globals_dict['fig']
    return None

def _get_qt_top_parent(globals_dict):
    if False:
        for i in range(10):
            print('nop')
    if 'QWidget' not in globals_dict and 'QMainWindow' not in globals_dict and ('QtWidgets' not in globals_dict):
        return None
    qtwidgets = globals_dict.get('QtWidgets')
    qmainwindow = globals_dict.get('QMainWindow', getattr(qtwidgets, 'QMainWindow', None))
    qwidget = globals_dict.get('QWidget', getattr(qtwidgets, 'QWidget', qmainwindow))
    all_qt_widgets = [widget for widget in globals_dict.values() if isinstance(widget, qwidget) and widget is not None]
    all_qt_mains = [widget for widget in all_qt_widgets if isinstance(widget, qmainwindow)]
    if all_qt_mains:
        return all_qt_mains[0]
    if all_qt_widgets:
        return all_qt_widgets[0]
    return None

class FrameGrabber:
    """Helper to grab a series of screenshots from the current Canvas-like object."""

    def __init__(self, canvas_obj, frame_grab_list: list[int]):
        if False:
            for i in range(10):
                print('nop')
        self._canvas = canvas_obj
        self._done = False
        self._current_frame = -1
        self._collected_images = []
        self._frames_to_grab = frame_grab_list[:]

    def cleanup(self):
        if False:
            for i in range(10):
                print('nop')
        from PyQt5.QtWidgets import QApplication
        for child_widget in QApplication.allWidgets():
            if hasattr(child_widget, 'close'):
                child_widget.close()
        QApplication.processEvents()

    def on_draw(self, _):
        if False:
            i = 10
            return i + 15
        if self._done:
            return
        self._current_frame += 1
        if self._current_frame in self._frames_to_grab:
            self._frames_to_grab.remove(self._current_frame)
            if isinstance(self._canvas, SceneCanvas):
                im = self._canvas.render(alpha=True)
            else:
                im = _screenshot()
            self._collected_images.append(im)
        if not self._frames_to_grab or self._current_frame > self._frames_to_grab[0]:
            self._done = True

    def collect_frames(self):
        if False:
            i = 10
            return i + 15
        'Show current Canvas and render and collect all frames requested.'
        if self._is_qt_widget():
            self._grab_qt_screenshot()
        else:
            self._grab_vispy_screenshots()

    def _is_qt_widget(self):
        if False:
            return 10
        try:
            from PyQt5.QtWidgets import QWidget
        except ImportError:
            return False
        return isinstance(self._canvas, QWidget)

    def _grab_qt_screenshot(self):
        if False:
            return 10
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtCore import QTimer
        self._canvas.show()
        self._canvas.raise_()
        time.sleep(1.5)
        QApplication.processEvents()
        QTimer.singleShot(1000, self._grab_widget_screenshot)
        time.sleep(1.5)
        QApplication.processEvents()

    def _grab_widget_screenshot(self):
        if False:
            i = 10
            return i + 15
        from PyQt5.QtWidgets import QApplication
        screen = QApplication.screenAt(self._canvas.pos())
        screenshot = screen.grabWindow(int(self._canvas.windowHandle().winId()))
        arr = self._qpixmap_to_ndarray(screenshot)
        self._collected_images.append(arr)

    @staticmethod
    def _qpixmap_to_ndarray(pixmap):
        if False:
            for i in range(10):
                print('nop')
        from PyQt5 import QtGui
        import numpy as np
        im = pixmap.toImage().convertToFormat(QtGui.QImage.Format.Format_RGB32)
        size = pixmap.size()
        width = size.width()
        height = size.height()
        im_bits = im.constBits()
        im_bits.setsize(height * width * 4)
        return np.array(im_bits).reshape((height, width, 4))[:, :, 2::-1]

    def _grab_vispy_screenshots(self):
        if False:
            while True:
                i = 10
        os.environ['VISPY_IGNORE_OLD_VERSION'] = 'true'
        self._canvas.events.draw.connect(self.on_draw, position='last')
        with self._canvas as c:
            self._collect_frames(c)

    def _collect_frames(self, canvas, limit=10000):
        if False:
            return 10
        n = 0
        while not self._done and n < limit:
            canvas.update()
            canvas.app.process_events()
            n += 1
        if n >= limit or len(self._frames_to_grab) > 0:
            raise RuntimeError('Could not collect any images')

    def save_frame(self, filename, frame_index=0):
        if False:
            while True:
                i = 10
        imsave(filename, self._collected_images[frame_index])

    def save_animation(self, filename):
        if False:
            print('Hello World!')
        import imageio
        imageio.mimsave(filename, self._collected_images)