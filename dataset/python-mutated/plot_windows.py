import collections
from operator import itemgetter
import functools
import time
import copy
import json
import traceback
import numpy as np
from functools import reduce
import threading
from matplotlib.figure import Figure
import matplotlib
import matplotlib.colors
import matplotlib.widgets
import matplotlib.cm
import matplotlib.ticker
import astropy.units
import vaex
import logging
import vaex.events
import vaex.kld
import vaex.ui.plugin.zoom
import vaex.ui.plugin.vector3d
import vaex.ui.plugin.animation
import vaex.ui.plugin.tasks
import vaex.ui.plugin.transferfunction
import vaex.ui.imageblending
import vaex.ui.colormaps
import vaex.ui.templates
import vaex.promise
import vaex.ui.volumerendering
import vaex.ui.qt as dialogs
from vaex.ui.icons import iconfile
import vaex.vaexfast
from vaex.ui import qt, undo
from vaex.utils import AttrDict
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
from vaex.ui.qt import *
if qt_mayor == 5:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
else:
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
grid_resolutions = [32, 64, 128, 256, 512, 1024]
vector_grid_resolutions = [8, 16, 32, 64, 128, 256]
logger = logging.getLogger('vaex.ui.window')

class Slicer(matplotlib.widgets.Widget):
    """
    """

    def __init__(self, plot_window, axes, canvas, useblit=True, horizOn=False, vertOn=True, radius=0.05, **lineprops):
        if False:
            print('Hello World!')
        self.plot_window = plot_window
        self.canvas = canvas
        self.axes = axes
        self.horizOn = horizOn
        self.vertOn = vertOn
        (xmin, xmax) = axes[-1].get_xlim()
        (ymin, ymax) = axes[-1].get_ylim()
        xmid = 0.5 * (xmin + xmax)
        ymid = 0.5 * (ymin + ymax)
        self.visible = True
        self.useblit = useblit and self.canvas.supports_blit
        self.background = None
        self.needclear = False
        self.radius = radius
        if self.useblit:
            lineprops['animated'] = True
        import matplotlib.patches
        self.ellipse = matplotlib.patches.Ellipse([xmid, ymid], 1, 1, alpha=0.2, visible=False)
        self.update_ellipse(xmid, ymid)
        axes[0].add_patch(self.ellipse)
        if vertOn:
            self.vlines = [ax.axvline(xmid, visible=False, **lineprops) for ax in axes]
        else:
            self.vlines = []
        if horizOn:
            self.hlines = [ax.axhline(ymid, visible=False, **lineprops) for ax in axes]
        else:
            self.hlines = []
        self.connect()

    def connect(self):
        if False:
            while True:
                i = 10
        'connect events'
        self._cidmotion = self.canvas.mpl_connect('motion_notify_event', self.onmove)
        self._ciddraw = self.canvas.mpl_connect('draw_event', self.clear)

    def disconnect(self):
        if False:
            for i in range(10):
                print('nop')
        'disconnect events'
        self.canvas.mpl_disconnect(self._cidmotion)
        self.canvas.mpl_disconnect(self._ciddraw)

    def clear(self, event):
        if False:
            print('Hello World!')
        'clear the cursor'
        if self.useblit:
            self.background = self.canvas.copy_from_bbox(self.canvas.figure.bbox)
        for line in self.vlines + self.hlines:
            line.set_visible(False)
        self.ellipse.set_visible(False)

    def onmove(self, event):
        if False:
            print('Hello World!')
        if event.inaxes is None:
            self.plot_window.slice_none()
            return
        if not self.canvas.widgetlock.available(self):
            return
        self.needclear = True
        if not self.visible:
            return
        if self.vertOn:
            for line in self.vlines:
                line.set_xdata((event.xdata, event.xdata))
                line.set_visible(self.visible)
        if self.horizOn:
            for line in self.hlines:
                line.set_ydata((event.ydata, event.ydata))
                line.set_visible(self.visible)
        self.plot_window.slice_circle(event.xdata, event.ydata, self.radius)
        self.update_ellipse(event.xdata, event.ydata)
        self.ellipse.set_visible(True)
        self._update()

    def update_ellipse(self, x, y):
        if False:
            for i in range(10):
                print('nop')
        (xlim, ylim) = self.plot_window.self.state.ranges_viewport
        width = abs(xlim[1] - xlim[0])
        height = abs(ylim[1] - ylim[0])
        scale = self.radius * 2
        self.ellipse.center = [x, y]
        self.ellipse.width = width * scale
        self.ellipse.height = height * scale

    def _update(self):
        if False:
            for i in range(10):
                print('nop')
        if self.useblit:
            if self.background is not None:
                self.canvas.restore_region(self.background)
            if self.vertOn:
                for (ax, line) in zip(self.axes, self.vlines):
                    ax.draw_artist(line)
            if self.horizOn:
                for (ax, line) in zip(self.axes, self.hlines):
                    ax.draw_artist(line)
            for ax in self.axes:
                ax.draw_artist(self.ellipse)
            self.canvas.blit(self.canvas.figure.bbox)
        else:
            self.canvas.draw_idle()

    def disconnect_events(self):
        if False:
            return 10
        self.ellipse.set_visible(False)
        self.canvas.restore_region(self.background)
        self.canvas.blit(self.canvas.figure.bbox)
        self.disconnect()

class PlotDialog(QtGui.QWidget):
    """
    :type app: VaexApp
    :type current_layer: LayerTable
    :type layers: list[LayerTable]
    """

    def __init__(self, parent, dataset, dimensions, axisnames, app, width=5, height=4, dpi=100, **options):
        if False:
            for i in range(10):
                print('nop')
        super(PlotDialog, self).__init__()
        self.parent_widget = parent
        self.data_panel = parent
        self.options = options
        self.dataset = dataset
        self.datasets = {}
        self.app = app
        self.layers = []
        index = len(self.app.windows)
        self.name = options.get('window_name', '%s-%d' % (self.dataset.name, index))
        self.dimensions = dimensions
        if 'fraction' in self.options:
            dataset.set_active_fraction(float(self.options['fraction']))
        self.state = AttrDict()
        self.state.xlabel = options.get('xlabel')
        self.state.ylabel = options.get('ylabel')
        self.enable_slicing = options.get('enable_slicing', False)
        self.menu_bar = QtGui.QMenuBar(self)
        self.menu_file = QtGui.QMenu('&File', self.menu_bar)
        self.menu_bar.addMenu(self.menu_file)
        self.menu_view = QtGui.QMenu('&View', self.menu_bar)
        self.menu_bar.addMenu(self.menu_view)
        self.menu_mode = QtGui.QMenu('&Mode', self.menu_bar)
        self.menu_bar.addMenu(self.menu_mode)
        self.menu_selection = QtGui.QMenu('&Selection', self.menu_bar)
        self.menu_bar.addMenu(self.menu_selection)
        self.menu_samp = QtGui.QMenu('SAM&P', self.menu_bar)
        self.menu_bar.addMenu(self.menu_samp)
        self.undoManager = parent.undoManager
        self.full_state_history = []
        self.full_state_history_index = -1
        self.full_state_history_change_reason = None
        self.setWindowTitle(self.name)
        self.dataset = dataset
        self.axisnames = axisnames
        if self.dimensions == 3:
            self.resize(800 + 400, 700)
        else:
            self.resize(800 + 150, 700)
        self.plugin_queue_toolbar = []
        self.plugins = [cls(self) for cls in vaex.ui.plugin.PluginPlot.registry if cls.useon(self.__class__)]
        self.plugins_map = {plugin.name: plugin for plugin in self.plugins}
        self.state.aspect = None
        self.state.axis_lock = False
        self.update_counter = 0
        self.t_0 = 0
        self.t_last = 0
        self.slice_radius = 0.05
        self.shortcuts = []
        self.messages = {}
        default_grid_size = 256 if self.dimensions == 2 else 128
        self.state.grid_size = eval(self.options.get('grid_size', str(default_grid_size)))
        self.state.vector_grid_size = eval(self.options.get('vector_grid_size', '16'))
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self)
        self.add_axes()
        self.queue_redraw = Queue('redraw', 5, self.canvas.draw)
        self.queue_replot = Queue('replot', 10, self.plot)

        def pre():
            if False:
                while True:
                    i = 10
            self.queue_replot.cancel()
            for layer in self.layers:
                layer.cancel_tasks()
        self.queue_update = Queue('update', 100, self.update_direct, pre=pre)
        self.layout_main = QtGui.QVBoxLayout()
        self.layout_content = QtGui.QHBoxLayout()
        self.layout_main.setContentsMargins(0, 0, 0, 0)
        self.layout_content.setContentsMargins(0, 0, 0, 0)
        self.layout_main.setSpacing(0)
        self.layout_content.setSpacing(0)
        self.layout_main.addWidget(self.menu_bar)
        self.boxlayout = QtGui.QVBoxLayout()
        self.boxlayout_right = QtGui.QVBoxLayout()
        self.boxlayout.setContentsMargins(0, 0, 0, 0)
        self.boxlayout_right.setContentsMargins(0, 0, 0, 0)
        self.state.ranges_viewport = [None for _ in range(self.dimensions)]
        self.state.range_level_show = None
        self.last_ranges_viewport = None
        self.last_range_level_show = None
        self._plot_event = None
        self.currentModes = None
        self.lastAction = None
        self.beforeCanvas(self.layout_main)
        self.layout_main.addLayout(self.layout_content, 1.0)
        self.layout_plot_region = QtGui.QHBoxLayout()
        self.layout_plot_region.addWidget(self.canvas, 1)
        self.boxlayout.addLayout(self.layout_plot_region, 1)
        self.addToolbar2(self.layout_main)
        self.afterCanvas(self.boxlayout_right)
        self.layout_content.addLayout(self.boxlayout, 1.0)
        self.status_bar = QtGui.QStatusBar(self)
        self.button_cancel = QtGui.QToolButton(self.status_bar)
        self.button_cancel.setText('cancel')
        self.button_cancel.setContentsMargins(0, 0, 0, 0)
        self.status_bar.layout().setContentsMargins(0, 0, 0, 0)
        self.status_bar.layout().setSpacing(0)
        self.progress_bar = QtGui.QProgressBar(self.status_bar)
        self.progress_bar.setMaximum(1000)
        self.progress_bar.setMinimumWidth(100)
        self.progress_bar.setFixedWidth(100)
        self.button_cancel.setEnabled(False)
        self.progress_start_time = time.time()
        self.label_time = QtGui.QLabel('', self.toolbar)
        self.status_bar.addPermanentWidget(self.progress_bar)
        self.status_bar.addPermanentWidget(self.button_cancel)
        self.status_bar.addPermanentWidget(self.label_time)

        def on_click_cancel():
            if False:
                for i in range(10):
                    print('nop')
            self.button_cancel.setEnabled(False)
            for layer in self.layers:
                layer.cancel_tasks()
        self.button_cancel.clicked.connect(on_click_cancel)
        self.layout_main.addWidget(self.status_bar)
        self.layout_content.addLayout(self.boxlayout_right, 0)
        self.setLayout(self.layout_main)
        FigureCanvas.setSizePolicy(self, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.currentMode = None
        self.shortcuts = []
        self.grabGesture(QtCore.Qt.PinchGesture)
        self.grabGesture(QtCore.Qt.PanGesture)
        self.grabGesture(QtCore.Qt.SwipeGesture)
        self.signal_samp_send_selection = vaex.events.Signal('samp send selection')
        self.signal_closed = vaex.events.Signal('close plot window')
        self.signal_plot_finished = vaex.events.Signal('plot finished')
        self.canvas.mpl_connect('resize_event', self.on_resize_event)
        self.canvas.mpl_connect('motion_notify_event', self.onMouseMove)

    def get_label(self, axis=0):
        if False:
            while True:
                i = 10
        if self.layers:
            layer_labels = [layer.state.labels[axis] for layer in self.layers]
            if any(layer_labels):
                return ','.join([label for label in layer_labels if label])
            else:
                return self.get_default_label(axis=axis)
        else:
            return ''

    def get_default_label(self, axis=0):
        if False:
            while True:
                i = 10
        if len(self.layers) == 0:
            return ''
        else:
            first_layer = self.layers[0]
            if axis == 0:
                expr = first_layer.x
            if axis == 1:
                if self.dimensions > 1:
                    expr = first_layer.y
                else:
                    return first_layer.amplitude_label()
            if axis == 2:
                expr = first_layer.z
            label = expr
            unit = first_layer.dataset.unit(expr)
            try:
                output_unit = astropy.units.Unit(first_layer.state.output_units[axis])
                if first_layer.state.output_units[axis] and unit:
                    output_unit.to(unit)
                    unit = output_unit
            except:
                logger.exception('unit error')
            logger.debug('unit: %r', unit)
            if unit is not None:
                label = '%s (%s)' % (label, unit.to_string('latex_inline'))
            return label

    def set_plot_size(self, width, height):
        if False:
            return 10
        self.canvas.setFixedSize(width, height)

    def set_layer_progress(self, layer, fraction):
        if False:
            while True:
                i = 10
        if fraction == 0:
            self.progress_start_time = time.time()
        if fraction == 1:
            delta_time = time.time() - self.progress_start_time
            info = '%.2f sec' % delta_time
            self.label_time.setText(info)
        elif fraction > 0:
            delta_time = time.time() - self.progress_start_time
            estimated = delta_time / fraction - delta_time
            info = 'ETA: %.2f sec' % estimated
            self.label_time.setText(info)
        self.button_cancel.setEnabled(True)
        fraction = self.get_progress_fraction()
        self.progress_bar.setValue(fraction * 1000)
        QtCore.QCoreApplication.instance().processEvents()

    def get_progress_fraction(self):
        if False:
            while True:
                i = 10
        total_fraction = 0
        updating_layers = [layer for layer in self.layers]
        total_fraction = sum([layer.get_progress_fraction() for layer in updating_layers])
        return total_fraction / len(updating_layers)

    def slice_none(self):
        if False:
            for i in range(10):
                print('nop')
        mask = np.ones((self.state.grid_size,) * self.dimensions, dtype=bool)
        layer = self.current_layer
        if layer:
            layer.signal_slice_change.emit(mask, False)

    def slice_circle(self, x, y, radius):
        if False:
            return 10
        logger.debug('slice circle: %r %r', x, y)
        (xlim, ylim) = self.state.ranges_viewport
        width = abs(xlim[1] - xlim[0])
        height = abs(ylim[1] - ylim[0])
        xrel = (x - xlim[0]) / width
        yrel = (y - ylim[0]) / height
        import vaex.utils
        x = vaex.utils.linspace_centers(0, 1, self.state.grid_size)
        y = vaex.utils.linspace_centers(0, 1, self.state.grid_size)
        (x, y) = np.meshgrid(x, y)
        distance = np.sqrt((x - xrel) ** 2 + (y - yrel) ** 2)
        mask = distance < radius
        layer = self.current_layer
        if layer:
            layer.signal_slice_change.emit(mask, False)

    def update_all_layers(self):
        if False:
            for i in range(10):
                print('nop')
        for layer in self.layers:
            layer.flag_needs_update()
        self.queue_update()

    def add_axes(self):
        if False:
            i = 10
            return i + 15
        self.axes = self.fig.add_subplot(111)
        self.axes.xaxis_index = 0
        if self.dimensions > 1:
            self.axes.yaxis_index = 1
        if int(matplotlib.__version__.split('.')[0]) < 3:
            self.axes.hold(True)

    def plug_toolbar(self, callback, order):
        if False:
            for i in range(10):
                print('nop')
        self.plugin_queue_toolbar.append((callback, order))

    def plug_page(self, callback, pagename, pageorder, order):
        if False:
            return 10
        self.plugin_queue_page.append((callback, pagename, pageorder, order))

    def plug_grids(self, callback_define, callback_draw):
        if False:
            while True:
                i = 10
        self.plugin_grids_defines.append(callback_define)
        self.plugin_grids_draw.append(callback_draw)

    def getAxesList(self):
        if False:
            i = 10
            return i + 15
        return [self.axes]

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return '<%s at 0x%x layers=%r>' % (self.__class__.__name__, id(self), self.layers)

    def get_options(self):
        if False:
            return 10
        options = collections.OrderedDict()
        options['grid_size'] = self.state.grid_size
        options['vector_grid_size'] = self.state.vector_grid_size
        options['ranges_viewport'] = self.state.ranges_viewport
        options['aspect'] = self.state.aspect
        layer = self.current_layer
        if layer is not None:
            options['layer'] = layer.get_options()
        options = copy.deepcopy(options)
        return dict(options)

    def apply_options(self, options, update=True):
        if False:
            for i in range(10):
                print('nop')
        recognize = 'ranges_viewport  aspect'.split()
        for key in recognize:
            if key in list(options.keys()):
                value = options[key]
                setattr(self, key, copy.copy(value))
                if key == 'aspect':
                    self.action_aspect_lock_one.setChecked(bool(value))
        for plugin in self.plugins:
            plugin.apply_options(options)
        for key in list(options.keys()):
            if key not in recognize:
                logger.error('option %s not recognized, ignored' % key)
        layer = self.current_layer
        if layer is not None:
            layer.apply_options(options['layer'], update=False)
        if update:
            self.queue_update()

    def load_options(self, name):
        if False:
            for i in range(10):
                print('nop')
        self.plugins_map['favorites'].load_options(name, update=False)

    def select_layer(self, index):
        if False:
            return 10
        self.layer_selection.setCurrentIndex(index + 1)

    def add_layer(self, expressions, dataset=None, name=None, **options):
        if False:
            for i in range(10):
                print('nop')
        if dataset is None:
            dataset = self.dataset
        if name is None:
            name = options.get('layer_name', 'Layer: ' + str(len(self.layers) + 1))
        self.datasets[dataset.path] = dataset
        ranges = copy.deepcopy(self.state.ranges_viewport)
        logger.debug('adding layer {name} with expressions {expressions} for dataset {dataset} and options {options}'.format(**locals()))
        if len(self.layers) > 0:
            first_layer = self.layers[0]
            assert len(expressions) == first_layer.dimensions
            for i in range(self.dimensions):
                if ranges[i] is None and first_layer.state.ranges_grid[i] is not None:
                    ranges[i] = copy.copy(first_layer.state.ranges_grid[i])
        layer = vaex.ui.layers.LayerTable(self, name, dataset, expressions, self.axisnames, options, self.fig, self.canvas, ranges)
        self.layers.append(layer)
        layer.build_widget_qt(self.widget_layer_stack)
        self.widget_layer_stack.addWidget(layer.widget)
        self.queue_history_change('added layer: ' + name)
        layer.widget.setVisible(False)
        self.layer_selection.addItem(name + '(%s)' % dataset.name)
        self.select_layer(len(self.layers) - 1)

        def on_expression_change(layer, axis_index, expression):
            if False:
                print('Hello World!')
            if not self.state.axis_lock:
                self.state.ranges_viewport[axis_index] = None
            self.compute()
            error_text = self.dataset.executor.execute()
            if error_text:
                dialog_error(self, 'Error in expression', 'Error: ' + error_text)

        def on_plot_dirty(layer=None):
            if False:
                print('Hello World!')
            logger.debug('received signal plot dirty, layer=%r' % layer)
            self.queue_replot()
        layer.signal_expression_change.connect(on_expression_change)
        layer.signal_plot_dirty.connect(on_plot_dirty)
        layer.signal_plot_update.connect(self.queue_update)
        if 'options' in options:
            assert self.current_layer == layer
            self.load_options(options['options'])
        self.queue_update(layer=layer)
        logger.debug('added layer')
        return layer

    def _wait(self):
        if False:
            print('Hello World!')
        'Used for unittesting to make sure the plots are all done'
        logger.debug('will wait for last plot to finish')
        self._plot_event = threading.Event()
        self.queue_update._wait()
        self.queue_replot._wait()
        self.queue_redraw._wait()
        qt_app = QtCore.QCoreApplication.instance()
        sleep = 10
        while not self._plot_event.is_set():
            logger.debug('waiting for last plot to finish')
            qt_app.processEvents()
            QtTest.QTest.qSleep(sleep)
        logger.debug('waiting for plot finished')

    def plot_to_png(self, filename=None):
        if False:
            i = 10
            return i + 15
        if filename is None:
            import tempfile
            (handle, filename) = tempfile.mkstemp('.png')
            logger.debug('write to %s' % filename)
        self.fig.savefig(filename)
        return filename

    def on_resize_event(self, event):
        if False:
            i = 10
            return i + 15
        if not self.action_mini_mode_ultra.isChecked():
            logger.debug('resize event')
            self.fig.tight_layout()
            self.queue_redraw()

    def event(self, event):
        if False:
            print('Hello World!')
        if isinstance(event, QtGui.QGestureEvent):
            for gesture in event.activeGestures():
                if isinstance(gesture, QtGui.QPinchGesture):
                    center = gesture.centerPoint()
                    (x, y) = (center.x(), center.y())
                    geometry = self.canvas.geometry()
                    if geometry.contains(x, y):
                        rx = x - geometry.x()
                        ry = y - geometry.y()
                        axes_list = [ax for ax in self.getAxesList() if ax.contains_point((rx, geometry.height() - 1 - ry))]
                        if len(axes_list) > 0:
                            axes = axes_list[0]
                            transform = axes.transData.inverted().transform
                            (x_data, y_data) = transform([rx, geometry.height() - 1 - ry])
                            if gesture.lastScaleFactor() != 0:
                                scale = gesture.scaleFactor() / gesture.lastScaleFactor()
                            else:
                                scale = gesture.scaleFactor()
                            scale = 1.0 / scale
                            self.zoom(scale, axes, x_data, y_data)
            return True
        else:
            return super(PlotDialog, self).event(event)

    def closeEvent(self, event):
        if False:
            i = 10
            return i + 15
        for layer in self.layers:
            layer.removed()
        for plugin in self.plugins:
            plugin.clean_up()
        super(PlotDialog, self).closeEvent(event)
        self.signal_closed.emit(self)

    def getExpressionList(self):
        if False:
            return 10
        return self.dataset.column_names

    def add_pages(self, toolbox):
        if False:
            for i in range(10):
                print('nop')
        pass

    def fill_menu_layer_new(self):
        if False:
            while True:
                i = 10
        self.menu_layer_new.clear()
        for dataset in self.data_panel.dataset_list:
            menu_dataset = QtGui.QMenu(dataset.name, self.menu_layer_new)
            self.menu_layer_new.addMenu(menu_dataset)
            for column1 in dataset.get_column_names(virtual=True):
                if self.dimensions == 1:
                    action_col1 = QtGui.QAction(column1, menu_dataset)
                    menu_dataset.addAction(action_col1)

                    def add_layer_1(ignore=None, column1=column1, dataset=dataset):
                        if False:
                            for i in range(10):
                                print('nop')
                        self.add_layer([column1], dataset=dataset)
                        dataset.executor.execute()
                    action_col1.triggered.connect(add_layer_1)
                else:
                    menu_col1 = QtGui.QMenu(column1, menu_dataset)
                    menu_dataset.addMenu(menu_col1)
                    for column2 in dataset.get_column_names(virtual=True):
                        if self.dimensions == 2:
                            action_col2 = QtGui.QAction(column2, menu_dataset)
                            menu_col1.addAction(action_col2)

                            def add_layer_2(_ignore=None, column1=column1, column2=column2, dataset=dataset):
                                if False:
                                    i = 10
                                    return i + 15
                                self.add_layer([column1, column2], dataset=dataset)
                                dataset.executor.execute()
                            action_col2.triggered.connect(add_layer_2)
                        else:
                            pass

    def remove_layer(self, layer=None, layer_index=None):
        if False:
            i = 10
            return i + 15
        if layer is not None:
            layer = layer
        elif layer_index is not None:
            layer = self.layers[layer_index]
        else:
            layer = self.current_layer
        logger.debug('remove layer: %r' % layer)
        if layer is not None:
            index = self.layers.index(layer)
            layer.removed()
            self.layers.remove(layer)
            self.layer_selection.removeItem(index + 1)
            self.widget_layer_stack.removeWidget(layer.widget)
            self.queue_history_change('Remove layer: %s' % layer.name)
            self.queue_push_full_state()
            self.plot()

    def afterCanvas(self, layout):
        if False:
            for i in range(10):
                print('nop')
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.layer_box = QtGui.QGroupBox('Layers', self)
        self.layout_layer_box = QtGui.QVBoxLayout()
        self.layout_layer_box.setSpacing(0)
        self.layout_layer_box.setContentsMargins(0, 0, 0, 0)
        self.layer_box.setLayout(self.layout_layer_box)
        self.layout_layer_buttons = QtGui.QHBoxLayout()
        self.button_layer_new = QtGui.QToolButton(self)
        self.button_layer_new.setIcon(QtGui.QIcon(iconfile('layer--plus')))
        self.button_layer_new.setPopupMode(QtGui.QToolButton.InstantPopup)
        self.button_layer_new.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        toolbuttonSizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Fixed)
        self.button_layer_new.setSizePolicy(toolbuttonSizePolicy)
        self.button_layer_new.setText('add')
        self.button_layer_new.setEnabled(self.dimensions < 3)
        self.menu_layer_new = QtGui.QMenu()
        self.fill_menu_layer_new()
        self.button_layer_new.setMenu(self.menu_layer_new)
        self.button_layer_delete = QtGui.QToolButton(self)
        self.button_layer_delete.setIcon(QtGui.QIcon(iconfile('layer--minus')))
        self.button_layer_delete.setText('remove')
        self.button_layer_delete.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.button_layer_delete.setSizePolicy(toolbuttonSizePolicy)
        self.button_layer_delete.setEnabled(self.dimensions < 3)

        def on_layer_remove(_ignore=None):
            if False:
                i = 10
                return i + 15
            logger.debug('remove layer')
            self.remove_layer()
        self.button_layer_delete.clicked.connect(on_layer_remove)
        self.layout_layer_buttons.addWidget(self.button_layer_new, 0)
        self.layout_layer_buttons.addWidget(self.button_layer_delete, 0)
        self.layer_selection = QtGui.QComboBox(self)
        self.layer_selection.addItems(['Layer controls'])
        self.layout_layer_box.addLayout(self.layout_layer_buttons)
        self.layout_layer_box.addWidget(self.layer_selection)
        self.current_layer = None

        def onSwitchLayer(index):
            if False:
                for i in range(10):
                    print('nop')
            logger.debug('switch to layer: %r %r' % (index, self.layers))
            layer_index = index - 1
            if index == 0:
                self.current_layer = None
                for layer in self.layers:
                    layer_control_widget = layer.grab_layer_control(self.frame_layer_controls)
                    self.layout_frame_layer_controls.addWidget(layer_control_widget)
            else:
                self.layers[layer_index].release_layer_control(self.frame_layer_controls)
                self.current_layer = self.layers[index - 1]
            self.update_favorite_selections()
            self.widget_layer_stack.setCurrentIndex(index)
        self.layer_selection.currentIndexChanged.connect(onSwitchLayer)
        self.bottomFrame = QtGui.QFrame(self)
        layout.addWidget(self.bottomFrame, 0)
        self.bottom_layout = QtGui.QVBoxLayout()
        self.bottom_layout.setContentsMargins(0, 0, 0, 0)
        self.bottom_layout.setSpacing(0)
        self.bottomFrame.setLayout(self.bottom_layout)
        self.bottom_layout.addWidget(self.layer_box)
        self.widget_layer_stack = QtGui.QStackedWidget(self)
        self.bottom_layout.addWidget(self.widget_layer_stack)
        self.frame_layer_controls = QtGui.QGroupBox('Layer controls', self.widget_layer_stack)
        self.layout_frame_layer_controls = QtGui.QVBoxLayout(self.frame_layer_controls)
        self.layout_frame_layer_controls.setAlignment(QtCore.Qt.AlignTop)
        self.frame_layer_controls.setLayout(self.layout_frame_layer_controls)
        self.widget_layer_stack.addWidget(self.frame_layer_controls)
        self.frame_layer_controls_result = QtGui.QGroupBox('Layer result', self.frame_layer_controls)
        self.layout_frame_layer_controls_result = QtGui.QGridLayout()
        self.frame_layer_controls_result.setLayout(self.layout_frame_layer_controls_result)
        self.layout_frame_layer_controls_result.setSpacing(0)
        self.layout_frame_layer_controls_result.setContentsMargins(0, 0, 0, 0)
        self.layout_frame_layer_controls.addWidget(self.frame_layer_controls_result)
        row = 0
        attr_name = 'layer_brightness'
        self.layer_brightness = 1.0
        self.slider_layer_brightness = Slider(self.frame_layer_controls_result, 'brightness', 10 ** (-1), 10 ** 1, 1000, attrgetter(self, attr_name), attrsetter(self, attr_name), uselog=True, update=self.plot)
        row = self.slider_layer_brightness.add_to_grid_layout(row, self.layout_frame_layer_controls_result)
        attr_name = 'layer_gamma'
        self.layer_gamma = 1.0
        self.slider_layer_gamma = Slider(self.frame_layer_controls_result, 'gamma', 10 ** (-1), 10 ** 1, 1000, attrgetter(self, attr_name), attrsetter(self, attr_name), uselog=True, update=self.plot)
        row = self.slider_layer_gamma.add_to_grid_layout(row, self.layout_frame_layer_controls_result)
        self.blend_modes = list(vaex.ui.imageblending.modes)
        self.blend_mode = self.blend_modes[0]
        self.option_layer_blend_mode = Option(self.frame_layer_controls_result, 'blend', self.blend_modes, getter=attrgetter(self, 'blend_mode'), setter=attrsetter(self, 'blend_mode'), update=self.plot)
        row = self.option_layer_blend_mode.add_to_grid_layout(row, self.layout_frame_layer_controls_result)
        self.background_colors = ['auto', 'white', 'black']
        self.background_color = self.background_colors[0]
        self.option_layer_background_color = Option(self.frame_layer_controls_result, 'background', self.background_colors, getter=attrgetter(self, 'background_color'), setter=attrsetter(self, 'background_color'), update=self.plot)
        row = self.option_layer_background_color.add_to_grid_layout(row, self.layout_frame_layer_controls_result)

    def add_shortcut(self, action, key):
        if False:
            while True:
                i = 10

        def trigger(action):
            if False:
                return 10

            def call(action=action):
                if False:
                    while True:
                        i = 10
                action.toggle()
                action.trigger()
            return call
        if action.isEnabled():
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            shortcut.activated.connect(trigger(action))
            self.shortcuts.append(shortcut)

    def checkUndoRedo(self):
        if False:
            print('Hello World!')
        can_undo = self.full_state_history_index >= 1
        can_redo = self.full_state_history_index < len(self.full_state_history) - 1
        self.action_undo.setEnabled(can_undo)
        if can_undo:
            (reason, state) = self.full_state_history[self.full_state_history_index]
            self.action_undo.setToolTip('Undo: ' + reason)
        self.action_redo.setEnabled(can_redo)
        if can_redo:
            (reason, state) = self.full_state_history[self.full_state_history_index + 1]
            self.action_redo.setToolTip('Redo: ' + reason)

    def onActionUndo(self):
        if False:
            print('Hello World!')
        logger.debug('undo')
        self.full_state_history_index -= 1
        (reason, state) = self.full_state_history[self.full_state_history_index]
        logger.debug('restore state')
        self.restore_full_state(state)
        logger.debug('state restored')
        self.checkUndoRedo()
        logger.debug('undo/redo buttons checked')

    def onActionRedo(self):
        if False:
            return 10
        logger.debug('redo')
        self.full_state_history_index += 1
        (reason, state) = self.full_state_history[self.full_state_history_index]
        logger.debug('restore state')
        self.restore_full_state(state)
        logger.debug('state restored')
        self.checkUndoRedo()
        logger.debug('undo/redo buttons checked')

    def onMouseMove(self, event):
        if False:
            while True:
                i = 10
        (x, y) = (event.xdata, event.ydata)
        if x is not None:
            extra_text = self.getExtraText(x, y)
            if extra_text:
                self.message('x,y=%5.4e,%5.4e %s' % (x, y, extra_text), index=0)
            else:
                self.message('x,y=%5.4e,%5.4e' % (x, y), index=0)
        else:
            self.message(None)

    def getExtraText(self, x, y):
        if False:
            return 10
        layer = self.current_layer
        if hasattr(layer, 'amplitude_grid'):
            amplitude = layer.amplitude_grid
            if len(amplitude.shape) == 1:
                N = amplitude.shape[0]
                if layer.state.ranges_grid[0] is not None:
                    (xmin, xmax) = layer.state.ranges_grid[0]
                    index = (x - xmin) / (xmax - xmin) * N
                    if index >= 0 and index < N:
                        index = int(index)
                        return 'value = %f' % amplitude[index]
            if len(amplitude.shape) == 2:
                (Nx, Ny) = amplitude.shape
                if layer.state.ranges_grid[0] is not None and layer.state.ranges_grid[1] is not None:
                    (xmin, xmax) = layer.state.ranges_grid[0]
                    (ymin, ymax) = layer.state.ranges_grid[1]
                    xindex = (x - xmin) / (xmax - xmin) * Nx
                    yindex = (y - ymin) / (ymax - ymin) * Ny
                    if xindex >= 0 and xindex < Nx and (yindex >= 0) and (yindex < Nx):
                        return 'value = %f' % amplitude[int(yindex), int(xindex)]

    def message(self, text, index=0):
        if False:
            print('Hello World!')
        if text is None:
            if index in self.messages:
                del self.messages[index]
        else:
            self.messages[index] = text
        text = ''
        keys = list(self.messages.keys())
        keys.sort()
        text_parts = [self.messages[key] for key in keys]
        self.status_bar.showMessage(' | '.join(text_parts))

    def beforeCanvas(self, layout):
        if False:
            while True:
                i = 10
        self.addToolbar(layout)

    def setMode(self, action, force=False):
        if False:
            while True:
                i = 10
        logger.debug('set mode %r %r %r' % (action, action.text(), action.isChecked()))
        if not action.isEnabled():
            logger.error('action selected that was disabled: %r' % action)
            self.setMode(self.lastAction)
            return
        if not action.isChecked():
            logger.debug('ignore action')
        else:
            self.lastAction = action
            axes_list = self.getAxesList()
            if self.currentModes is not None:
                logger.debug('disconnect %r' % (self.currentModes,))
                for mode in self.currentModes:
                    mode.disconnect_events()
                    mode.active = False
            useblit = True
            if action == self.action_move:
                self.currentModes = [Mover(self, axes) for axes in axes_list]
            if action == self.action_pick:
                layer = self.current_layer
                if layer is not None:
                    hasx = True
                    hasy = len(layer.state.expressions) > 1
                    self.currentModes = [matplotlib.widgets.Cursor(axes, hasy, hasx, color='red', linestyle='dashed', useblit=useblit) for axes in axes_list]
                    for cursor in self.currentModes:

                        def onmove(event, current=cursor, cursors=self.currentModes):
                            if False:
                                for i in range(10):
                                    print('nop')
                            if event.inaxes:
                                for other_cursor in cursors:
                                    if current != other_cursor:
                                        other_cursor.onmove(event)
                        cursor.connect_event('motion_notify_event', onmove)
                    if hasx and hasy:
                        for mode in self.currentModes:
                            mode.connect_event('button_press_event', self.onPickXY)
                    elif hasx:
                        for mode in self.currentModes:
                            mode.connect_event('button_press_event', self.onPickX)
                    elif hasy:
                        for mode in self.currentModes:
                            mode.connect_event('button_press_event', self.onPickY)
                if useblit:
                    self.canvas.draw()
            if action == self.action_xrange:
                logger.debug('setting last select action to xrange')
                self.lastActionSelect = self.action_xrange
                self.currentModes = [matplotlib.widgets.SpanSelector(axes, functools.partial(self.onSelectX, axes=axes), 'horizontal', useblit=useblit) for axes in axes_list]
                if useblit:
                    self.canvas.draw()
            if action == self.action_yrange:
                logger.debug('setting last select action to yrange')
                self.lastActionSelect = self.action_yrange
                self.currentModes = [matplotlib.widgets.SpanSelector(axes, functools.partial(self.onSelectY, axes=axes), 'vertical', useblit=useblit) for axes in axes_list]
                if useblit:
                    self.canvas.draw()
            if action == self.action_lasso:
                logger.debug('setting last select action to lasso')
                self.lastActionSelect = self.action_lasso
                self.currentModes = [matplotlib.widgets.LassoSelector(axes, functools.partial(self.onSelectLasso, axes=axes)) for axes in axes_list]
                if useblit:
                    self.canvas.draw()
            if action == self.action_select_rectangle:
                logger.debug('setting last select action to rect')
                self.lastActionSelect = self.action_select_rectangle
                self.currentModes = [matplotlib.widgets.RectangleSelector(axes, functools.partial(self.on_select_rectangle, axes=axes), spancoords='data') for axes in axes_list]
                if useblit:
                    self.canvas.draw()
            if action == self.action_slice:
                logger.debug('setting mode to slice')
                self.currentModes = [Slicer(self, [axes], canvas=self.canvas, horizOn=False, vertOn=False, radius=self.slice_radius) for axes in axes_list]
                if useblit:
                    self.canvas.draw()
            for plugin in self.plugins:
                logger.debug('plugin %r %r.setMode' % (plugin, plugin.name))
                plugin.setMode(action)
        self.syncToolbar()

    def onPickX(self, event):
        if False:
            print('Hello World!')
        (x, y) = (event.xdata, event.ydata)
        self.selected_point = None
        layer = self.current_layer
        axes = event.inaxes
        logger.debug('pickx %r %r' % (layer, axes))
        if layer is not None and axes is not None:
            layer.coordinates_picked_row = None
            promise = layer.subspace.nearest([x])
            layer.dataset.executor.execute()

            def set(value):
                if False:
                    while True:
                        i = 10
                if value is None:
                    logger.error('could not find nearest')
                else:
                    (index, distance, point) = value
                    layer.dataset.set_current_row(index)
            promise.then(set).end()
            self.setMode(self.lastAction)

    def onPickY(self, event):
        if False:
            i = 10
            return i + 15
        (x, y) = (event.xdata, event.ydata)
        self.selected_point = None
        layer = self.current_layer
        axes = event.inaxes
        logger.debug('pickx %r %r' % (layer, axes))
        if layer is not None and axes is not None:
            layer.coordinates_picked_row = None
            promise = layer.subspace.nearest([y])
            layer.dataset.executor.execute()

            def set(value):
                if False:
                    for i in range(10):
                        print('nop')
                if value is None:
                    logger.error('could not find nearest')
                else:
                    (index, distance, point) = value
                    layer.dataset.set_current_row(index)
            promise.then(set).end()
            self.setMode(self.lastAction)

    def onPickXY(self, event):
        if False:
            return 10
        (x, y) = (event.xdata, event.ydata)
        wx = self.state.ranges_viewport[0][1] - self.state.ranges_viewport[0][0]
        wy = self.state.ranges_viewport[1][1] - self.state.ranges_viewport[1][0]
        (x, y) = (event.xdata, event.ydata)
        self.selected_point = None
        layer = self.current_layer
        axes = event.inaxes
        logger.debug('pickx %r %r' % (layer, axes))
        if layer is not None and axes is not None:
            layer.coordinates_picked_row = None
            promise = layer.subspace.nearest([x, y], metric=[1.0 / wx, 1.0 / wy])
            layer.dataset.executor.execute()

            def set(value):
                if False:
                    while True:
                        i = 10
                if value is None:
                    logger.error('could not find nearest')
                else:
                    (index, distance, point) = value
                    layer.dataset.set_current_row(index)
            promise.then(set).end()
        return

    def select(self, name, vmin, vmax):
        if False:
            print('Hello World!')
        values = [vmin, vmax]
        (vmin, vmax) = (min(values), max(values))
        for layer in self.active_layers():
            expr = getattr(layer, name)
            boolean_expression = '((%s) >= %f) & ((%s) < %f)' % (expr, vmin, expr, vmax)
            logger.debug('expression: %s', boolean_expression)
            layer.dataset.select(boolean_expression, self.select_mode)

    def onSelectX(self, xmin, xmax, axes):
        if False:
            print('Hello World!')
        self.select('x', xmin, xmax)

    def onSelectY(self, ymin, ymax, axes):
        if False:
            return 10
        self.select('y', ymin, ymax)

    def onSelectLasso(self, vertices, axes):
        if False:
            print('Hello World!')
        (x, y) = np.array(vertices).T
        x = np.ascontiguousarray(x, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        for layer in self.active_layers():
            layer.dataset.select_lasso(layer.state.expressions[axes.xaxis_index], layer.state.expressions[axes.yaxis_index], x, y, mode=self.select_mode)
            meanx = x.mean()
            meany = y.mean()
            mask = layer.dataset.mask
            self.checkUndoRedo()
            self.queue_update()
        return

    def on_select_rectangle(self, pos1, pos2, axes):
        if False:
            return 10
        layer = self.current_layer
        if layer is not None:
            xmin = min(pos1.xdata, pos2.xdata)
            xmax = max(pos1.xdata, pos2.xdata)
            ymin = min(pos1.ydata, pos2.ydata)
            ymax = max(pos1.ydata, pos2.ydata)
            args = (layer.x, xmin, layer.x, xmax, layer.y, ymin, layer.y, ymax)
            expression = '((%s) >= %f) & ((%s) <= %f) & ((%s) >= %f) & ((%s) <= %f)' % args
            logger.debug('rectangle selection using expression: %r' % expression)
            layer.dataset.select(expression, mode=self.select_mode)
            self.checkUndoRedo()
            self.queue_update()
        return

    def set_ranges(self, axis_indices, ranges_viewport=None, range_level=None, add_to_history=False, reason=None):
        if False:
            return 10
        if add_to_history:
            action = undo.ActionZoom(self.undoManager, reason or 'Change in ranges', self.set_ranges, list(range(self.dimensions)), copy.deepcopy(self.state.ranges_viewport), copy.deepcopy(self.state.range_level_show), axis_indices, ranges_viewport=self.state.ranges_viewport, range_level_show=range_level)
            self.checkUndoRedo()
        logger.debug('set axis/self.state.ranges_viewport: %r / %r' % (axis_indices, self.state.ranges_viewport))
        if axis_indices is None:
            for axis_index in range(self.dimensions):
                self.state.ranges_viewport[axis_index] = None
                for layer in self.layers:
                    layer.state.ranges_grid[axis_index] = None
        else:
            for (i, axis_index) in enumerate(axis_indices):
                if self.state.ranges_viewport:
                    self.state.ranges_viewport[axis_index] = ranges_viewport[i]
                    for layer in self.layers:
                        if self.state.ranges_viewport[i] is not None:
                            layer.set_range(self.state.ranges_viewport[i][0], ranges_viewport[i][1], axis_index)
                        else:
                            layer.set_range(None, None, axis_index)
                i
        logger.debug('set range_level: %r' % (range_level,))
        self.state.range_level_show = range_level
        if len(axis_indices) > 0:
            self.check_aspect(axis_indices[0])
        self.update_all_layers()

    def update_plot(self):
        if False:
            while True:
                i = 10
        self.update_direct()

    def update_direct(self, layer=None):
        if False:
            return 10
        logger.info('update plot: ranges_viewport=%r' % (self.state.ranges_viewport,))
        if layer:
            logger.info('only update layer %r (index %d)' % (layer, self.layers.index(layer)))
            layers = [layer]
        else:
            logger.info('updating all layers')
            layers = self.layers
        layers = [layer for layer in self.layers if layer.get_needs_update()]
        if not layers:
            logger.error('update requested while no layer needs it')
        timelog('begin computation', reset=True)
        promises = [layer.add_tasks_ranges() for layer in layers]
        executors = list(set([layer.dataset.executor for layer in layers]))
        try:
            for executor in executors:
                executor.execute()
        except SyntaxError as e:
            msg = '%s: %r' % (e.args[0], e.args[1][3])
            qt.dialog_error(self, 'Syntax error', 'Syntax error: %s ' % msg)
        except KeyError as e:
            msg = e.args[0]
            qt.dialog_error(self, 'Unknown variable', 'Unknown variable or column: %s ' % msg)
        promise_ranges_done = vaex.promise.listPromise(promises)
        promise_ranges_done.then(self._update_step2, self.on_error_or_cancel).end()
        logger.debug('waiting for promises %r to finish', promises)

    def on_error_or_cancel(self, error):
        if False:
            return 10
        logger.exception('error occured: %r', error, exc_info=error)
        traceback.print_exc()
        raise error

    def _update_step2(self, layers):
        if False:
            print('Hello World!')
        "Each layer has it's own ranges_grid computed now, unless something went wrong\n        But all layers are shown with the same ranges (self.state.ranges_viewport)\n        If any of the ranges is None, take the min/max of each layer\n        "
        logger.info('done with ranges, now update step2 for layers: %r', layers)
        for dimension in range(self.dimensions):
            if self.state.ranges_viewport[dimension] is None:
                vmin = min([layer.state.ranges_grid[dimension][0] for layer in layers])
                vmax = max([layer.state.ranges_grid[dimension][1] for layer in layers])
                self.state.ranges_viewport[dimension] = [vmin, vmax]
        logger.debug('ranges before aspect check: %r', self.state.ranges_viewport)
        self.check_aspect(0)
        logger.debug('ranges after aspect check: %r', self.state.ranges_viewport)
        for layer in layers:
            for d in range(layer.dimensions):
                layer.set_range(self.state.ranges_viewport[d][0], self.state.ranges_viewport[d][1], d)
        promises = [layer.add_tasks_histograms() for layer in layers]
        executors = list(set([layer.dataset.executor for layer in layers]))
        for executor in executors:
            executor.execute()
        promises_histograms_done = vaex.promise.listPromise(promises)
        promises_histograms_done.then(self._update_step3, self.on_error_or_cancel).end()

    def _update_step3(self, layers):
        if False:
            i = 10
            return i + 15
        logger.info('done with histograms, now update step3, layers = %r' % layers)
        if self.state.range_level_show is None:
            self.calculate_range_level_show()
        timelog('computation done')
        self.push_full_state()
        self.queue_replot()

    def queue_history_change(self, reason):
        if False:
            print('Hello World!')
        self.full_state_history_change_reason = reason

    def get_full_state(self):
        if False:
            while True:
                i = 10
        return {'window': copy.deepcopy(dict(self.state)), 'layers': [copy.deepcopy(dict(layer.state)) for layer in self.layers]}

    def push_full_state(self):
        if False:
            i = 10
            return i + 15
        if not self.full_state_history:
            self.full_state_history.append(('initial state', self.get_full_state()))
            self.full_state_history_index += 1
        elif self.full_state_history_change_reason is not None:
            self.full_state_history = self.full_state_history[:self.full_state_history_index + 1]
            state = self.full_state_history.append((self.full_state_history_change_reason, self.get_full_state()))
            self.full_state_history_index += 1
            self.full_state_history_change_reason = None
        self.checkUndoRedo()

    def queue_push_full_state(self):
        if False:
            i = 10
            return i + 15
        QtCore.QTimer.singleShot(1, self.push_full_state)

    def restore_full_state(self, state):
        if False:
            print('Hello World!')
        layer_states = [AttrDict(copy.deepcopy(layer)) for layer in state['layers']]
        logger.debug('removing possible layers')
        while len(layer_states) < len(self.layers):
            logger.debug('removing layer: %r', self.layers[len(self.layers) - 1])
            self.remove_layer(layer_index=len(self.layers) - 1)
        for (layer_state, layer) in zip(layer_states, self.layers):
            logger.debug('restoring state of layer %r to %r', layer, layer_state)
            layer.restore_state(layer_state)
        leftover_layer_states = layer_states[len(self.layers):]
        for layer_state in leftover_layer_states:
            layer = self.add_layer(expressions=layer_state.expressions, dataset=self.datasets[layer_state.dataset_path], name=layer_state.name)
            logger.debug('restoring (left over) state of layer %r to %r', layer, layer_state)
            layer.restore_state(layer_state)
        logger.debug('queue history change')
        self.queue_history_change(None)
        self.state = AttrDict(copy.deepcopy(state['window']))
        logger.debug('updating gui')
        self.action_axes_lock.setChecked(bool(self.state.axis_lock))
        self.action_aspect_lock_one.setChecked(bool(self.state.aspect))
        self.action_resolution_list[grid_resolutions.index(self.state.grid_size)].setChecked(True)
        self.action_resolution_vector_list[vector_grid_resolutions.index(self.state.vector_grid_size)].setChecked(True)
        logger.debug('update all layers')
        self.update_all_layers()

    def calculate_range_level_show(self):
        if False:
            print('Hello World!')
        layers = [layer for layer in self.layers if layer.range_level is not None]
        if layers:
            for layer in layers:
                logger.debug('layer %r has range_level %r' % (layer, layer.range_level))
            vmin = min([layer.range_level[0] for layer in layers])
            vmax = max([layer.range_level[1] for layer in layers])
            self.state.range_level_show = [vmin, vmax]
        logger.debug('range_level_show = %r' % (self.state.range_level_show,))

    def set_range(self, min, max, dimension=0):
        if False:
            while True:
                i = 10
        was_equal = self.state.ranges_viewport[dimension] is not None and list(self.state.ranges_viewport[dimension]) == [min, max]
        dimension_names = 'xyz'
        dim_name = dimension_names[dimension]
        action = undo.ActionZoom(self.undoManager, 'change range in dimension %s' % dim_name, self.set_ranges, list(range(self.dimensions)), copy.deepcopy(self.state.ranges_viewport), copy.deepcopy(self.state.range_level_show), [dimension], ranges_viewport=[[min, max]])
        action.do()
        self.checkUndoRedo()
        return was_equal

    def zoom(self, factor, axes, x=None, y=None, delay=300, *args):
        if False:
            return 10
        logger.info('zooming at location %r, with a factor of %r', (x, y), factor)
        if self.last_ranges_viewport is None:
            self.last_ranges_viewport = copy.deepcopy(self.state.ranges_viewport)
        if self.last_range_level_show is None:
            self.last_range_level_show = copy.deepcopy(self.state.range_level_show)
        (xmin, xmax) = axes.get_xlim()
        width = xmax - xmin
        if x is None:
            x = xmin + width / 2
        fraction = (x - xmin) / width
        range_level_show = None
        ranges_viewport = []
        ranges = []
        axis_indices = []
        ranges_viewport.append([x - width * fraction * factor, x + width * (1 - fraction) * factor])
        axis_indices.append(axes.xaxis_index)
        (ymin, ymax) = axes.get_ylim()
        height = ymax - ymin
        if y is None:
            y = ymin + height / 2
        fraction = (y - ymin) / height
        (ymin_show, ymax_show) = (y - height * fraction * factor, y + height * (1 - fraction) * factor)
        if len(self.state.ranges_viewport) == 1:
            range_level_show = (ymin_show, ymax_show)
            if QtGui.QApplication.keyboardModifiers() == QtCore.Qt.AltModifier or QtGui.QApplication.keyboardModifiers() == QtCore.Qt.ControlModifier:
                range_level_show = (ymin, ymax)
            else:
                range_level_show = (ymin_show, ymax_show)
        else:
            ranges_viewport.append([ymin_show, ymax_show])
            axis_indices.append(axes.yaxis_index)
        reason = 'zoom ' + ('out' if factor > 1 else 'in')
        self.queue_history_change(reason)

        def delayed_zoom():
            if False:
                return 10
            action = undo.ActionZoom(self.undoManager, reason, self.set_ranges, list(range(self.dimensions)), self.last_ranges_viewport, self.last_range_level_show, axis_indices, ranges_viewport=ranges_viewport, range_level_show=range_level_show)
            self.last_ranges_viewport = None
            self.last_range_level_show = None
            action.do()
            self.checkUndoRedo()
        for layer in self.layers:
            layer.flag_needs_update()
        self.queue_update(delayed_zoom, delay=delay)
        if 1:
            self.check_aspect(axes.xaxis_index)
            if self.dimensions in [2, 3]:
                self.state.ranges_viewport[axes.xaxis_index] = list(ranges_viewport[0])
                self.state.ranges_viewport[axes.yaxis_index] = list(ranges_viewport[1])
                first_layer = None
                if len(self.layers):
                    first_layer = self.layers[0]
                for ax in self.getAxesList():
                    ax.set_xlim(self.state.ranges_viewport[ax.xaxis_index])
                    ax.set_ylim(self.state.ranges_viewport[ax.yaxis_index])
            if self.dimensions == 1:
                self.state.ranges_viewport[axis_indices[0]] = list(ranges_viewport[0])
                self.state.range_level_show = list(range_level_show)
                axes.set_xlim(self.state.ranges_viewport[0])
                axes.set_ylim(self.state.range_level_show)
            self.queue_redraw()

    def onActionSaveFigure(self, *ignore_args):
        if False:
            i = 10
            return i + 15
        filetypes = dict(self.fig.canvas.get_supported_filetypes())
        pngtype = [('png', filetypes['png'])]
        del filetypes['png']
        filetypes = [value + '(*.%s)' % key for (key, value) in pngtype + list(filetypes.items())]
        import string

        def make_save(expr):
            if False:
                for i in range(10):
                    print('nop')
            save_expr = ''
            for char in expr:
                if char not in string.whitespace:
                    if char in string.ascii_letters or char in string.digits or char in '._':
                        save_expr += char
                    else:
                        save_expr += '_'
            return save_expr
        layer = self.current_layer
        if layer is not None:
            save_expressions = list(map(make_save, layer.state.expressions))
            type = 'histogram' if self.dimensions == 1 else 'density'
            filename = layer.dataset.name + '_%s_' % type + '-vs-'.join(save_expressions) + '.png'
            filename = QtGui.QFileDialog.getSaveFileName(self, 'Export to figure', filename, ';;'.join(filetypes))
            if isinstance(filename, tuple):
                filename = filename[0]
            filename = str(filename)
            if filename:
                logger.debug('saving to figure: %s' % filename)
                self.fig.savefig(filename)
                self.filename_figure_last = filename
                self.action_save_figure_again.setEnabled(True)

    def onActionSaveFigureAgain(self, *ignore_args):
        if False:
            return 10
        logger.debug('saving to figure: %s' % self.filename_figure_last)
        self.fig.savefig(self.filename_figure_last)

    def get_aspect(self):
        if False:
            i = 10
            return i + 15
        if 0:
            (xmin, xmax) = self.axes.get_xlim()
            (ymin, ymax) = self.axes.get_ylim()
            height = ymax - ymin
            width = xmax - xmin
        return 1

    def onActionAspectLockOne(self, *ignore_args):
        if False:
            for i in range(10):
                print('nop')
        self.state.aspect = self.get_aspect() if self.action_aspect_lock_one.isChecked() else None
        logger.debug('set aspect to: %r' % self.state.aspect)
        self.check_aspect(0)
        self.queue_history_change('Set equal aspect' if self.state.aspect else 'Unset equal aspect')
        self.update_all_layers()

    def _onActionAspectLockOne(self, *ignore_args):
        if False:
            while True:
                i = 10
        self.state.aspect = 1
        logger.debug('set aspect to: %r' % self.state.aspect)

    def onActionExport(self):
        if False:
            print('Hello World!')
        if self.dimensions == 3:
            (ok, mask) = select_many(self, 'Choose export options', ['Export grid', 'Export grid from selection', 'Export vector grid'])
            name = self.current_layer.dataset.name
            name = gettext(self, 'Export name', 'Give a base name for the export files', name)
            if name:
                dir_path = getdir(self, 'Choose directory where to save')
                if dir_path is None:
                    return
                msg_list = []
                yesall = False
                optionsname = os.path.join(dir_path, name + '_meta.json')
                options = {}
                options['extent'] = list(self.current_layer.state.ranges_grid[0]) + list(self.current_layer.state.ranges_grid[1])
                with open(optionsname, 'w') as f:
                    json.dump(options, f, indent=4)
                msg_list.append('wrote: ' + optionsname)
                if mask[0]:
                    gridname = os.path.join(dir_path, name + '_grid.npy')
                    if not os.path.exists(gridname) or yesall:
                        (yes, yesall) = (True, yesall)
                    else:
                        (yes, yesall) = dialog_confirm(self, 'Overwrite', 'Overwrite: ' + gridname, to_all=True)
                    if yes or yesall:
                        np.save(gridname, self.current_layer.amplitude_grid)
                        msg_list.append('wrote: ' + gridname)
                if mask[1]:
                    if self.current_layer.dataset.mask is not None:
                        gridname = os.path.join(dir_path, name + '_grid_selection.npy')
                        if not os.path.exists(gridname) or yesall:
                            (yes, yesall) = (True, yesall)
                        else:
                            (yes, yesall) = dialog_confirm(self, 'Overwrite', 'Overwrite: ' + gridname, to_all=True)
                        if yes or yesall:
                            np.save(gridname, self.current_layer.amplitude_selection)
                            msg_list.append('wrote: ' + gridname)
                if mask[2]:
                    if hasattr(self.current_layer, 'vector_grids'):
                        gridname = os.path.join(dir_path, name + '_grid_vector.npy')
                        if not os.path.exists(gridname) or yesall:
                            (yes, yesall) = (True, yesall)
                        else:
                            (yes, yesall) = dialog_confirm(self, 'Overwrite', 'Overwrite: ' + gridname, to_all=True)
                        if yes or yesall:
                            np.save(gridname, self.current_layer.vector_grids)
                            msg_list.append('wrote: ' + gridname)
                msg = '\n'.join(msg_list)
                dialog_info(self, 'Finished export', msg)
        elif self.dimensions == 2:
            (ok, mask) = select_many(self, 'Choose export options', ['Export matplotlib python script', 'Export grid', 'Export selection', 'Export vector grid'])
            if ok:
                name = self.current_layer.dataset.name
                name = gettext(self, 'Export name', 'Give a base name for the export files', name)
                if name:
                    dir_path = getdir(self, 'Choose directory where to save')
                    if dir_path is None:
                        return
                    msg_list = []
                    yesall = False
                    if mask[0]:
                        scriptname = os.path.join(dir_path, name + '_plot.py')
                        template = vaex.ui.templates.matplotlib
                        if not os.path.exists(scriptname):
                            (yes, yesall) = (True, False)
                        else:
                            (yes, yesall) = dialog_confirm(self, 'Overwrite', 'Overwrite: ' + scriptname, to_all=True)
                        if yes or yesall:
                            with open(scriptname, 'w') as f:
                                f.write(template.format(name=name))
                                msg_list.append('wrote: ' + scriptname)
                    optionsname = os.path.join(dir_path, name + '_meta.json')
                    options = {}
                    options['extent'] = list(self.current_layer.state.ranges_grid[0]) + list(self.current_layer.state.ranges_grid[1])
                    with open(optionsname, 'w') as f:
                        json.dump(options, f, indent=4)
                    msg_list.append('wrote: ' + optionsname)
                    if mask[1]:
                        gridname = os.path.join(dir_path, name + '_grid.npy')
                        if not os.path.exists(gridname) or yesall:
                            (yes, yesall) = (True, yesall)
                        else:
                            (yes, yesall) = dialog_confirm(self, 'Overwrite', 'Overwrite: ' + gridname, to_all=True)
                        if yes or yesall:
                            np.save(gridname, self.current_layer.amplitude_grid)
                            msg_list.append('wrote: ' + gridname)
                        if mask[2]:
                            if self.current_layer.dataset.mask is not None:
                                gridname = os.path.join(dir_path, name + '_grid_selection.npy')
                                if not os.path.exists(gridname) or yesall:
                                    (yes, yesall) = (True, yesall)
                                else:
                                    (yes, yesall) = dialog_confirm(self, 'Overwrite', 'Overwrite: ' + gridname, to_all=True)
                                if yes or yesall:
                                    np.save(gridname, self.current_layer.amplitude_selection)
                                    msg_list.append('wrote: ' + gridname)
                        if mask[3]:
                            if hasattr(self.current_layer, 'vector_grids'):
                                gridname = os.path.join(dir_path, name + '_grid_vector.npy')
                                if not os.path.exists(gridname) or yesall:
                                    (yes, yesall) = (True, yesall)
                                else:
                                    (yes, yesall) = dialog_confirm(self, 'Overwrite', 'Overwrite: ' + gridname, to_all=True)
                                if yes or yesall:
                                    np.save(gridname, self.current_layer.vector_grids)
                                    msg_list.append('wrote: ' + gridname)
                        msg = '\n'.join(msg_list)
                        dialog_info(self, 'Finished export', msg)
        else:
            dialog_info(self, 'Export failure', 'Oops sorry, export only supported for 2d and 3d (partially) plots at the moment')

    def active_layers(self):
        if False:
            i = 10
            return i + 15
        return [self.current_layer] if self.current_layer else self.layers

    def update_favorite_selections(self):
        if False:
            return 10
        if hasattr(self, '_favorite_selections_actions'):
            for action in self._favorite_selections_actions:
                self.menu_selection.removeAction(action)
        self._favorite_selections_actions = []
        if self.current_layer:
            for (key, value) in self.current_layer.dataset.favorite_selections.items():

                def select(ignore=None, key=key):
                    if False:
                        for i in range(10):
                            print('nop')
                    self.current_layer.dataset.selection_favorite_apply(name=key)
                action = QtGui.QAction(key, self)
                action.triggered.connect(select)
                self._favorite_selections_actions.append(action)
                self.menu_selection.addAction(action)

    def addToolbar2(self, layout, contrast=True, gamma=True):
        if False:
            i = 10
            return i + 15
        self.toolbar2 = QtGui.QToolBar(self)
        self.toolbar2.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.toolbar2.setIconSize(QtCore.QSize(16, 16))
        layout.addWidget(self.toolbar2)

        def on_store_selection():
            if False:
                return 10
            if self.current_layer:
                dataset = self.current_layer.dataset
                if not dataset.has_selection():
                    dialog_error(self, 'No selection', 'No selection made')
                else:
                    path = dataset.name + '-selection.yaml'
                    path = dialogs.get_path_save(self, 'Save selection', path, 'YAML (*.yaml);;JSON (*.json)')
                    if path:
                        (_, ext) = os.path.splitext(path)
                        data = dataset.get_selection().to_dict()
                        if ext == '.yaml':
                            import yaml
                            with open(path, 'w') as f:
                                yaml.dump(data, f)
                        elif ext == '.json':
                            import json
                            with open(path, 'w') as f:
                                json.dump(data, f)
                        else:
                            dialogs.dialog_error(self, 'Unknown extension', 'Unknown extension %r' % ext)
        self.action_selection_store = QtGui.QAction(QtGui.QIcon(iconfile('tag-export')), '&Export selection query', self)
        self.action_selection_store.triggered.connect(on_store_selection)
        self.menu_selection.addAction(self.action_selection_store)

        def on_load_selection():
            if False:
                for i in range(10):
                    print('nop')
            if self.current_layer:
                dataset = self.current_layer.dataset
                path = dataset.name + '-selection.yaml'
                path = get_path_open(self, 'Open selection', path, 'Compatible files for vaex (*.yaml *.json)')
                if path:
                    (_, ext) = os.path.splitext(path)
                    if ext == '.yaml':
                        import yaml
                        with open(path) as f:
                            data = yaml.load(f)
                    elif ext == '.json':
                        import json
                        with open(path) as f:
                            json.load(f)
                    else:
                        dialogs.dialog_error(self, 'Unknown extension', 'Unknown extension %r' % ext)
                    try:
                        selection = vaex.dataset.selection_from_dict(dataset, data)
                    except Exception as e:
                        logger.exception('error reading in selection')
                        dialogs.dialog_error(self, 'Error reading in selection', 'Error reading in selection: %r' % e)
                    dataset.set_selection(selection)
                    self.queue_update()
        self.action_selection_load = QtGui.QAction(QtGui.QIcon(iconfile('tag-import')), '&Import selection query', self)
        self.action_selection_load.triggered.connect(on_load_selection)
        self.menu_selection.addAction(self.action_selection_load)
        self.menu_selection.addSeparator()

        def on_selection_copy():
            if False:
                while True:
                    i = 10
            if self.current_layer:
                selection = self.current_layer.dataset.get_selection()
                if self.current_layer:
                    if selection:
                        data = selection.to_dict()
                        clipboard = QtGui.QApplication.clipboard()
                        f = StringIO()
                        vaex.utils.yaml_dump(f, data)
                        text = str(f.getvalue())
                        clipboard.setText(text)
                    else:
                        dialogs.dialog_error(self, 'No selection', 'No selection exists')
            else:
                dialogs.dialog_error(self, 'No layer', 'No active layer')
        self.action_selection_copy = QtGui.QAction('&Copy selection query', self)
        self.action_selection_copy.triggered.connect(on_selection_copy)
        self.menu_selection.addAction(self.action_selection_copy)

        def on_selection_paste():
            if False:
                while True:
                    i = 10
            clipboard = QtGui.QApplication.clipboard()
            f = StringIO()
            f.write(clipboard.text())
            f.seek(0)
            data = vaex.utils.yaml_load(f)
            for layer in self.active_layers():
                selection = vaex.dataset.selection_from_dict(layer.dataset, data)
                layer.dataset.set_selection(selection)
        self.action_selection_paste = QtGui.QAction('&Paste selection query', self)
        self.action_selection_paste.triggered.connect(on_selection_paste)
        self.menu_selection.addAction(self.action_selection_paste)
        self.menu_selection.addSeparator()

        def on_selection_add_favorite():
            if False:
                print('Hello World!')
            if self.current_layer:
                dataset = self.current_layer.dataset
                if dataset.get_selection():
                    name = dialogs.gettext(self, 'Add selection to favorites', 'Name', 'my selection')
                    if name:
                        if name not in dataset.favorite_selections or dialogs.dialog_confirm(self, 'Overwrite', 'Overwrite selection with the same name?'):
                            dataset.selection_favorite_add(name)
                            self.update_favorite_selections()
                else:
                    dialogs.dialog_error(self, 'No selection', 'No selection exists')
            else:
                dialogs.dialog_error(self, 'No layer', 'No active layer')
        self.action_selection_add_favorites = QtGui.QAction(QtGui.QIcon(iconfile('star')), '&Add to favorites', self)
        self.action_selection_add_favorites.triggered.connect(on_selection_add_favorite)
        self.menu_selection.addAction(self.action_selection_add_favorites)

        def on_selection_remove_favorite():
            if False:
                for i in range(10):
                    print('nop')
            if self.current_layer:
                dataset = self.current_layer.dataset
                if dataset.favorite_selections:
                    names = list(dataset.favorite_selections.keys())
                    index = dialogs.choose(self, 'Remove favorite', 'Favorite', names)
                    if index is not None:
                        key = names[index]
                        dataset.selection_favorite_remove(key)
                        self.update_favorite_selections()
                else:
                    dialogs.dialog_error(self, 'No favorites', 'No favorites exists')
            else:
                dialogs.dialog_error(self, 'No layer', 'No active layer')
        self.action_selection_remove_favorites = QtGui.QAction(QtGui.QIcon(iconfile('star--minus')), '&Remove a favorite', self)
        self.action_selection_remove_favorites.triggered.connect(on_selection_remove_favorite)
        self.menu_selection.addAction(self.action_selection_remove_favorites)
        self.action_save_figure = QtGui.QAction(QtGui.QIcon(iconfile('image-export')), '&Export figure', self)
        self.action_save_figure_again = QtGui.QAction(QtGui.QIcon(iconfile('image-export')), '&Export figure again', self)
        self.menu_file.addAction(self.action_save_figure)
        self.menu_file.addAction(self.action_save_figure_again)
        self.action_save_figure.triggered.connect(self.onActionSaveFigure)
        self.action_save_figure_again.triggered.connect(self.onActionSaveFigureAgain)
        self.action_save_figure_again.setEnabled(False)
        self.action_export = QtGui.QAction(QtGui.QIcon(iconfile('script-export')), '&Export data/script', self)
        self.menu_file.addSeparator()
        self.menu_file.addAction(self.action_export)
        self.action_export.triggered.connect(self.onActionExport)
        self.action_aspect_lock_one = QtGui.QAction(QtGui.QIcon(iconfile('control-stop-square')), 'Equal aspect', self)
        self.action_aspect_lock_one.setShortcut('Ctrl+=')
        self.menu_view.insertAction(self.action_mini_mode_normal, self.action_aspect_lock_one)
        self.action_aspect_lock_one.setCheckable(True)
        self.action_aspect_lock_one.triggered.connect(self.onActionAspectLockOne)
        self.action_undo = QtGui.QAction(QtGui.QIcon(iconfile('arrow-curve-180-left')), 'Back', self)
        self.action_redo = QtGui.QAction(QtGui.QIcon(iconfile('arrow-curve-000-left')), 'Forward', self)
        self.action_undo.setShortcut('Ctrl+Left')
        self.action_redo.setShortcut('Ctrl+Right')
        self.toolbar.insertAction(self.action_move, self.action_undo)
        self.toolbar.insertAction(self.action_move, self.action_redo)
        self.menu_view.insertAction(self.action_aspect_lock_one, self.action_undo)
        self.menu_view.insertAction(self.action_aspect_lock_one, self.action_redo)
        self.action_undo.triggered.connect(self.onActionUndo)
        self.action_redo.triggered.connect(self.onActionRedo)
        self.checkUndoRedo()
        self.action_disjoin = QtGui.QAction(QtGui.QIcon(iconfile('sql-join-outer-exclude')), 'Disjoined', self)
        self.action_disjoin.setCheckable(True)
        self.action_disjoin.triggered.connect(self.onActionDisjoin)
        self.action_disjoin.setShortcut('Ctrl+D')
        self.menu_view.insertAction(self.action_mini_mode_normal, self.action_disjoin)
        self.action_axes_lock = QtGui.QAction(QtGui.QIcon(iconfile('lock')), 'Lock axis', self)
        self.action_axes_lock.setCheckable(True)
        self.action_axes_lock.triggered.connect(self.onActionAxesLock)
        self.action_axes_lock.setShortcut('Ctrl+Shift+L')
        self.menu_view.insertAction(self.action_mini_mode_normal, self.action_axes_lock)
        self.menu_view.insertSeparator(self.action_aspect_lock_one)
        self.menu_view.insertSeparator(self.action_mini_mode_normal)
        self.toolbar2.setVisible(False)

    def onActionAxesLock(self, ignore=None):
        if False:
            return 10
        self.state.axis_lock = self.action_axes_lock.isChecked()
        self.queue_history_change('Set axis lock' if self.action_axes_lock.isChecked() else 'Unset axis lock')
        self.push_full_state()
        self.checkUndoRedo()

    def onActionShuffled(self, ignore=None):
        if False:
            while True:
                i = 10
        self.xoffset = 1 if self.action_shuffled.isChecked() else 0
        self.compute()
        self.dataset.executor.execute()
        logger.debug('xoffset = %r' % self.xoffset)

    def onActionDisjoin(self, ignore=None):
        if False:
            print('Hello World!')
        layer = self.current_layer
        if layer:
            layer.show_disjoined = self.action_disjoin.isChecked()
            layer.calculate_amplitudes()
            logger.debug('show_disjoined = %r' % layer.show_disjoined)
            self.queue_replot()

    def addToolbar(self, layout, pick=True, xselect=True, yselect=True, lasso=True):
        if False:
            i = 10
            return i + 15
        self.toolbar = QtGui.QToolBar(self)
        self.toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.action_group_main = QtGui.QActionGroup(self)
        self.action_group_mainSelectMode = QtGui.QActionGroup(self)
        self.action_group_display = QtGui.QActionGroup(self)
        self.actiongroup_display_mode = QtGui.QActionGroup(self)
        self.action_mini_mode_normal = QtGui.QAction(QtGui.QIcon(iconfile('layout-hf-2')), 'Normal', self)
        self.action_mini_mode_compact = QtGui.QAction(QtGui.QIcon(iconfile('layout-hf')), 'Compact', self)
        self.action_mini_mode_ultra = QtGui.QAction(QtGui.QIcon(iconfile('layout')), 'Ultra compact', self)
        self.action_mini_mode_normal.setShortcut('Ctrl+Shift+N')
        self.action_mini_mode_compact.setShortcut('Ctrl+Shift+C')
        self.action_mini_mode_ultra.setShortcut('Ctrl+Shift+U')
        self.action_group_mini_mode = QtGui.QActionGroup(self)
        self.action_group_mini_mode.addAction(self.action_mini_mode_normal)
        self.action_group_mini_mode.addAction(self.action_mini_mode_compact)
        self.action_group_mini_mode.addAction(self.action_mini_mode_ultra)
        self.action_fullscreen = QtGui.QAction(QtGui.QIcon(iconfile('picture_empty')), '&Fullscreen', self)
        self.action_fullscreen.setCheckable(True)
        self.action_fullscreen.setShortcut('Ctrl+F')
        self.action_toolbar_toggle = QtGui.QAction(QtGui.QIcon(iconfile('ui-toolbar')), '&Toolbars', self)
        self.action_toolbar_toggle.setCheckable(True)
        self.action_toolbar_toggle.setChecked(True)
        self.action_toolbar_toggle.setShortcut('Ctrl+Shift+T')

        def toggle_fullscreen(ignore=None):
            if False:
                i = 10
                return i + 15
            fullscreen = self.windowState() & QtCore.Qt.WindowFullScreen
            fullscreen = not fullscreen
            self.action_fullscreen.setChecked(fullscreen)
            if fullscreen:
                self.setWindowState(self.windowState() | QtCore.Qt.WindowFullScreen)
            else:
                self.setWindowState(self.windowState() ^ QtCore.Qt.WindowFullScreen)
        self.action_fullscreen.triggered.connect(toggle_fullscreen)
        self.action_toolbar_toggle.triggered.connect(self.on_toolbar_toggle)
        self.action_move = QtGui.QAction(QtGui.QIcon(iconfile('edit-move')), '&Move', self)
        self.action_move.setShortcut('Ctrl+M')
        self.menu_mode.addAction(self.action_move)
        self.action_pick = QtGui.QAction(QtGui.QIcon(iconfile('cursor')), '&Pick', self)
        self.action_pick.setShortcut('Ctrl+P')
        self.menu_mode.addAction(self.action_pick)
        self.action_select = QtGui.QAction(QtGui.QIcon(iconfile('glue_lasso16')), '&Select(you should not read this)', self)
        self.action_xrange = QtGui.QAction(QtGui.QIcon(iconfile('glue_xrange_select16')), '&x-range', self)
        self.action_yrange = QtGui.QAction(QtGui.QIcon(iconfile('glue_yrange_select16')), '&y-range', self)
        self.action_lasso = QtGui.QAction(QtGui.QIcon(iconfile('glue_lasso16')), '&Lasso', self)
        self.action_select_rectangle = QtGui.QAction(QtGui.QIcon(iconfile('glue_square16')), '&Rectangle', self)
        self.action_slice = QtGui.QAction(QtGui.QIcon(iconfile('cutlery-knife')), '&Slice', self)
        self.action_select_viewport = QtGui.QAction(QtGui.QIcon(iconfile('control-stop-square')), '&Select viewport', self)
        self.action_select_none = QtGui.QAction(QtGui.QIcon(iconfile('cross')), '&No selection', self)
        self.action_select_invert = QtGui.QAction(QtGui.QIcon(iconfile('arrow-circle-double-135')), '&Invert', self)
        self.action_xrange.setShortcut('Ctrl+Shift+X')
        self.menu_mode.addAction(self.action_xrange)
        self.menu_selection.addAction(self.action_xrange)
        self.action_yrange.setShortcut('Ctrl+Shift+Y')
        self.menu_mode.addAction(self.action_yrange)
        self.menu_selection.addAction(self.action_yrange)
        self.action_lasso.setShortcut('Ctrl+L')
        self.menu_mode.addAction(self.action_lasso)
        self.menu_selection.addAction(self.action_lasso)
        self.action_select_rectangle.setShortcut('Ctrl+R')
        self.menu_mode.addAction(self.action_select_rectangle)
        self.menu_selection.addAction(self.action_select_rectangle)
        self.menu_selection.addSeparator()
        self.action_select_viewport.setShortcut('Ctrl+Shift+V')
        self.menu_selection.addAction(self.action_select_viewport)
        self.action_select_none.setShortcut('Ctrl+N')
        self.menu_selection.addAction(self.action_select_none)
        self.action_select_invert.setShortcut('Ctrl+I')
        self.menu_selection.addAction(self.action_select_invert)
        self.menu_selection.addSeparator()
        self.menu_mode.addSeparator()
        self.action_select_invert.setShortcut('Ctrl+I')
        self.action_select_mode_replace = QtGui.QAction(QtGui.QIcon(iconfile('sql-join-right')), '&Replace', self)
        self.action_select_mode_and = QtGui.QAction(QtGui.QIcon(iconfile('sql-join-inner')), '&And', self)
        self.action_select_mode_or = QtGui.QAction(QtGui.QIcon(iconfile('sql-join-outer')), '&Or', self)
        self.action_select_mode_xor = QtGui.QAction(QtGui.QIcon(iconfile('sql-join-outer-exclude')), 'Xor', self)
        self.action_select_mode_subtract = QtGui.QAction(QtGui.QIcon(iconfile('sql-join-left-exclude')), 'Subtract', self)
        self.action_select_mode_replace.setShortcut('Ctrl+Shift+=')
        self.action_select_mode_and.setShortcut('Ctrl+Shift+&')
        self.action_select_mode_or.setShortcut('Ctrl+Shift+|')
        self.action_select_mode_xor.setShortcut('Ctrl+Shift+^')
        self.action_select_mode_subtract.setShortcut('Ctrl+Shift+-')
        self.menu_mode.addAction(self.action_select_mode_replace)
        self.menu_mode.addAction(self.action_select_mode_and)
        self.menu_mode.addAction(self.action_select_mode_or)
        self.menu_mode.addAction(self.action_select_mode_xor)
        self.menu_mode.addAction(self.action_select_mode_subtract)
        self.action_samp_send_table_select_row_list = QtGui.QAction(QtGui.QIcon(iconfile('block--arrow')), 'Broadcast selection over SAMP', self)
        self.action_samp_send_table_select_row_list.setShortcut('Ctrl+Shift+B')
        self.menu_samp.addAction(self.action_samp_send_table_select_row_list)

        def send_samp_selection(ignore=None):
            if False:
                for i in range(10):
                    print('nop')
            self.signal_samp_send_selection.emit(self.dataset)
        self.send_samp_selection_reference = send_samp_selection
        self.action_samp_send_table_select_row_list.triggered.connect(send_samp_selection)
        self.action_display_mode_both = QtGui.QAction(QtGui.QIcon(iconfile('picture_empty')), 'Show both', self)
        self.action_display_mode_full = QtGui.QAction(QtGui.QIcon(iconfile('picture_empty')), 'Show full', self)
        self.action_display_mode_selection = QtGui.QAction(QtGui.QIcon(iconfile('picture_empty')), 'Show selection', self)
        self.action_display_mode_both_contour = QtGui.QAction(QtGui.QIcon(iconfile('picture_empty')), 'Show contour', self)
        self.actions_display = [self.action_display_mode_both, self.action_display_mode_full, self.action_display_mode_selection, self.action_display_mode_both_contour]
        for action in self.actions_display:
            self.action_group_display.addAction(action)
            action.setCheckable(True)
        action = self.actions_display[0]
        action.setChecked(True)
        self.action_display_current = action
        self.action_group_display.triggered.connect(self.onActionDisplay)
        self.action_group_mainSelectMode.addAction(self.action_select_mode_replace)
        self.action_group_mainSelectMode.addAction(self.action_select_mode_and)
        self.action_group_mainSelectMode.addAction(self.action_select_mode_or)
        self.action_group_mainSelectMode.addAction(self.action_select_mode_xor)
        self.action_group_mainSelectMode.addAction(self.action_select_mode_subtract)
        self.action_group_main.addAction(self.action_move)
        self.action_group_main.addAction(self.action_pick)
        self.action_group_main.addAction(self.action_xrange)
        self.action_group_main.addAction(self.action_yrange)
        self.action_group_main.addAction(self.action_lasso)
        self.action_group_main.addAction(self.action_select_rectangle)
        if self.enable_slicing:
            self.action_group_main.addAction(self.action_slice)
        self.menu_view.addAction(self.action_mini_mode_normal)
        self.menu_view.addAction(self.action_mini_mode_compact)
        self.menu_view.addAction(self.action_mini_mode_ultra)
        self.menu_view.addSeparator()
        self.menu_view.addAction(self.action_fullscreen)
        self.menu_view.addAction(self.action_toolbar_toggle)
        self.menu_view.addSeparator()
        self.action_group_resolution = QtGui.QActionGroup(self)
        self.action_resolution_list = []
        for (index, resolution) in enumerate(grid_resolutions):
            action_resolution = QtGui.QAction('Grid Resolution: %d' % resolution, self)

            def do(ignore=None, resolution=resolution):
                if False:
                    return 10
                self.state.grid_size = resolution
                self.queue_history_change('Set grid resolution to %s' % resolution)
                self.update_all_layers()
            action_resolution.setCheckable(True)
            if resolution == int(self.state.grid_size):
                action_resolution.setChecked(True)
            action_resolution.triggered.connect(do)
            action_resolution.setShortcut('Ctrl+Alt+%d' % (index + 1))
            self.menu_view.addAction(action_resolution)
            self.action_group_resolution.addAction(action_resolution)
            self.action_resolution_list.append(action_resolution)
        self.menu_view.addSeparator()
        self.action_group_resolution_vector = QtGui.QActionGroup(self)
        self.action_resolution_vector_list = []
        for (index, resolution) in enumerate(vector_grid_resolutions):
            action_resolution = QtGui.QAction('Vector grid Resolution: %d' % resolution, self)

            def do(ignore=None, resolution=resolution):
                if False:
                    while True:
                        i = 10
                self.state.vector_grid_size = resolution
                self.queue_history_change('Set vector grid resolution to %s' % resolution)
                self.update_all_layers()
            action_resolution.setCheckable(True)
            if resolution == int(self.state.vector_grid_size):
                action_resolution.setChecked(True)
            action_resolution.triggered.connect(do)
            action_resolution.setShortcut('Ctrl+Shift+Alt+%d' % (index + 1))
            self.menu_view.addAction(action_resolution)
            self.action_group_resolution_vector.addAction(action_resolution)
            self.action_resolution_vector_list.append(action_resolution)
        self.toolbar.addAction(self.action_move)
        if pick:
            self.toolbar.addAction(self.action_pick)
            self.lastAction = self.action_pick
        self.toolbar.addAction(self.action_select)
        self.select_menu = QtGui.QMenu()
        self.action_select.setMenu(self.select_menu)
        self.select_menu.addAction(self.action_lasso)
        self.select_menu.addAction(self.action_select_rectangle)
        if yselect:
            self.select_menu.addAction(self.action_yrange)
            if self.dimensions > 1:
                self.lastActionSelect = self.action_yrange
        if xselect:
            self.select_menu.addAction(self.action_xrange)
            self.lastActionSelect = self.action_xrange
        if lasso:
            if self.dimensions > 1:
                self.lastActionSelect = self.action_lasso
        else:
            self.action_lasso.setEnabled(False)
            self.action_select_rectangle.setEnabled(False)
        if self.enable_slicing:
            self.toolbar.addAction(self.action_slice)
        self.select_menu.addSeparator()
        self.select_menu.addAction(self.action_select_viewport)
        self.select_menu.addAction(self.action_select_none)
        self.select_menu.addAction(self.action_select_invert)
        self.select_menu.addSeparator()
        self.select_mode_button = QtGui.QToolButton()
        self.select_mode_button.setPopupMode(QtGui.QToolButton.InstantPopup)
        self.select_mode_button.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.select_mode_button_menu = QtGui.QMenu()
        self.select_mode_button.setMenu(self.select_mode_button_menu)
        self.select_mode_button_menu.addAction(self.action_select_mode_replace)
        self.select_mode_button_menu.addAction(self.action_select_mode_or)
        self.select_mode_button_menu.addAction(self.action_select_mode_and)
        self.select_mode_button_menu.addAction(self.action_select_mode_xor)
        self.select_mode_button_menu.addAction(self.action_select_mode_subtract)
        self.select_mode_button.setDefaultAction(self.action_select_mode_replace)
        self.toolbar.addWidget(self.select_mode_button)
        if 0:
            self.toolbar.addAction(self.action_zoom)
            self.zoom_menu = QtGui.QMenu()
            self.action_zoom.setMenu(self.zoom_menu)
            self.zoom_menu.addAction(self.action_zoom_rect)
            self.zoom_menu.addAction(self.action_zoom_x)
            self.zoom_menu.addAction(self.action_zoom_y)
            if self.dimensions == 1:
                self.lastActionZoom = self.action_zoom_x
            else:
                self.lastActionZoom = self.action_zoom_rect
            self.toolbar.addSeparator()
            self.toolbar.addAction(self.action_zoom_fit)
        else:
            plugin_chain_toolbar = sorted(self.plugin_queue_toolbar, key=itemgetter(1))
            for (plug, order) in plugin_chain_toolbar:
                plug()
        self.action_group_main.triggered.connect(self.setMode)
        self.action_group_mainSelectMode.triggered.connect(self.setSelectMode)
        self.action_mini_mode_normal.triggered.connect(self.onActionMiniModeNormal)
        self.action_mini_mode_compact.triggered.connect(self.onActionMiniModeCompact)
        self.action_mini_mode_ultra.triggered.connect(self.onActionMiniModeUltra)
        self.action_select.triggered.connect(self.onActionSelect)
        self.action_select_viewport.triggered.connect(self.onActionSelectViewport)
        self.action_select_none.triggered.connect(self.onActionSelectNone)
        self.action_select_invert.triggered.connect(self.onActionSelectInvert)
        self.action_select_mode_replace.setCheckable(True)
        self.action_select_mode_and.setCheckable(True)
        self.action_select_mode_or.setCheckable(True)
        self.action_select_mode_xor.setCheckable(True)
        self.action_select_mode_subtract.setCheckable(True)
        self.action_mini_mode_normal.setCheckable(True)
        self.action_mini_mode_normal.setChecked(True)
        self.action_mini_mode_compact.setCheckable(True)
        self.action_mini_mode_ultra.setCheckable(True)
        self.action_move.setCheckable(True)
        self.action_pick.setCheckable(True)
        self.action_move.setChecked(True)
        self.action_select.setCheckable(True)
        self.action_xrange.setCheckable(True)
        self.action_yrange.setCheckable(True)
        self.action_lasso.setCheckable(True)
        self.action_select_rectangle.setCheckable(True)
        self.action_slice.setCheckable(True)
        self.syncToolbar()
        self.select_mode = 'replace'
        self.setMode(self.action_move)
        self.toolbar.setIconSize(QtCore.QSize(16, 16))
        layout.addWidget(self.toolbar)

    def onActionDisplay(self, action):
        if False:
            return 10
        logger.debug('display = %r' % action.text())
        self.action_display_current = action
        self.plot()

    def onActionMiniMode(self):
        if False:
            for i in range(10):
                print('nop')
        enabled_mini_mode = self.action_mini_mode_compact.isChecked() or self.action_mini_mode_ultra.isChecked()
        ultra_mode = self.action_mini_mode_ultra.isChecked()
        logger.debug('mini screen: %r (ultra: %r)' % (enabled_mini_mode, ultra_mode))
        toolbuttons = self.toolbar.findChildren(QtGui.QToolButton)
        for toolbutton in toolbuttons:
            toolbutton.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly if enabled_mini_mode else QtCore.Qt.ToolButtonTextUnderIcon)
        if enabled_mini_mode:
            values = self.fig.subplotpars
            self.subplotpars_values = {'left': values.left, 'right': values.right, 'bottom': values.bottom, 'top': values.top}
            self.bottomHeight = self.bottomFrame.height()
        self.bottomFrame.setVisible(not enabled_mini_mode)
        if 0:
            if enabled_mini_mode:
                self.resize(QtCore.QSize(self.width(), self.height() - self.bottomHeight))
            else:
                self.resize(QtCore.QSize(self.width(), self.height() + self.bottomHeight))
        self.fig.tight_layout()
        if enabled_mini_mode:
            if ultra_mode:
                self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1.0)
        else:
            self.fig.subplots_adjust(**self.subplotpars_values)
        self.canvas.draw()

    def on_toolbar_toggle(self, ignore=None):
        if False:
            for i in range(10):
                print('nop')
        visible = self.action_toolbar_toggle.isChecked()
        logger.debug('toolbar visible? %r' % (visible,))
        for widget in [self.toolbar, self.toolbar2, self.status_bar]:
            widget.setVisible(visible)

    def onActionMiniModeNormal(self, *args):
        if False:
            print('Hello World!')
        self.onActionMiniMode()
        pass

    def onActionMiniModeCompact(self, *args):
        if False:
            while True:
                i = 10
        self.onActionMiniMode()

    def onActionMiniModeUltra(self, *args):
        if False:
            return 10
        self.onActionMiniMode()

    def setSelectMode(self, action):
        if False:
            for i in range(10):
                print('nop')
        self.select_mode_button.setDefaultAction(action)
        if action == self.action_select_mode_replace:
            self._select_mode = 'replace'
        if action == self.action_select_mode_and:
            self._select_mode = 'and'
        if action == self.action_select_mode_or:
            self._select_mode = 'or'
        if action == self.action_select_mode_xor:
            self._select_mode = 'xor'
        if action == self.action_select_mode_subtract:
            self._select_mode = 'subtract'

    @property
    def select_mode(self):
        if False:
            print('Hello World!')
        return self._select_mode

    @select_mode.setter
    def select_mode(self, value):
        if False:
            while True:
                i = 10
        logger.debug('set to: %r', value)
        action = getattr(self, 'action_select_mode_%s' % value)
        self.setSelectMode(action)

    def onActionSelectViewport(self):
        if False:
            i = 10
            return i + 15
        for layer in self.active_layers():
            if self.dimensions == 3:
                ((xmin, xmax), (ymin, ymax), (zmin, zmax)) = self.state.ranges_viewport
                args = (layer.x, xmin, layer.x, xmax, layer.y, ymin, layer.y, ymax, layer.z, zmin, layer.z, zmax)
                expression = '((%s) >= %f) & ((%s) <= %f) & ((%s) >= %f) & ((%s) <= %f)' % args
            if self.dimensions == 2:
                ((xmin, xmax), (ymin, ymax)) = self.state.ranges_viewport
                args = (layer.x, xmin, layer.x, xmax, layer.y, ymin, layer.y, ymax)
                expression = '((%s) >= %f) & ((%s) <= %f) & ((%s) >= %f) & ((%s) <= %f)' % args
            if self.dimensions == 1:
                (xmin, xmax) = self.state.ranges_viewport
                args = (layer.x, xmin, layer.x, xmax, layer.y, ymin, layer.y, ymax)
                expression = '((%s) >= %f) & ((%s) <= %f)' % args
            logger.debug('rectangle selection using expression: %r' % expression)
            layer.dataset.select(expression, mode=self.select_mode)
            mask = layer.dataset.mask
            self.queue_update()

    def onActionSelectNone(self):
        if False:
            for i in range(10):
                print('nop')
        for layer in self.active_layers():
            layer.dataset.select(None)

    def onActionSelectInvert(self):
        if False:
            for i in range(10):
                print('nop')
        for layer in self.active_layers():
            layer.dataset.select_inverse()
            self.queue_update()

    def onActionSelect(self):
        if False:
            for i in range(10):
                print('nop')
        self.lastActionSelect.setChecked(True)
        self.setMode(self.lastActionSelect)
        self.syncToolbar()

    def syncToolbar(self):
        if False:
            return 10
        for plugin in self.plugins:
            plugin.syncToolbar()
        for action in [self.action_select]:
            logger.debug('sync action: %r' % action.text())
            subactions = action.menu().actions()
            subaction_selected = [subaction for subaction in subactions if subaction.isChecked()]
            logger.debug(' subaction_selected: %r' % subaction_selected)
            logger.debug(' action was selected?: %r' % action.isChecked())
            action.setChecked(len(subaction_selected) > 0)
            logger.debug(' action  is selected?: %r' % action.isChecked())
        logger.debug('last select action: %r' % self.lastActionSelect.text())
        self.action_select.setText(self.lastActionSelect.text())
        self.action_select.setIcon(self.lastActionSelect.icon())

    def check_aspect(self, axis_follow):
        if False:
            for i in range(10):
                print('nop')
        if self.state.aspect is not None:
            if self.state.ranges_viewport is None:
                return
            if any([k is None for k in self.state.ranges_viewport]):
                return
            width = self.state.ranges_viewport[axis_follow][1] - self.state.ranges_viewport[axis_follow][0]
            centers = [(self.state.ranges_viewport[i][1] + self.state.ranges_viewport[i][0]) / 2.0 for i in range(self.dimensions)]
            for i in range(self.dimensions):
                if i != axis_follow:
                    self.state.ranges_viewport[i] = [None, None]
                    self.state.ranges_viewport[i][0] = centers[i] - width / 2
                    self.state.ranges_viewport[i][1] = centers[i] + width / 2
            return
            otheraxes = list(range(self.dimensions))
            allaxes = list(range(self.dimensions))
            otheraxes.remove(axis_follow)
            ranges = [self.state.ranges_viewport[i] for i in otheraxes]
            logger.debug('aspect 1')
            if None in ranges:
                return
            width = self.state.ranges_viewport[axis_follow][1] - self.state.ranges_viewport[axis_follow][0]
            center = (self.state.ranges_viewport[axis_follow][1] + self.state.ranges_viewport[axis_follow][0]) / 2.0
            logger.debug('aspect 2')
            centers = [(self.state.ranges_viewport[i][1] + self.state.ranges_viewport[i][0]) / 2.0 for i in range(self.dimensions)]
            logger.debug('aspect 3')
            for i in range(self.dimensions - 1):
                axis_index = otheraxes[i]
                self.state.ranges_viewport[axis_index] = [None, None]
                self.state.ranges_viewport[axis_index][0] = centers[axis_index] - width / 2
                self.state.ranges_viewport[axis_index][1] = centers[axis_index] + width / 2
                logger.debug('aspect i=%d,%d', i, axis_index)
            for layer in self.layers:
                for i in range(self.dimensions - 1):
                    axis_index = otheraxes[i]
                    layer.set_range(list(self.state.ranges_viewport[axis_index]), axis_index)
                layer.state.ranges_grid[axis_follow] = list(self.state.ranges_viewport[axis_follow])

class HistogramPlotDialog(PlotDialog):
    type_name = 'histogram'

    def __init__(self, parent, dataset, app, **kwargs):
        if False:
            while True:
                i = 10
        super(HistogramPlotDialog, self).__init__(parent, dataset, 1, ['x'], app, **kwargs)

    def beforeCanvas(self, layout):
        if False:
            i = 10
            return i + 15
        self.addToolbar(layout, yselect=False, lasso=False)

    def _afterCanvas(self, layout):
        if False:
            for i in range(10):
                print('nop')
        self.addToolbar2(layout, contrast=False, gamma=False)
        super(HistogramPlotDialog, self).afterCanvas(layout)

    def add_histogram(self, x, counts, selection):
        if False:
            i = 10
            return i + 15
        self.histograms.append((x, counts))

    def plot(self):
        if False:
            for i in range(10):
                print('nop')
        t0 = time.time()
        self.axes.cla()
        self.axes.autoscale(False)
        if len(self.layers) == 0:
            return
        first_layer = self.layers[0]
        if self.state.range_level_show is None:
            self.state.range_level_show = first_layer.range_level
        if self.state.range_level_show is None:
            logger.error('cannot plot when range_level_show is None')
            return
        for layer in self.layers:
            layer.plot([self.axes], self.add_histogram)
        (xmin_show, xmax_show) = self.state.ranges_viewport[0]
        self.axes.set_xlim(xmin_show, xmax_show)
        (ymin_show, ymax_show) = self.state.range_level_show
        self.axes.set_xlabel(self.get_label(0))
        self.axes.set_ylabel(self.get_label(1))
        self.axes.set_ylim(ymin_show, ymax_show)
        if not self.action_mini_mode_ultra.isChecked():
            self.fig.tight_layout(pad=0.0)
        self.canvas.draw()
        self.update()
        self.message('plotting %.2fs' % (time.time() - t0), index=100)
        self.signal_plot_finished.emit(self, self.fig)
        if self._plot_event:
            logger.debug('plotting done')
            self._plot_event.set()
        return
        Nvector = self.state.grid_size
        width = self.state.ranges_viewport[0][1] - self.state.ranges_viewport[0][0]
        x = np.arange(0, Nvector) / float(Nvector) * width + self.state.ranges_viewport[0][0]
        (xmin, xmax) = self.state.ranges_viewport[0]
        (xmin, xmax) = self.state.ranges_viewport[0]
        if self.state.ranges_viewport[0] is None:
            self.state.ranges_viewport[0] = [xmin, xmax]
        self.delta = (xmax - xmin) / self.state.grid_size
        self.centers = (np.arange(self.state.grid_size) + 0.5) * self.delta + xmin
        logger.debug('expr for amplitude: %r' % self.amplitude_expression)
        grid_map = self.create_grid_map(self.state.grid_size, False)
        amplitude = self.eval_amplitude(self.amplitude_expression, locals=grid_map)
        use_selection = self.dataset.mask is not None
        if use_selection:
            grid_map_selection = self.create_grid_map(self.state.grid_size, True)
            amplitude_selection = self.eval_amplitude(self.amplitude_expression, locals=grid_map_selection)
        if use_selection:
            self.axes.bar(self.centers, amplitude, width=self.delta, align='center')
            self.axes.bar(self.centers, amplitude_selection, width=self.delta, align='center', color='red', alpha=0.8)
        else:
            self.axes.bar(self.centers, amplitude, width=self.delta, align='center')
        if self.range_level is None:
            if self.state.weight_expression:
                self.range_level = (np.nanmin(amplitude) * 1.1, np.nanmax(amplitude) * 1.1)
            else:
                self.range_level = (0, np.nanmax(amplitude) * 1.1)
        if self.action_mini_mode_compact:
            self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1.0)

class ScatterPlotDialog(PlotDialog):
    type_name = 'density2d'

    def __init__(self, parent, dataset, app, **options):
        if False:
            return 10
        super(ScatterPlotDialog, self).__init__(parent, dataset, 2, 'x y'.split(), app, **options)

    def _afterCanvas(self, layout):
        if False:
            i = 10
            return i + 15
        self.addToolbar2(layout)
        super(ScatterPlotDialog, self).afterCanvas(layout)

    def add_image_layer(self, rgba, intensity):
        if False:
            print('Hello World!')
        self.image_layers.append(rgba)

    def post_draw(self, fig, axes):
        if False:
            i = 10
            return i + 15
        pass

    def plot(self):
        if False:
            i = 10
            return i + 15
        self.fig.clf()
        self.add_axes()
        if 0:
            self.fig.delaxes(self.fig.axes[1])
            self.colorbar = None
        self.image_layers = []
        self.axes.rgb_images = self.image_layers
        self.axes.cla()
        if len(self.layers) == 0:
            self.canvas.draw()
            self.update()
            return
        first_layer = self.layers[0]
        N = self.state.grid_size
        background = np.ones((N, N, 4), dtype=np.float64)
        background_color = self.background_color
        if background_color == 'auto':
            if self.blend_mode in 'screen lighten'.split():
                background_color = 'black'
            else:
                background_color = 'white'
        background[:, :, 0:3] = matplotlib.colors.colorConverter.to_rgb(background_color)
        background[:, :, 3] = 1.0
        ranges = []
        if first_layer.state.ranges_grid is None:
            return
        if None in first_layer.state.ranges_grid:
            return
        for (minimum, maximum) in first_layer.state.ranges_grid:
            ranges.append(minimum)
            ranges.append(maximum)
        background = np.transpose(background, (1, 0, 2))
        placeholder = self.axes.imshow(background, extent=ranges, origin='lower')
        self.add_image_layer(background, None)
        for i in range(self.dimensions):
            if self.state.ranges_viewport[i] is None:
                self.state.ranges_viewport[i] = copy.copy(first_layer.state.ranges_grid[i])
        (xmin, xmax) = self.state.ranges_viewport[0]
        (ymin, ymax) = self.state.ranges_viewport[1]
        width = xmax - xmin
        height = ymax - ymin
        extent = [xmin - width, xmax + width, ymin - height, ymax + height]
        Z1 = np.array(([0, 1] * 8 + [1, 0] * 8) * 8)
        Z1.shape = (16, 16)
        for layer in self.layers:
            layer.plot([self.axes], self.add_image_layer)
        rgba = vaex.ui.imageblending.blend(self.image_layers, self.blend_mode)
        rgba[..., 3] = rgba[..., 3] * 0 + 1
        for c in range(4):
            rgba[:, :, c] = np.clip(rgba[:, :, c] ** self.layer_gamma * self.layer_brightness, 0.0, 1.0)
        rgba = np.transpose(rgba, (1, 0, 2))
        placeholder.set_data((rgba * 255).astype(np.uint8))
        if self.state.aspect is None:
            self.axes.set_aspect('auto')
        else:
            self.axes.set_aspect(self.state.aspect)
        if 1:
            self.axes.set_xlabel(self.get_label(0))
            self.axes.set_ylabel(self.get_label(1))
        self.axes.set_xlim(*self.state.ranges_viewport[0])
        self.axes.set_ylim(*self.state.ranges_viewport[1])
        layer = first_layer

        class Dummy(object):
            pass

        class ColorbarWrapper(matplotlib.cm.ScalarMappable):

            def __init__(self, layer):
                if False:
                    while True:
                        i = 10
                self.layer = layer
                self.cmap = matplotlib.cm.cmap_d[self.layer.state.colormap]
                self.norm = matplotlib.colors.Normalize(layer.level_ranges[0], layer.level_ranges[1])
                self.norm = matplotlib.colors.Normalize(max(layer.level_ranges), min(layer.level_ranges))
                matplotlib.cm.ScalarMappable.__init__(self, self.norm, self.cmap)

            def autoscale_None(self):
                if False:
                    print('Hello World!')
                pass
        if first_layer.state.colorbar:
            wrapper = ColorbarWrapper(first_layer)
            self.colorbar = self.fig.colorbar(wrapper, use_gridspec=True)
            self.colorbar.ax.set_ylabel(first_layer.amplitude_label())
            (ymin, ymax) = self.colorbar.ax.get_ylim()
            if (layer.level_ranges[0] > layer.level_ranges[1]) != (ymin > ymax):
                self.colorbar.ax.invert_yaxis()
        if 0:
            title_text = self.title_expression.format(**self.getVariableDict())
            if hasattr(self, 'title'):
                self.title.set_text(title_text)
            else:
                self.title = self.fig.suptitle(title_text)
            self.canvas.draw()
        if self.action_mini_mode_ultra.isChecked():
            logger.debug('ultra compact mode')
            self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1.0)

        class ScaledLocator(matplotlib.ticker.MaxNLocator):
            """
            Locates regular intervals along an axis scaled by *dx* and shifted by
            *x0*. For example, this would locate minutes on an axis plotted in seconds
            if dx=60.  This differs from MultipleLocator in that an approriate interval
            of dx units will be chosen similar to the default MaxNLocator.
            """

            def __init__(self, dx=1.0, x0=0.0):
                if False:
                    while True:
                        i = 10
                self.dx = dx
                self.x0 = x0
                matplotlib.ticker.MaxNLocator.__init__(self, nbins=9, steps=[1, 2, 5, 10])

            def rescale(self, x):
                if False:
                    return 10
                return x / self.dx + self.x0

            def inv_rescale(self, x):
                if False:
                    while True:
                        i = 10
                return (x - self.x0) * self.dx

            def __call__(self):
                if False:
                    for i in range(10):
                        print('nop')
                (vmin, vmax) = self.axis.get_view_interval()
                (vmin, vmax) = (self.rescale(vmin), self.rescale(vmax))
                (vmin, vmax) = matplotlib.transforms.nonsingular(vmin, vmax, expander=0.05)
                locs = self.bin_boundaries(vmin, vmax)
                locs = self.inv_rescale(locs)
                prune = self._prune
                if prune == 'lower':
                    locs = locs[1:]
                elif prune == 'upper':
                    locs = locs[:-1]
                elif prune == 'both':
                    locs = locs[1:-1]
                return self.raise_if_exceeds(locs)

        class ScaledFormatter(matplotlib.ticker.OldScalarFormatter):
            """Formats tick labels scaled by *dx* and shifted by *x0*."""

            def __init__(self, dx=1.0, x0=0.0, **kwargs):
                if False:
                    while True:
                        i = 10
                (self.dx, self.x0) = (dx, x0)

            def rescale(self, x):
                if False:
                    i = 10
                    return i + 15
                return x / self.dx + self.x0

            def __call__(self, x, pos=None):
                if False:
                    for i in range(10):
                        print('nop')
                (xmin, xmax) = self.axis.get_view_interval()
                (xmin, xmax) = (self.rescale(xmin), self.rescale(xmax))
                d = abs(xmax - xmin)
                x = self.rescale(x)
                s = self.pprint_val(x, d)
                return s
        for dim in range(2):
            if first_layer.state.output_units[dim]:
                unit = first_layer.dataset.unit(first_layer.state.expressions[dim])
                try:
                    scale = astropy.units.Unit(first_layer.state.output_units[dim]).to(unit)
                    axis = self.axes.xaxis if dim == 0 else self.axes.yaxis
                    axis.set_major_locator(ScaledLocator(dx=scale))
                    axis.set_major_formatter(ScaledFormatter(dx=scale))
                except:
                    pass
        titles = [layer.state.title for layer in self.layers if layer.state.title]
        if titles:
            self.title = self.axes.set_title(','.join(titles))
        self.post_draw(self.fig, self.axes)
        self.canvas.draw()
        self.fig.tight_layout()
        self.canvas.draw()
        self.update()
        if 0:
            if self.first_time:
                self.first_time = False
                if 'filename' in self.options:
                    self.filename_figure_last = self.options['filename']
                    self.fig.savefig(self.filename_figure_last)
        self.setMode(self.lastAction)
        self.signal_plot_finished.emit(self, self.fig)
        logger.debug('plot finished')
        if self._plot_event:
            self._plot_event.set()
        return
        if 1:
            ranges = []
            logger.debug('self.ranges == %r' % (self.ranges,))
            for (minimum, maximum) in self.ranges:
                ranges.append(minimum)
                ranges.append(maximum)
            logger.debug('expr for amplitude: %r' % self.amplitude_expression)
            grid_map = self.create_grid_map(self.state.grid_size, False)
            try:
                amplitude = self.eval_amplitude(self.amplitude_expression, locals=grid_map)
            except Exception as e:
                self.error_in_field(self.amplitude_box, 'amplitude', e)
                return
            use_selection = self.dataset.mask is not None
            if use_selection:
                grid_map_selection = self.create_grid_map(self.state.grid_size, True)
                amplitude_selection = self.eval_amplitude(self.amplitude_expression, locals=grid_map_selection)
        if self.action_display_current == self.action_display_mode_both:
            self.axes.imshow(self.contrast(amplitude), origin='lower', extent=ranges, alpha=0.4 if use_selection else 1.0, cmap=self.colormap)
            if use_selection:
                self.axes.imshow(self.contrast(amplitude_selection), origin='lower', extent=ranges, alpha=1, cmap=self.colormap)
        if self.action_display_current == self.action_display_mode_full:
            self.axes.imshow(self.contrast(amplitude), origin='lower', extent=ranges, cmap=self.colormap)
        if self.action_display_current == self.action_display_mode_selection:
            if self.counts_mask is not None:
                self.axes.imshow(self.contrast(amplitude_mask), origin='lower', extent=ranges, alpha=1, cmap=self.colormap)
        if 1:
            locals = {}
            for name in list(self.grids.grids.keys()):
                grid = self.grids.grids[name]
                if name == 'counts' or (grid.weight_expression is not None and len(grid.weight_expression) > 0):
                    if grid.max_size >= self.state.vector_grid_size:
                        locals[name] = grid.get_data(self.state.vector_grid_size, use_selection)
                else:
                    locals[name] = None
            if 1:
                grid_map_vector = self.create_grid_map(self.state.vector_grid_size, use_selection)
                if grid_map_vector['weightx'] is not None and grid_map_vector['weighty'] is not None:
                    mask = grid_map_vector['counts'] > self.min_level_vector2d * grid_map_vector['counts'].max()
                    x = grid_map_vector['x']
                    y = grid_map_vector['y']
                    (x2d, y2d) = np.meshgrid(x, y)
                    vx = self.eval_amplitude('weightx/counts', locals=grid_map_vector)
                    vy = self.eval_amplitude('weighty/counts', locals=grid_map_vector)
                    meanvx = 0 if self.vectors_subtract_mean is False else vx[mask].mean()
                    meanvy = 0 if self.vectors_subtract_mean is False else vy[mask].mean()
                    vx -= meanvx
                    vy -= meanvy
                    if grid_map_vector['weightz'] is not None and self.vectors_color_code_3rd:
                        colors = self.eval_amplitude('weightz/counts', locals=grid_map_vector)
                        self.axes.quiver(x2d[mask], y2d[mask], vx[mask], vy[mask], colors[mask], cmap=self.colormap_vector)
                    else:
                        self.axes.quiver(x2d[mask], y2d[mask], vx[mask], vy[mask], color='black')
                        colors = None
        if self.action_display_current == self.action_display_mode_both_contour:
            self.axes.imshow(amplitude, origin='lower', extent=ranges, cmap=self.colormap)
            if self.counts_mask is not None:
                values = amplitude_mask[~np.isinf(amplitude_mask)]
                levels = np.linspace(values.min(), values.max(), 5)
                self.axes.contour(amplitude_mask, origin='lower', extent=ranges, levels=levels, linewidths=2, colors='red')
        for callback in self.plugin_grids_draw:
            callback(self.axes, grid_map, grid_map_vector)
        if self.state.aspect is None:
            self.axes.set_aspect('auto')
        else:
            self.axes.set_aspect(self.state.aspect)
        if 0:
            index = self.dataset.selected_row_index
            if index is not None and self.selected_point is None:
                logger.debug('point selected but after computation')

                def find_selected_point(info, blockx, blocky):
                    if False:
                        for i in range(10):
                            print('nop')
                    if index >= info.i1 and index < info.i2:
                        self.selected_point = (blockx[index - info.i1], blocky[index - info.i1])
                self.dataset.evaluate(find_selected_point, *self.expressions, **self.getVariableDict())
            if self.selected_point:
                (x, y) = self.selected_point
                self.axes.scatter([x], [y], color='red')
        self.axes.set_xlabel(self.expressions[0])
        self.axes.set_ylabel(self.expressions[1])
        self.axes.set_xlim(*self.state.ranges_viewport[0])
        self.axes.set_ylim(*self.state.ranges_viewport[1])
        title_text = self.title_expression.format(**self.getVariableDict())
        if hasattr(self, 'title'):
            self.title.set_text(title_text)
        else:
            self.title = self.fig.suptitle(title_text)
        if not self.action_mini_mode_ultra.isChecked():
            self.fig.tight_layout(pad=0.0)
        self.fig.tight_layout()
        self.canvas.draw()
        self.update()
        if self.first_time:
            self.first_time = False
            if 'filename' in self.options:
                self.filename_figure_last = self.options['filename']
                self.fig.savefig(self.filename_figure_last)

class ScatterPlotMatrixDialog(PlotDialog):

    def __init__(self, parent, dataset, expressions):
        if False:
            for i in range(10):
                print('nop')
        super(ScatterPlotMatrixDialog, self).__init__(parent, dataset, list(expressions), 'X Y Z W V U T S R Q P'.split()[:len(expressions)])

    def getAxesList(self):
        if False:
            i = 10
            return i + 15
        return reduce(lambda x, y: x + y, self.axes_grid, [])

    def addAxes(self):
        if False:
            for i in range(10):
                print('nop')
        self.axes_grid = [[None] * self.dimensions for _ in range(self.dimensions)]
        index = 0
        for i in range(self.dimensions)[::1]:
            for j in range(self.dimensions)[::1]:
                index = (self.dimensions - 1 - j) * self.dimensions + i + 1
                axes = self.axes_grid[i][j] = self.fig.add_subplot(self.dimensions, self.dimensions, index)
                axes.xaxis_index = i
                axes.yaxis_index = j
                if i > 0:
                    axes.yaxis.set_visible(False)
                if j > 0:
                    axes.xaxis.set_visible(False)
                if int(matplotlib.__version__.split('.')[0]) < 3:
                    self.axes_grid[i][j].hold(True)
                index += 1
        self.fig.subplots_adjust(hspace=0, wspace=0)

    def calculate_visuals(self, info, *blocks):
        if False:
            print('Hello World!')
        data_blocks = blocks[:self.dimensions]
        if len(blocks) > self.dimensions:
            weights_block = blocks[self.dimensions]
        else:
            weights_block = None
        elapsed = time.time() - info.time_start
        self.message('computation %.2f%% (%f seconds)' % (info.percentage, elapsed), index=9)
        QtCore.QCoreApplication.instance().processEvents()
        self.expression_error = False
        N = self.state.grid_size
        mask = self.dataset.mask
        if info.first:
            self.counts = np.zeros((N,) * self.dimensions, dtype=np.float64)
            self.counts_weights = self.counts
            if weights_block is not None:
                self.counts_weights = np.zeros((N,) * self.dimensions, dtype=np.float64)
            self.selected_point = None
            if mask is not None:
                self.counts_mask = np.zeros((N,) * self.dimensions, dtype=np.float64)
                self.counts_weights_mask = self.counts_mask
                if weights_block is not None:
                    self.counts_weights_mask = np.zeros((N,) * self.dimensions, dtype=np.float64)
            else:
                self.counts_mask = None
                self.counts_weights_mask = None
        if info.error:
            self.expression_error = True
            self.message(info.error_text)
            return
        (xmin, xmax) = self.ranges[0]
        (ymin, ymax) = self.ranges[1]
        for i in range(self.dimensions):
            if self.state.ranges_viewport[i] is None:
                self.state.ranges_viewport[i] = self.ranges[i]
        index = self.dataset.selected_row_index
        if index is not None:
            if index >= info.i1 and index < info.i2:
                self.selected_point = (blockx[index - info.i1], blocky[index - info.i1])
        t0 = time.time()
        ranges = []
        for (minimum, maximum) in self.ranges:
            ranges.append(minimum)
            if minimum == maximum:
                maximum += 1
            ranges.append(maximum)
        try:
            args = (data_blocks, self.counts, ranges)
            if self.dimensions == 2:
                vaex.histogram.hist3d(data_blocks[0], data_blocks[1], self.counts, *ranges)
            if self.dimensions == 3:
                vaex.histogram.hist3d(data_blocks[0], data_blocks[1], data_blocks[2], self.counts, *ranges)
            if weights_block is not None:
                args = (data_blocks, weights_block, self.counts, ranges)
                vaex.histogram.hist2d_weights(blockx, blocky, self.counts_weights, weights_block, *ranges)
        except:
            raise
        if mask is not None:
            subsets = [block[mask[info.i1:info.i2]] for block in data_blocks]
            if self.dimensions == 2:
                vaex.histogram.hist2d(subsets[0], subsets[1], self.counts_weights_mask, *ranges)
            if self.dimensions == 3:
                vaex.histogram.hist3d(subsets[0], subsets[1], subsets[2], self.counts_weights_mask, *ranges)
            if weights_block is not None:
                subset_weights = weights_block[mask[info.i1:info.i2]]
                if self.dimensions == 2:
                    vaex.histogram.hist2d_weights(subsets[0], subsets[1], self.counts_weights_mask, subset_weights, *ranges)
                if self.dimensions == 3:
                    vaex.histogram.hist3d_weights(subsets[0], subsets[1], subsets[2], self.counts_weights_mask, subset_weights, *ranges)
        if info.last:
            elapsed = time.time() - info.time_start
            self.message('computation (%f seconds)' % elapsed, index=9)

    def plot(self):
        if False:
            print('Hello World!')
        t0 = time.time()
        ranges = []
        for (minimum, maximum) in self.ranges:
            ranges.append(minimum)
            ranges.append(maximum)
        amplitude = self.counts
        logger.debug('expr for amplitude: %r' % self.amplitude_expression)
        if self.amplitude_expression is not None:
            locals = {'counts': self.counts_weights, 'counts1': self.counts}
            globals = np.__dict__
            amplitude = eval(self.amplitude_expression, globals, locals)

        def multisum(a, axes):
            if False:
                i = 10
                return i + 15
            correction = 0
            for axis in axes:
                a = np.sum(a, axis=axis - correction)
                correction += 1
            return a
        for i in range(self.dimensions):
            for j in range(self.dimensions):
                axes = self.axes_grid[i][j]
                ranges = self.ranges[i] + self.ranges[j]
                axes.clear()
                allaxes = list(range(self.dimensions))
                if 0:
                    for label in axes.get_yticklabels():
                        label.set_visible(False)
                    axes.yaxis.offsetText.set_visible(False)
                if 0:
                    for label in axes.get_xticklabels():
                        label.set_visible(False)
                    axes.xaxis.offsetText.set_visible(False)
                if i != j:
                    allaxes.remove(i)
                    allaxes.remove(j)
                    counts_mask = None
                    counts = multisum(self.counts, allaxes)
                    if self.counts_mask is not None:
                        counts_mask = multisum(self.counts_mask, allaxes)
                    if i > j:
                        counts = counts.T
                    axes.imshow(np.log10(counts), origin='lower', extent=ranges, alpha=1 if counts_mask is None else 0.4)
                    if counts_mask is not None:
                        if i > j:
                            counts_mask = counts_mask.T
                        axes.imshow(np.log10(counts_mask), origin='lower', extent=ranges)
                    axes.set_aspect('auto')
                    if self.dataset.selected_row_index is not None:
                        (x, y) = (self.getdatax()[self.dataset.selected_row_index], self.getdatay()[self.dataset.selected_row_index])
                        axes.scatter([x], [y], color='red')
                    axes.set_xlim(self.state.ranges_viewport[i][0], self.state.ranges_viewport[i][1])
                    axes.set_ylim(self.state.ranges_viewport[j][0], self.state.ranges_viewport[j][1])
                else:
                    allaxes.remove(j)
                    counts = multisum(self.counts, allaxes)
                    N = len(counts)
                    (xmin, xmax) = self.ranges[i]
                    delta = (xmax - xmin) / N
                    centers = np.arange(N) * delta + xmin
                    if 1:
                        axes.bar(centers, counts, width=delta, align='center')
                    else:
                        self.axes.bar(self.centers, self.counts, width=self.delta, align='center', alpha=0.5)
                        self.axes.bar(self.centers, self.counts_mask, width=self.delta, align='center', color='red')
                    axes.set_xlim(self.state.ranges_viewport[i][0], self.state.ranges_viewport[i][1])
                    axes.set_ylim(0, np.max(counts) * 1.1)
        if 0:
            self.axes.imshow(amplitude.T, origin='lower', extent=ranges, alpha=1 if self.counts_mask is None else 0.4, cmap=cm_plusmin)
            if 1:
                if self.counts_mask is not None:
                    if self.amplitude_expression is not None:
                        locals = {'counts': self.counts_weights_mask, 'counts1': self.counts_mask}
                        globals = np.__dict__
                        amplitude_mask = eval(self.amplitude_expression, globals, locals)
                    self.axes.imshow(amplitude_mask.T, origin='lower', extent=ranges, alpha=1, cmap=cm_plusmin)
            self.axes.set_aspect('auto')
            index = self.dataset.selected_row_index
            if index is not None and self.selected_point is None:
                logger.debug('point selected but after computation')

                def find_selected_point(info, blockx, blocky):
                    if False:
                        return 10
                    if index >= info.i1 and index < info.i2:
                        self.selected_point = (blockx[index - info.i1], blocky[index - info.i1])
                self.dataset.evaluate(find_selected_point, *self.expressions, **self.getVariableDict())
            if self.selected_point:
                (x, y) = self.selected_point
                self.axes.scatter([x], [y], color='red')
            self.axes.set_xlabel(self.expressions[0])
            self.axes.set_ylabel(self.expressions[0])
            self.axes.set_xlim(*self.state.ranges_viewport[0])
            self.axes.set_ylim(*self.state.ranges_viewport[1])
        self.canvas.draw()
        self.message('ploting %f' % (time.time() - t0), index=5)
time_previous = time.time()
time_start = time.time()

def timelog(msg, reset=False):
    if False:
        for i in range(10):
            print('nop')
    global time_previous, time_start
    now = time.time()
    if reset:
        time_start = now
    T = now - time_start
    deltaT = now - time_previous
    logger.info('*** TIMELOG: %s (T=%f deltaT=%f)' % (msg, T, deltaT))
    time_previous = now

class VolumeRenderingPlotDialog(PlotDialog):
    type_name = 'volumerendering'

    def __init__(self, parent, dataset, app, **options):
        if False:
            print('Hello World!')
        super(VolumeRenderingPlotDialog, self).__init__(parent, dataset, 3, 'x y x'.split(), app, **options)

    def closeEvent(self, event):
        if False:
            i = 10
            return i + 15
        self.widget_volume.orbit_stop()
        super(VolumeRenderingPlotDialog, self).closeEvent(event)

    def afterCanvas(self, layout):
        if False:
            for i in range(10):
                print('nop')
        self.widget_volume = vaex.ui.volumerendering.VolumeRenderWidget(self)
        self.layout_plot_region.insertWidget(0, self.widget_volume, 1)
        super(VolumeRenderingPlotDialog, self).afterCanvas(layout)
        self.menu_view.addSeparator()
        self.action_group_quality_3d = QtGui.QActionGroup(self)
        self.actions_quality_3d = []
        for (index, (name, resolution, iterations)) in enumerate([('Fast', 256, 200), ('Medium', 256 + 128, 300), ('Best', 512, 500)]):
            if 'vr_quality' in self.options:
                vr_index = int(eval(self.options.get('vr_quality')))
                if index == vr_index:
                    self.widget_volume.ray_iterations = iterations
                    self.widget_volume.texture_size = resolution
            action_quality_3d = QtGui.QAction(QtGui.QIcon(iconfile('picture_empty')), name + ' volume rendering', self)

            def do(ignore=None, resolution=resolution, iterations=iterations):
                if False:
                    for i in range(10):
                        print('nop')
                self.widget_volume.set_iterations(iterations)
                self.widget_volume.setResolution(resolution)
                self.widget_volume.update()
            action_quality_3d.setCheckable(True)
            if resolution == self.widget_volume.texture_size and iterations == self.widget_volume.ray_iterations:
                action_quality_3d.setChecked(True)
            action_quality_3d.triggered.connect(do)
            action_quality_3d.setShortcut('Ctrl+Shift+Meta+%d' % (index + 1))
            self.menu_view.addAction(action_quality_3d)
            self.action_group_quality_3d.addAction(action_quality_3d)
            self.actions_quality_3d.append(action_quality_3d)

    def getAxesList(self):
        if False:
            return 10
        return [self.axis_top, self.axis_bottom]

    def add_pages(self, toolbox):
        if False:
            while True:
                i = 10
        self.frame_options_volume_rendering = QtGui.QFrame(self)
        toolbox.addItem(self.frame_options_volume_rendering, 'Volume rendering')
        toolbox.setCurrentIndex(3)
        self.fill_page_volume_rendering(self.frame_options_volume_rendering)

    def add_axes(self):
        if False:
            while True:
                i = 10
        self.axes_grid = [[None] * self.dimensions for _ in range(self.dimensions)]
        self.axis_top = self.fig.add_subplot(2, 1, 1)
        self.axis_bottom = self.fig.add_subplot(2, 1, 2)
        self.axis_top.xaxis_index = 0
        self.axis_top.yaxis_index = 1
        self.axis_bottom.xaxis_index = 0
        self.axis_bottom.yaxis_index = 2

    def add_image_layer(self, rgba, intensity):
        if False:
            print('Hello World!')
        self.image_layers.append(intensity)

    def plot(self):
        if False:
            return 10
        self.image_layers = []
        axes_list = self.getAxesList()
        for axes in axes_list:
            axes.cla()
        if len(self.layers) == 0:
            return
        first_layer = self.layers[0]
        for i in range(self.dimensions):
            if self.state.ranges_viewport[i] is None:
                self.state.ranges_viewport[i] = copy.copy(first_layer.state.ranges_grid[i])
        for axes in axes_list:
            ranges = []
            for (minimum, maximum) in [self.state.ranges_viewport[axes.xaxis_index], self.state.ranges_viewport[axes.yaxis_index]]:
                ranges.append(minimum)
                ranges.append(maximum)
            axes.rgb_images = []
            N = self.state.grid_size
            background = np.ones((N, N, 4), dtype=np.float64)
            background_color = self.background_color
            if background_color == 'auto':
                if self.blend_mode in 'screen lighten'.split():
                    background_color = 'black'
                else:
                    background_color = 'white'
            background[:, :, 0:3] = matplotlib.colors.colorConverter.to_rgb(background_color)
            background[:, :, 3] = 1.0
            axes.placeholder = axes.imshow(background, extent=ranges, origin='lower')
            axes.rgb_images.append(background)
            colors = 'red green blue'.split()
            axes.spines['bottom'].set_color(colors[axes.xaxis_index])
            axes.spines['left'].set_color(colors[axes.yaxis_index])
            linewidth = 2.0
            axes.spines['bottom'].set_linewidth(linewidth)
            axes.spines['left'].set_linewidth(linewidth)
            if self.state.aspect is None:
                axes.set_aspect('auto')
            else:
                axes.set_aspect(self.state.aspect)
            axes.set_xlabel(self.get_label(axes.xaxis_index))
            axes.set_ylabel(self.get_label(axes.yaxis_index))
        for layer in self.layers:
            layer.plot(axes_list, self.add_image_layer)
        if first_layer.amplitude_grid_selection is not None:
            self.widget_volume.setGrid(first_layer.amplitude_grid_selection.T, first_layer.amplitude_grid.T, first_layer.vector_grid)
        else:
            self.widget_volume.setGrid(first_layer.amplitude_grid.T, vectorgrid=first_layer.vector_grid)
        for axes in axes_list:
            rgba = vaex.ui.imageblending.blend(axes.rgb_images, self.blend_mode)
            rgba[..., 3] = rgba[..., 3] * 0 + 1
            for c in range(4):
                rgba[:, :, c] = np.clip(rgba[:, :, c] ** self.layer_gamma * self.layer_brightness, 0.0, 1.0)
            rgba = np.transpose(rgba, (1, 0, 2))
            axes.placeholder.set_data((rgba * 255).astype(np.uint8))
        self.fig.tight_layout()
        self.canvas.draw()
        self.update()

    def plot_(self):
        if False:
            return 10
        timelog('plot start', reset=False)
        t0 = time.time()
        if 1:
            ranges = []
            for (minimum, maximum) in self.state.ranges_viewport:
                ranges.append(minimum)
                ranges.append(maximum)
            timelog('creating grid map')
            grid_map = self.create_grid_map(self.state.grid_size, False)
            timelog('eval amplitude')
            amplitude = self.eval_amplitude(self.amplitude_expression, locals=grid_map)
            timelog('eval amplitude done')
            use_selection = self.dataset.mask is not None
            if use_selection:
                timelog('repeat for selection')
                grid_map_selection = self.create_grid_map(self.state.grid_size, True)
                amplitude_selection = self.eval_amplitude(self.amplitude_expression, locals=grid_map_selection)
            timelog('creating grid map vector')
            grid_map_vector = self.create_grid_map(self.state.vector_grid_size, use_selection)
            vector_grid = None
            vector_counts = grid_map_vector['counts']
            vector_mask = vector_counts > 0
            if grid_map_vector['weightx'] is not None:
                vector_x = grid_map_vector['x']
                vx = self.eval_amplitude('weightx/counts', locals=grid_map_vector)
            else:
                vector_x = None
                vx = None
            if grid_map_vector['weighty'] is not None:
                vector_y = grid_map_vector['y']
                vy = self.eval_amplitude('weighty/counts', locals=grid_map_vector)
            else:
                vector_y = None
                vy = None
            if grid_map_vector['weightz'] is not None:
                vector_z = grid_map_vector['z']
                vz = self.eval_amplitude('weightz/counts', locals=grid_map_vector)
            else:
                vector_z = None
                vz = None
            if vx is not None and vy is not None and (vz is not None):
                timelog('making vector grid')
                vector_grid = np.zeros((4,) + (vx.shape[0],) * 3, dtype=np.float32)
                mask = vector_counts > 0
                meanvx = 0 if self.vectors_subtract_mean is False else vx[mask].mean()
                meanvy = 0 if self.vectors_subtract_mean is False else vy[mask].mean()
                meanvz = 0 if self.vectors_subtract_mean is False else vz[mask].mean()
                vector_grid[0] = vx - meanvx
                vector_grid[1] = vy - meanvy
                vector_grid[2] = vz - meanvz
                vector_grid[3] = vector_counts
                vector_grid = np.swapaxes(vector_grid, 0, 3)
                vector_grid = vector_grid * 1.0
            timelog('setting grid')
            if use_selection:
                self.widget_volume.setGrid(amplitude_selection, amplitude, vectorgrid=vector_grid)
            else:
                self.widget_volume.setGrid(amplitude, vectorgrid=vector_grid)
            timelog('grid')
            if 0:
                self.tool.grid = amplitude
                self.tool.update()

            def multisum(a, axes):
                if False:
                    while True:
                        i = 10
                correction = 0
                for axis in axes:
                    a = np.nansum(a, axis=axis - correction)
                    correction += 1
                return a
            axeslist = self.getAxesList()
            vector_values = [vx, vy, vz]
            vector_positions = [vector_x, vector_y, vector_z]
            for i in range(2):
                timelog('axis: ' + str(i))
                axes = axeslist[i]
                i1 = 0
                i2 = i + 1
                i3 = 2 - i
                ranges = list(self.ranges[i1]) + list(self.ranges[i2])
                axes.clear()
                allaxes = list(range(self.dimensions))
                if 0:
                    for label in axes.get_yticklabels():
                        label.set_visible(False)
                    axes.yaxis.offsetText.set_visible(False)
                if 0:
                    for label in axes.get_xticklabels():
                        label.set_visible(False)
                    axes.xaxis.offsetText.set_visible(False)
                if 1:
                    allaxes.remove(2 - 0)
                    allaxes.remove(2 - (1 + i))
                    counts_mask = None
                    colors = 'red green blue'.split()
                    axes.spines['bottom'].set_color(colors[i1])
                    axes.spines['left'].set_color(colors[i2])
                    linewidth = 2.0
                    axes.spines['bottom'].set_linewidth(linewidth)
                    axes.spines['left'].set_linewidth(linewidth)
                    grid_map_2d = {key: None if grid is None else grid if grid.ndim != 3 else multisum(grid, allaxes) for (key, grid) in list(grid_map.items())}
                    amplitude = self.eval_amplitude(self.amplitude_expression, locals=grid_map_2d)
                    if use_selection:
                        grid_map_selection_2d = {key: None if grid is None else grid if grid.ndim != 3 else multisum(grid, allaxes) for (key, grid) in list(grid_map_selection.items())}
                        amplitude_selection = self.eval_amplitude(self.amplitude_expression, locals=grid_map_selection_2d)
                    axes.imshow(self.contrast(amplitude), origin='lower', extent=ranges, alpha=0.4 if use_selection else 1.0, cmap=self.colormap)
                    if use_selection:
                        axes.imshow(self.contrast(amplitude_selection), origin='lower', extent=ranges, alpha=1, cmap=self.colormap)
                    if vector_positions[i1] is not None and vector_positions[i2] is not None:
                        mask = multisum(vector_counts, allaxes) > 0
                        (x, y) = np.meshgrid(vector_positions[i1], vector_positions[i2])
                        U = multisum(vector_values[i1], allaxes)
                        V = multisum(vector_values[i2], allaxes)
                        if np.any(mask):
                            meanU = 0 if self.vectors_subtract_mean is False else np.nanmean(U[mask])
                            meanV = 0 if self.vectors_subtract_mean is False else np.nanmean(V[mask])
                            U -= meanU
                            V -= meanV
                        if vector_positions[i3] is not None and self.vectors_color_code_3rd:
                            W = multisum(vector_values[i3], allaxes)
                            if np.any(mask):
                                meanW = 0 if self.vectors_subtract_mean is False else np.nanmean(W[mask])
                                W -= meanW
                            axes.quiver(x[mask], y[mask], U[mask], V[mask], W[mask], cmap=self.colormap_vector)
                        else:
                            axes.quiver(x[mask], y[mask], U[mask], V[mask], color='black')
                    if 0:
                        (x, y) = (self.getdatax()[self.dataset.selected_row_index], self.getdatay()[self.dataset.selected_row_index])
                        axes.scatter([x], [y], color='red')
                if self.state.aspect is None:
                    axes.set_aspect('auto')
                else:
                    axes.set_aspect(self.state.aspect)
                axes.set_xlim(self.state.ranges_viewport[i1][0], self.state.ranges_viewport[i1][1])
                axes.set_ylim(self.state.ranges_viewport[i2][0], self.state.ranges_viewport[i2][1])
                axes.set_xlabel(self.expressions[i1])
                axes.set_ylabel(self.expressions[i2])
            if 0:
                self.axes.imshow(amplitude.T, origin='lower', extent=ranges, alpha=1 if self.counts_mask is None else 0.4, cmap=cm_plusmin)
                if 1:
                    if self.counts_mask is not None:
                        if self.amplitude_expression is not None:
                            locals = {'counts': self.counts_weights_mask, 'counts1': self.counts_mask}
                            globals = np.__dict__
                            amplitude_mask = eval(self.amplitude_expression, globals, locals)
                        self.axes.imshow(amplitude_mask.T, origin='lower', extent=ranges, alpha=1, cmap=cm_plusmin)
                self.axes.set_aspect('auto')
                index = self.dataset.selected_row_index
                if index is not None and self.selected_point is None:
                    logger.debug('point selected but after computation')

                    def find_selected_point(info, blockx, blocky):
                        if False:
                            return 10
                        if index >= info.i1 and index < info.i2:
                            self.selected_point = (blockx[index - info.i1], blocky[index - info.i1])
                    self.dataset.evaluate(find_selected_point, *self.expressions, **self.getVariableDict())
                if self.selected_point:
                    (x, y) = self.selected_point
                    self.axes.scatter([x], [y], color='red')
                self.axes.set_xlabel(self.expressions[0])
                self.axes.set_ylabel(self.expressions[0])
                self.axes.set_xlim(*self.state.ranges_viewport[0])
                self.axes.set_ylim(*self.state.ranges_viewport[1])
        self.canvas.draw()
        timelog('plot end')
        self.message('ploting %f' % (time.time() - t0), index=5)

class Rank1ScatterPlotDialog(ScatterPlotDialog):
    type_name = 'sequence-density2d'

    def __init__(self, parent, dataset, xname=None, yname=None):
        if False:
            print('Hello World!')
        self.nSlices = dataset.rank1s[list(dataset.rank1s.keys())[0]].shape[0]
        self.serieIndex = dataset.selected_serie_index if dataset.selected_serie_index is not None else 0
        self.record_frames = False
        super(Rank1ScatterPlotDialog, self).__init__(parent, dataset, xname, yname)

    def getTitleExpressionList(self):
        if False:
            i = 10
            return i + 15
        return ['%s: {%s: 4f}' % (name, name) for name in self.dataset.axis_names]

    def addToolbar2(self, layout, contrast=True, gamma=True):
        if False:
            i = 10
            return i + 15
        super(Rank1ScatterPlotDialog, self).addToolbar2(layout, contrast, gamma)
        self.action_save_frames = QtGui.QAction(QtGui.QIcon(iconfile('film')), '&Export frames', self)
        self.menu_save.addAction(self.action_save_frames)
        self.action_save_frames.triggered.connect(self.onActionSaveFrames)

    def onActionSaveFrames(self, ignore=None):
        if False:
            return 10
        directory = qt.getdir(self, 'Choose where to save frames', '')
        self.frame_template = os.path.join(directory, '%s_{index:05}.png' % self.dataset.name)
        self.frame_template = qt.gettext(self, 'template for frame filenames', 'template:', self.frame_template)
        self.record_frames = True
        self.onPlayOnce()

    def plot(self):
        if False:
            print('Hello World!')
        super(Rank1ScatterPlotDialog, self).plot()
        if self.record_frames:
            index = self.serieIndex
            path = self.frame_template.format(**locals())
            self.fig.savefig(path)
            if self.serieIndex == self.nSlices - 1:
                self.record_frames = False

    def onSerieIndexSelect(self, serie_index):
        if False:
            for i in range(10):
                print('nop')
        if serie_index != self.serieIndex:
            self.serieIndex = serie_index
            self.seriesbox.setCurrentIndex(self.serieIndex)
        else:
            self.serieIndex = serie_index
        self.compute()

    def getExpressionList(self):
        if False:
            while True:
                i = 10
        names = []
        for rank1name in self.dataset.rank1names:
            names.append(rank1name + '[index]')
        return names

    def getVariableDict(self):
        if False:
            return 10
        vars = {'index': self.serieIndex}
        for name in self.dataset.axis_names:
            vars[name] = self.dataset.axes[name][self.serieIndex]
        return vars

    def _getVariableDictMinMax(self):
        if False:
            return 10
        return {'index': slice(None, None, None)}

    def afterCanvas(self, layout):
        if False:
            i = 10
            return i + 15
        super(Rank1ScatterPlotDialog, self).afterCanvas(layout)
        self.seriesbox = QtGui.QComboBox(self)
        self.seriesbox.addItems([str(k) for k in range(self.nSlices)])
        self.seriesbox.setCurrentIndex(self.serieIndex)
        self.seriesbox.currentIndexChanged.connect(self.onSerieIndex)
        self.grid_layout.addWidget(self.seriesbox, 10, 1)

    def onPlayOnce(self):
        if False:
            i = 10
            return i + 15
        self.delay = 10
        if not self.state.axis_lock:
            for i in range(self.dimensions):
                self.ranges[i] = None
            for i in range(self.dimensions):
                self.state.ranges_viewport[i] = None
        self.dataset.selectSerieIndex(0)
        self.dataset.executor.execute()
        QtCore.QTimer.singleShot(self.delay if not self.record_frames else 0, self.onNextFrame)

    def onNextFrame(self, *args):
        if False:
            while True:
                i = 10
        step = 1
        next = self.serieIndex + step
        if next >= self.nSlices:
            next = self.nSlices - 1
        if not self.state.axis_lock:
            for i in range(self.dimensions):
                self.ranges[i] = None
            for i in range(self.dimensions):
                self.state.ranges_viewport[i] = None
        self.dataset.selectSerieIndex(next)
        self.dataset.executor.execute()
        if self.serieIndex < self.nSlices - 1:
            QtCore.QTimer.singleShot(self.delay, self.onNextFrame)

    def onSerieIndex(self, index):
        if False:
            for i in range(10):
                print('nop')
        if index != self.dataset.selected_serie_index:
            if not self.state.axis_lock:
                for i in range(self.dimensions):
                    self.ranges[i] = None
                for i in range(self.dimensions):
                    self.state.ranges_viewport[i] = None
            self.dataset.selectSerieIndex(index)
            self.dataset.executor.execute()

class Mover(object):

    def __init__(self, plot, axes):
        if False:
            print('Hello World!')
        self.plot = plot
        self.axes = axes
        self.canvas = self.axes.figure.canvas
        self.axes = None
        self.canvas.mpl_connect('scroll_event', self.mpl_scroll)
        (self.last_x, self.last_y) = (None, None)
        self.handles = []
        self.handles.append(self.canvas.mpl_connect('motion_notify_event', self.mouse_move))
        self.handles.append(self.canvas.mpl_connect('button_press_event', self.mouse_down))
        self.handles.append(self.canvas.mpl_connect('button_release_event', self.mouse_up))
        (self.begin_x, self.begin_y) = (None, None)
        self.moved = False
        self.zoom_queue = []
        self.zoom_counter = 0

    def disconnect_events(self):
        if False:
            i = 10
            return i + 15
        for handle in self.handles:
            self.canvas.mpl_disconnect(handle)

    def mouse_up(self, event):
        if False:
            return 10
        (self.last_x, self.last_y) = (None, None)
        if self.moved:
            self.plot.queue_history_change('pan')
            self.plot.update_all_layers()
            self.moved = False

    def mouse_down(self, event):
        if False:
            print('Hello World!')
        self.moved = False
        if event.dblclick:
            factor = 0.333
            if event.button != 1:
                factor = 1 / factor
            self.plot.zoom(factor, axes=event.inaxes, x=event.xdata, y=event.ydata)
            self.plot.queue_history_change('zoom in around (%s, %s)' % (event.xdata, event.ydata))
        else:
            (self.begin_x, self.begin_y) = (event.xdata, event.ydata)
            (self.last_x, self.last_y) = (event.xdata, event.ydata)
            self.current_axes = event.inaxes
            self.plot.ranges_begin = list(self.plot.state.ranges_viewport)

    def mouse_move(self, event):
        if False:
            for i in range(10):
                print('nop')
        if self.last_x is not None and event.xdata is not None and (self.current_axes is not None):
            transform = self.current_axes.transData.inverted().transform
            (x_data, y_data) = (event.xdata, event.ydata)
            self.moved = True
            dx = self.last_x - x_data
            dy = self.last_y - y_data
            if self.plot.state.ranges_viewport is None or None in self.plot.state.ranges_viewport:
                return
            (xmin, xmax) = (self.plot.state.ranges_viewport[self.current_axes.xaxis_index][0] + dx, self.plot.state.ranges_viewport[self.current_axes.xaxis_index][1] + dx)
            if self.plot.dimensions == 1:
                (ymin, ymax) = (self.plot.state.range_level_show[0] + dy, self.plot.state.range_level_show[1] + dy)
            else:
                (ymin, ymax) = (self.plot.state.ranges_viewport[self.current_axes.yaxis_index][0] + dy, self.plot.state.ranges_viewport[self.current_axes.yaxis_index][1] + dy)
            self.plot.state.ranges_viewport[self.current_axes.xaxis_index] = [xmin, xmax]
            if self.plot.dimensions == 1:
                self.plot.state.range_level_show = [ymin, ymax]
            else:
                self.plot.state.ranges_viewport[self.current_axes.yaxis_index] = [ymin, ymax]
            for axes in self.plot.getAxesList():
                if self.plot.dimensions == 1:
                    axes.set_xlim(*self.plot.state.ranges_viewport[self.current_axes.xaxis_index])
                    axes.set_ylim(*self.plot.state.range_level_show)
                else:
                    if axes.xaxis_index == self.current_axes.xaxis_index:
                        axes.set_xlim(*self.plot.state.ranges_viewport[self.current_axes.xaxis_index])
                    if axes.yaxis_index == self.current_axes.xaxis_index:
                        axes.set_ylim(*self.plot.state.ranges_viewport[self.current_axes.xaxis_index])
                    if axes.xaxis_index == self.current_axes.yaxis_index:
                        axes.set_xlim(*self.plot.state.ranges_viewport[self.current_axes.yaxis_index])
                    if axes.yaxis_index == self.current_axes.yaxis_index:
                        axes.set_ylim(*self.plot.state.ranges_viewport[self.current_axes.yaxis_index])
            transform = self.current_axes.transData.inverted().transform
            (x_data, y_data) = transform([event.x * 1.0, event.y * 1])
            (self.last_x, self.last_y) = (x_data, y_data)
            self.canvas.draw_idle()

    def mpl_scroll(self, event):
        if False:
            while True:
                i = 10
        factor = 10 ** (-event.step / 8)
        self.zoom_counter += 1
        if event.inaxes is None:
            return
        if factor < 1:
            self.plot.zoom(factor, event.inaxes, event.xdata, event.ydata)
        else:
            self.plot.zoom(factor, event.inaxes, event.xdata, event.ydata)
        return

        def idle_zoom(ignore=None, zoom_counter=None, axes=None):
            if False:
                while True:
                    i = 10
            if zoom_counter < self.zoom_counter:
                pass
            else:
                for (i, (factor, x, y)) in enumerate(self.zoom_queue):
                    is_last = i == len(self.zoom_queue) - 1
                    self.plot.zoom(factor, axes=axes, x=x, y=y, recalculate=False, history=False, redraw=is_last)
                self.zoom_queue = []
        if event.axes:
            QtCore.QTimer.singleShot(1, functools.partial(idle_zoom, zoom_counter=self.zoom_counter, axes=event.inaxes))

class Queue(object):
    logger = logging.getLogger('vaex.ui.queue')

    def __init__(self, name, default_delay, default_callable, pre=lambda : None):
        if False:
            i = 10
            return i + 15
        self.name = name
        self.default_delay = default_delay
        self.counter = 0
        self.counter_processed = 0
        self.default_callable = default_callable
        self.pre = pre

    def is_empty(self):
        if False:
            for i in range(10):
                print('nop')
        return self.counter_processed == self.counter

    def in_queue(self, minimum=1):
        if False:
            return 10
        return self.counter_processed <= self.counter - minimum

    def _wait(self, sleep=10):
        if False:
            while True:
                i = 10
        if self.counter > 0:
            qt_app = QtCore.QCoreApplication.instance()
            logger.debug('*** waiting for queue %r' % self.name)
            while self.counter_processed != self.counter:
                logger.debug('counter=%i counter_processed=%i', self.counter, self.counter_processed)
                qt_app.processEvents()
                QtTest.QTest.qSleep(sleep)
            logger.debug('*** done with queue %r' % self.name)

    def cancel(self):
        if False:
            print('Hello World!')

        def nop():
            if False:
                print('Hello World!')
            self.logger.debug('nop called in queue %r' % self.name)
        self.logger.debug('cancelling last call in queue %r by inserting a nop' % self.name)
        self(nop)

    def __call__(self, callable=None, delay=None, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.pre()
        if delay is None:
            delay = self.default_delay
        callable = callable or self.default_callable

        def call(_=None, counter=None, callable=None):
            if False:
                for i in range(10):
                    print('nop')
            try:
                if counter < self.counter:
                    pass
                    self.logger.debug('ignoring this event in queue %r, since a new one is scheduled' % self.name)
                else:
                    self.logger.debug('calling callback in queue %r' % self.name)
                    callable()
                    self.counter_processed = self.counter - 1
            finally:
                self.counter_processed = counter
        callable = functools.partial(callable, *args, **kwargs)
        self.counter += 1
        self.logger.debug('add in queue %r %r %s' % (self.name, delay, threading.currentThread()))
        if delay == 0:
            call(counter=self.counter, callable=callable)
        else:
            QtCore.QTimer.singleShot(delay, functools.partial(call, counter=self.counter, callable=callable))
from vaex.ui.layers import LayerTable
import vaex.ui.layers
try:
    from vaex.ui.main import VaexApp
except:
    pass