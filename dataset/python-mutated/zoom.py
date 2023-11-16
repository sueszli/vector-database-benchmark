import functools
import matplotlib.widgets
import vaex.ui.plugin
from vaex.ui import undo
from vaex.ui.qt import *
from vaex.ui.icons import iconfile
import logging
import vaex.ui.undo as undo
logger = logging.getLogger('plugin.zoom')

@vaex.ui.plugin.pluginclass
class ZoomPlugin(vaex.ui.plugin.PluginPlot):
    name = 'zoom'

    def __init__(self, dialog):
        if False:
            for i in range(10):
                print('nop')
        super(ZoomPlugin, self).__init__(dialog)
        dialog.plug_toolbar(self.plug_toolbar, 1.2)

    def plug_toolbar(self):
        if False:
            return 10
        logger.info('adding zoom plugin')
        self.dialog.menu_mode.addSeparator()
        self.action_zoom_rect = QtGui.QAction(QtGui.QIcon(iconfile('zoom')), '&Zoom to rect', self.dialog)
        self.action_zoom_rect.setShortcut('Ctrl+Alt+Z')
        self.dialog.menu_mode.addAction(self.action_zoom_rect)
        self.action_zoom_x = QtGui.QAction(QtGui.QIcon(iconfile('zoom_x')), '&Zoom x', self.dialog)
        self.action_zoom_y = QtGui.QAction(QtGui.QIcon(iconfile('zoom_y')), '&Zoom y', self.dialog)
        self.action_zoom = QtGui.QAction(QtGui.QIcon(iconfile('zoom')), '&Zoom(you should not read this)', self.dialog)
        self.action_zoom_x.setShortcut('Ctrl+Alt+X')
        self.action_zoom_y.setShortcut('Ctrl+Alt+Y')
        self.dialog.menu_mode.addAction(self.action_zoom_x)
        self.dialog.menu_mode.addAction(self.action_zoom_y)
        self.dialog.menu_mode.addSeparator()
        self.action_zoom_out = QtGui.QAction(QtGui.QIcon(iconfile('zoom_out')), '&Zoom out', self.dialog)
        self.action_zoom_in = QtGui.QAction(QtGui.QIcon(iconfile('zoom_in')), '&Zoom in', self.dialog)
        self.action_zoom_fit = QtGui.QAction(QtGui.QIcon(iconfile('arrow_out')), '&Reset view', self.dialog)
        self.action_zoom_out.setShortcut('Ctrl+Alt+-')
        self.action_zoom_in.setShortcut('Ctrl+Alt++')
        self.action_zoom_fit.setShortcut('Ctrl+Alt+0')
        self.dialog.menu_mode.addAction(self.action_zoom_out)
        self.dialog.menu_mode.addAction(self.action_zoom_in)
        self.dialog.menu_mode.addAction(self.action_zoom_fit)
        self.dialog.action_group_main.addAction(self.action_zoom_rect)
        self.dialog.action_group_main.addAction(self.action_zoom_x)
        self.dialog.action_group_main.addAction(self.action_zoom_y)
        self.dialog.toolbar.addAction(self.action_zoom)
        self.zoom_menu = QtGui.QMenu()
        self.action_zoom.setMenu(self.zoom_menu)
        self.zoom_menu.addAction(self.action_zoom_rect)
        self.zoom_menu.addAction(self.action_zoom_x)
        self.zoom_menu.addAction(self.action_zoom_y)
        if self.dialog.dimensions == 1:
            self.lastActionZoom = self.action_zoom_x
        else:
            self.lastActionZoom = self.action_zoom_rect
        self.dialog.toolbar.addSeparator()
        self.dialog.toolbar.addAction(self.action_zoom_fit)
        self.action_zoom.triggered.connect(self.onActionZoom)
        self.action_zoom_out.triggered.connect(self.onZoomOut)
        self.action_zoom_in.triggered.connect(self.onZoomIn)
        self.action_zoom_fit.triggered.connect(self.onZoomFit)
        self.action_zoom.setCheckable(True)
        self.action_zoom_rect.setCheckable(True)
        self.action_zoom_x.setCheckable(True)
        self.action_zoom_y.setCheckable(True)

    def setMode(self, action):
        if False:
            for i in range(10):
                print('nop')
        useblit = True
        axes_list = self.dialog.getAxesList()
        if action == self.action_zoom_x:
            print('zoom x')
            self.lastActionZoom = self.action_zoom_x
            self.dialog.currentModes = [matplotlib.widgets.SpanSelector(axes, functools.partial(self.onZoomX, axes=axes), 'horizontal', useblit=useblit) for axes in axes_list]
            if useblit:
                self.dialog.canvas.draw()
        if action == self.action_zoom_y:
            self.lastActionZoom = self.action_zoom_y
            self.dialog.currentModes = [matplotlib.widgets.SpanSelector(axes, functools.partial(self.onZoomY, axes=axes), 'vertical', useblit=useblit) for axes in axes_list]
            if useblit:
                self.dialog.canvas.draw()
        if action == self.action_zoom_rect:
            print('zoom rect')
            self.lastActionZoom = self.action_zoom_rect
            self.dialog.currentModes = [matplotlib.widgets.RectangleSelector(axes, functools.partial(self.onZoomRect, axes=axes), useblit=useblit) for axes in axes_list]
            if useblit:
                self.dialog.canvas.draw()

    def onZoomIn(self, *args):
        if False:
            while True:
                i = 10
        axes = self.getAxesList()[0]
        self.dialog.zoom(0.5, axes)
        self.dialog.queue_history_change('zoom in')

    def onZoomOut(self):
        if False:
            for i in range(10):
                print('nop')
        axes = self.dialog.getAxesList()[0]
        self.dialog.zoom(2.0, axes)
        self.dialog.queue_history_change('zoom out')

    def onActionZoom(self):
        if False:
            print('Hello World!')
        print('onactionzoom')
        self.lastActionZoom.setChecked(True)
        self.dialog.setMode(self.lastActionZoom)
        self.syncToolbar()

    def onZoomFit(self, *args):
        if False:
            while True:
                i = 10
        if 0:
            for axisIndex in range(self.dimensions):
                linkButton = self.linkButtons[axisIndex]
                link = linkButton.link
                if link:
                    logger.debug('sending link messages')
                    link.sendRanges(self.dialog.ranges[axisIndex], linkButton)
                    link.sendRangesShow(self.dialog.state.ranges_viewport[axisIndex], linkButton)
        action = undo.ActionZoom(self.dialog.undoManager, 'zoom to fit', self.dialog.set_ranges, list(range(self.dialog.dimensions)), self.dialog.state.ranges_viewport, self.dialog.state.range_level_show, list(range(self.dialog.dimensions)), ranges_viewport=[None] * self.dialog.dimensions, range_level_show=None)
        action.do()
        self.dialog.checkUndoRedo()
        self.dialog.queue_history_change('zoom to fit')
        if 0:
            linked_buttons = [button for button in self.linkButtons if button.link is not None]
            links = [button.link for button in linked_buttons]
            if len(linked_buttons) > 0:
                logger.debug('sending compute message')
                vaex.dataset.Link.sendCompute(links, linked_buttons)

    def onZoomUse(self, *args):
        if False:
            return 10
        for i in range(self.dimensions):
            self.dialog.ranges[i] = self.dialog.state.ranges_viewport[i]
        self.range_level = None
        for axisIndex in range(self.dimensions):
            linkButton = self.linkButtons[axisIndex]
            link = linkButton.link
            if link:
                logger.debug('sending link messages')
                link.sendRanges(self.dialog.ranges[axisIndex], linkButton)
        linked_buttons = [button for button in self.linkButtons if button.link is not None]
        links = [button.link for button in linked_buttons]
        if len(linked_buttons) > 0:
            logger.debug('sending compute message')
            vaex.dataset.Link.sendCompute(links, linked_buttons)
        self.compute()
        self.dataset.executor.execute()

    def onZoomX(self, xmin, xmax, axes):
        if False:
            while True:
                i = 10
        axisIndex = axes.xaxis_index
        if 0:
            linkButton = self.linkButtons[axisIndex]
            link = linkButton.link
            if link:
                logger.debug('sending link messages')
                link.sendRangesShow(self.dialog.state.ranges_viewport[axisIndex], linkButton)
                link.sendPlot(linkButton)
        action = undo.ActionZoom(self.dialog.undoManager, 'zoom x [%f,%f]' % (xmin, xmax), self.dialog.set_ranges, list(range(self.dialog.dimensions)), self.dialog.state.ranges_viewport, self.dialog.state.range_level_show, [axisIndex], ranges_viewport=[[xmin, xmax]])
        action.do()
        self.dialog.checkUndoRedo()
        self.dialog.queue_history_change('zoom x')

    def onZoomY(self, ymin, ymax, axes):
        if False:
            return 10
        if len(self.dialog.state.ranges_viewport) == 1:
            action = undo.ActionZoom(self.dialog.undoManager, 'change level [%f,%f]' % (ymin, ymax), self.dialog.set_ranges, list(range(self.dialog.dimensions)), self.dialog.state.ranges_viewport, self.dialog.state.range_level_show, [], range_level_show=[ymin, ymax])
        else:
            action = undo.ActionZoom(self.dialog.undoManager, 'zoom y [%f,%f]' % (ymin, ymax), self.dialog.set_ranges, list(range(self.dialog.dimensions)), self.dialog.state.ranges_viewport, self.dialog.state.range_level_show, [axes.yaxis_index], ranges_viewport=[[ymin, ymax]])
        action.do()
        self.dialog.checkUndoRedo()
        self.dialog.queue_history_change('zoom y')

    def onZoomRect(self, eclick, erelease, axes):
        if False:
            while True:
                i = 10
        (x1, y1) = (eclick.xdata, eclick.ydata)
        (x2, y2) = (erelease.xdata, erelease.ydata)
        x = [x1, x2]
        y = [y1, y2]
        range_level = None
        ranges_show = []
        ranges = []
        axis_indices = []
        (xmin_show, xmax_show) = (min(x), max(x))
        (ymin_show, ymax_show) = (min(y), max(y))
        if self.dialog.state.ranges_viewport[0][0] > self.dialog.state.ranges_viewport[0][1]:
            (xmin_show, xmax_show) = (xmax_show, xmin_show)
        if len(self.dialog.state.ranges_viewport) == 1 and self.dialog.state.range_level_show[0] > self.dialog.state.range_level_show[1]:
            (ymin_show, ymax_show) = (ymax_show, ymin_show)
        elif self.dialog.state.ranges_viewport[1][0] > self.dialog.state.ranges_viewport[1][1]:
            (ymin_show, ymax_show) = (ymax_show, ymin_show)
        axis_indices.append(axes.xaxis_index)
        ranges_show.append([xmin_show, xmax_show])
        if len(self.dialog.state.ranges_viewport) == 1:
            range_level = (ymin_show, ymax_show)
            logger.debug('range refers to level: %r' % (self.dialog.state.range_level_show,))
        else:
            axis_indices.append(axes.yaxis_index)
            ranges_show.append([ymin_show, ymax_show])

        def delayed_zoom():
            if False:
                while True:
                    i = 10
            action = undo.ActionZoom(self.dialog.undoManager, 'zoom to rect', self.dialog.set_ranges, list(range(self.dialog.dimensions)), self.dialog.state.ranges_viewport, self.dialog.state.range_level_show, axis_indices, ranges_viewport=ranges_show, range_level_show=range_level)
            action.do()
            self.dialog.checkUndoRedo()
        delayed_zoom()
        self.dialog.queue_history_change('zoom to rectangle')
        if 1:
            self.dialog.state.ranges_viewport[axes.xaxis_index] = list(ranges_show[0])
            if self.dialog.dimensions == 2:
                self.dialog.state.ranges_viewport[axes.yaxis_index] = list(ranges_show[1])
                self.dialog.check_aspect(1)
                axes.set_xlim(self.dialog.state.ranges_viewport[0])
                axes.set_ylim(self.dialog.state.ranges_viewport[1])
            if self.dialog.dimensions == 1:
                self.dialog.state.range_level_show = range_level
                axes.set_xlim(self.dialog.state.ranges_viewport[0])
                axes.set_ylim(self.dialog.state.range_level_show)
            self.dialog.queue_redraw()
        if 0:
            for axisIndex in range(self.dimensions):
                linkButton = self.linkButtons[axisIndex]
                link = linkButton.link
                if link:
                    logger.debug('sending link messages')
                    link.sendRangesShow(self.dialog.state.ranges_viewport[axisIndex], linkButton)
                    link.sendPlot(linkButton)
            action = undo.ActionZoom(self.undoManager, 'zoom to rect', self.set_ranges, list(range(self.dimensions)), self.dialog.ranges, self.dialog.state.ranges_viewport, self.range_level, axis_indices, ranges_viewport=ranges_show, range_level=range_level)
            action.do()
            self.checkUndoRedo()
            if 0:
                if self.autoRecalculate():
                    for i in range(self.dimensions):
                        self.dialog.ranges[i] = self.dialog.state.ranges_viewport[i]
                        self.range_level = None
                    self.compute()
                    self.dataset.executor.execute()
                else:
                    self.plot()

    def syncToolbar(self):
        if False:
            while True:
                i = 10
        for action in [self.action_zoom]:
            logger.debug('sync action: %r' % action.text())
            subactions = action.menu().actions()
            subaction_selected = [subaction for subaction in subactions if subaction.isChecked()]
            logger.debug(' subaction_selected: %r' % subaction_selected)
            logger.debug(' action was selected?: %r' % action.isChecked())
            action.setChecked(len(subaction_selected) > 0)
            logger.debug(' action  is selected?: %r' % action.isChecked())
        logger.debug('last zoom action: %r' % self.lastActionZoom.text())
        self.action_zoom.setText(self.lastActionZoom.text())
        self.action_zoom.setIcon(self.lastActionZoom.icon())