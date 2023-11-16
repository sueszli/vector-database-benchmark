import weakref
from math import ceil, floor, isfinite, log10, sqrt, frexp, floor
import numpy as np
from .. import debug as debug
from .. import functions as fn
from .. import getConfigOption
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from .GraphicsWidget import GraphicsWidget
__all__ = ['AxisItem']

class AxisItem(GraphicsWidget):
    """
    GraphicsItem showing a single plot axis with ticks, values, and label.
    Can be configured to fit on any side of a plot, 
    Can automatically synchronize its displayed scale with ViewBox items.
    Ticks can be extended to draw a grid.
    If maxTickLength is negative, ticks point into the plot.
    """

    def __init__(self, orientation, pen=None, textPen=None, tickPen=None, linkView=None, parent=None, maxTickLength=-5, showValues=True, text='', units='', unitPrefix='', **args):
        if False:
            for i in range(10):
                print('nop')
        "\n        =============== ===============================================================\n        **Arguments:**\n        orientation     one of 'left', 'right', 'top', or 'bottom'\n        maxTickLength   (px) maximum length of ticks to draw. Negative values draw\n                        into the plot, positive values draw outward.\n        linkView        (ViewBox) causes the range of values displayed in the axis\n                        to be linked to the visible range of a ViewBox.\n        showValues      (bool) Whether to display values adjacent to ticks\n        pen             (QPen) Pen used when drawing axis and (by default) ticks\n        textPen         (QPen) Pen used when drawing tick labels.\n        tickPen         (QPen) Pen used when drawing ticks.\n        text            The text (excluding units) to display on the label for this\n                        axis.\n        units           The units for this axis. Units should generally be given\n                        without any scaling prefix (eg, 'V' instead of 'mV'). The\n                        scaling prefix will be automatically prepended based on the\n                        range of data displayed.\n        args            All extra keyword arguments become CSS style options for\n                        the <span> tag which will surround the axis label and units.\n        =============== ===============================================================\n        "
        GraphicsWidget.__init__(self, parent)
        self.label = QtWidgets.QGraphicsTextItem(self)
        self.picture = None
        self.orientation = orientation
        if orientation not in ['left', 'right', 'top', 'bottom']:
            raise Exception("Orientation argument must be one of 'left', 'right', 'top', or 'bottom'.")
        if orientation in ['left', 'right']:
            self.label.setRotation(-90)
            hide_overlapping_labels = False
        else:
            hide_overlapping_labels = True
        self.style = {'tickTextOffset': [5, 2], 'tickTextWidth': 30, 'tickTextHeight': 18, 'autoExpandTextSpace': True, 'autoReduceTextSpace': True, 'hideOverlappingLabels': hide_overlapping_labels, 'tickFont': None, 'stopAxisAtTick': (False, False), 'textFillLimits': [(0, 0.8), (2, 0.6), (4, 0.4), (6, 0.2)], 'showValues': showValues, 'tickLength': maxTickLength, 'maxTickLevel': 2, 'maxTextLevel': 2, 'tickAlpha': None}
        self.textWidth = 30
        self.textHeight = 18
        self.fixedWidth = None
        self.fixedHeight = None
        self.labelText = text
        self.labelUnits = units
        self.labelUnitPrefix = unitPrefix
        self.labelStyle = args
        self.logMode = False
        self._tickDensity = 1.0
        self._tickLevels = None
        self._tickSpacing = None
        self.scale = 1.0
        self.autoSIPrefix = True
        self.autoSIPrefixScale = 1.0
        self.showLabel(False)
        self.setRange(0, 1)
        if pen is None:
            self.setPen()
        else:
            self.setPen(pen)
        if textPen is None:
            self.setTextPen()
        else:
            self.setTextPen(textPen)
        if tickPen is None:
            self.setTickPen()
        else:
            self.setTickPen(tickPen)
        self._linkedView = None
        if linkView is not None:
            self._linkToView_internal(linkView)
        self.grid = False

    def setStyle(self, **kwds):
        if False:
            while True:
                i = 10
        "\n        Set various style options.\n\n        ===================== =======================================================\n        Keyword Arguments:\n        tickLength            (int) The maximum length of ticks in pixels.\n                              Positive values point toward the text; negative\n                              values point away.\n        tickTextOffset        (int) reserved spacing between text and axis in px\n        tickTextWidth         (int) Horizontal space reserved for tick text in px\n        tickTextHeight        (int) Vertical space reserved for tick text in px\n        autoExpandTextSpace   (bool) Automatically expand text space if the tick\n                              strings become too long.\n        autoReduceTextSpace   (bool) Automatically shrink the axis if necessary\n        hideOverlappingLabels (bool or int)\n\n                              * *True*  (default for horizontal axis): Hide tick labels which extend beyond the AxisItem's geometry rectangle.\n                              * *False* (default for vertical axis): Labels may be drawn extending beyond the extent of the axis.\n                              * *(int)* sets the tolerance limit for how many pixels a label is allowed to extend beyond the axis. Defaults to 15 for `hideOverlappingLabels = False`.\n\n        tickFont              (QFont or None) Determines the font used for tick\n                              values. Use None for the default font.\n        stopAxisAtTick        (tuple: (bool min, bool max)) If True, the axis\n                              line is drawn only as far as the last tick.\n                              Otherwise, the line is drawn to the edge of the\n                              AxisItem boundary.\n        textFillLimits        (list of (tick #, % fill) tuples). This structure\n                              determines how the AxisItem decides how many ticks\n                              should have text appear next to them. Each tuple in\n                              the list specifies what fraction of the axis length\n                              may be occupied by text, given the number of ticks\n                              that already have text displayed. For example::\n\n                                  [(0, 0.8), # Never fill more than 80% of the axis\n                                   (2, 0.6), # If we already have 2 ticks with text,\n                                             # fill no more than 60% of the axis\n                                   (4, 0.4), # If we already have 4 ticks with text,\n                                             # fill no more than 40% of the axis\n                                   (6, 0.2)] # If we already have 6 ticks with text,\n                                             # fill no more than 20% of the axis\n\n        showValues            (bool) indicates whether text is displayed adjacent\n                              to ticks.\n        tickAlpha             (float or int or None) If None, pyqtgraph will draw the\n                              ticks with the alpha it deems appropriate.  Otherwise,\n                              the alpha will be fixed at the value passed.  With int,\n                              accepted values are [0..255].  With value of type\n                              float, accepted values are from [0..1].\n        ===================== =======================================================\n\n        Added in version 0.9.9\n        "
        for (kwd, value) in kwds.items():
            if kwd not in self.style:
                raise NameError('%s is not a valid style argument.' % kwd)
            if kwd in ('tickLength', 'tickTextOffset', 'tickTextWidth', 'tickTextHeight'):
                if not isinstance(value, int):
                    raise ValueError("Argument '%s' must be int" % kwd)
            if kwd == 'tickTextOffset':
                if self.orientation in ('left', 'right'):
                    self.style['tickTextOffset'][0] = value
                else:
                    self.style['tickTextOffset'][1] = value
            elif kwd == 'stopAxisAtTick':
                try:
                    assert len(value) == 2 and isinstance(value[0], bool) and isinstance(value[1], bool)
                except:
                    raise ValueError("Argument 'stopAxisAtTick' must have type (bool, bool)")
                self.style[kwd] = value
            else:
                self.style[kwd] = value
        self.picture = None
        self._adjustSize()
        self.update()

    def close(self):
        if False:
            return 10
        self.scene().removeItem(self.label)
        self.label = None
        self.scene().removeItem(self)

    def setGrid(self, grid):
        if False:
            for i in range(10):
                print('nop')
        'Set the alpha value (0-255) for the grid, or False to disable.\n\n        When grid lines are enabled, the axis tick lines are extended to cover\n        the extent of the linked ViewBox, if any.\n        '
        self.grid = grid
        self.picture = None
        self.prepareGeometryChange()
        self.update()

    def setLogMode(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Set log scaling for x and/or y axes.\n\n        If two positional arguments are provided, the first will set log scaling\n        for the x axis and the second for the y axis. If a single positional\n        argument is provided, it will set the log scaling along the direction of\n        the AxisItem. Alternatively, x and y can be passed as keyword arguments.\n\n        If an axis is set to log scale, ticks are displayed on a logarithmic scale\n        and values are adjusted accordingly. (This is usually accessed by changing\n        the log mode of a :func:`PlotItem <pyqtgraph.PlotItem.setLogMode>`.) The \n        linked ViewBox will be informed of the change.\n        '
        if len(args) == 1:
            self.logMode = args[0]
        else:
            if len(args) == 2:
                (x, y) = args
            else:
                x = kwargs.get('x')
                y = kwargs.get('y')
            if x is not None and self.orientation in ('top', 'bottom'):
                self.logMode = x
            if y is not None and self.orientation in ('left', 'right'):
                self.logMode = y
        if self._linkedView is not None:
            if self.orientation in ('top', 'bottom'):
                self._linkedView().setLogMode('x', self.logMode)
            elif self.orientation in ('left', 'right'):
                self._linkedView().setLogMode('y', self.logMode)
        self.picture = None
        self.update()

    def setTickFont(self, font):
        if False:
            for i in range(10):
                print('nop')
        '\n        (QFont or None) Determines the font used for tick values. \n        Use None for the default font.\n        '
        self.style['tickFont'] = font
        self.picture = None
        self.prepareGeometryChange()
        self.update()

    def resizeEvent(self, ev=None):
        if False:
            i = 10
            return i + 15
        nudge = 5
        if self.label is None:
            self.picture = None
            return
        br = self.label.boundingRect()
        p = QtCore.QPointF(0, 0)
        if self.orientation == 'left':
            p.setY(int(self.size().height() / 2 + br.width() / 2))
            p.setX(-nudge)
        elif self.orientation == 'right':
            p.setY(int(self.size().height() / 2 + br.width() / 2))
            p.setX(int(self.size().width() - br.height() + nudge))
        elif self.orientation == 'top':
            p.setY(-nudge)
            p.setX(int(self.size().width() / 2.0 - br.width() / 2.0))
        elif self.orientation == 'bottom':
            p.setX(int(self.size().width() / 2.0 - br.width() / 2.0))
            p.setY(int(self.size().height() - br.height() + nudge))
        self.label.setPos(p)
        self.picture = None

    def showLabel(self, show=True):
        if False:
            i = 10
            return i + 15
        'Show/hide the label text for this axis.'
        self.label.setVisible(show)
        if self.orientation in ['left', 'right']:
            self._updateWidth()
        else:
            self._updateHeight()
        if self.autoSIPrefix:
            self.updateAutoSIPrefix()

    def setLabel(self, text=None, units=None, unitPrefix=None, **args):
        if False:
            return 10
        'Set the text displayed adjacent to the axis.\n\n        ==============  =============================================================\n        **Arguments:**\n        text            The text (excluding units) to display on the label for this\n                        axis.\n        units           The units for this axis. Units should generally be given\n                        without any scaling prefix (eg, \'V\' instead of \'mV\'). The\n                        scaling prefix will be automatically prepended based on the\n                        range of data displayed.\n        args            All extra keyword arguments become CSS style options for\n                        the <span> tag which will surround the axis label and units.\n        ==============  =============================================================\n\n        The final text generated for the label will look like::\n\n            <span style="...options...">{text} (prefix{units})</span>\n\n        Each extra keyword argument will become a CSS option in the above template.\n        For example, you can set the font size and color of the label::\n\n            labelStyle = {\'color\': \'#FFF\', \'font-size\': \'14pt\'}\n            axis.setLabel(\'label text\', units=\'V\', **labelStyle)\n\n        '
        self.labelText = text or ''
        self.labelUnits = units or ''
        self.labelUnitPrefix = unitPrefix or ''
        if len(args) > 0:
            self.labelStyle = args
        visible = True if text or units else False
        self.showLabel(visible)
        self._updateLabel()

    def _updateLabel(self):
        if False:
            return 10
        'Internal method to update the label according to the text'
        self.label.setHtml(self.labelString())
        self._adjustSize()
        self.picture = None
        self.update()

    def labelString(self):
        if False:
            return 10
        if self.labelUnits == '':
            if not self.autoSIPrefix or self.autoSIPrefixScale == 1.0:
                units = ''
            else:
                units = '(x%g)' % (1.0 / self.autoSIPrefixScale)
        else:
            units = '(%s%s)' % (self.labelUnitPrefix, self.labelUnits)
        s = '%s %s' % (self.labelText, units)
        style = ';'.join(['%s: %s' % (k, self.labelStyle[k]) for k in self.labelStyle])
        return "<span style='%s'>%s</span>" % (style, s)

    def _updateMaxTextSize(self, x):
        if False:
            i = 10
            return i + 15
        if self.orientation in ['left', 'right']:
            if self.style['autoReduceTextSpace']:
                if x > self.textWidth or x < self.textWidth - 10:
                    self.textWidth = x
            else:
                mx = max(self.textWidth, x)
                if mx > self.textWidth or mx < self.textWidth - 10:
                    self.textWidth = mx
            if self.style['autoExpandTextSpace']:
                self._updateWidth()
        else:
            if self.style['autoReduceTextSpace']:
                if x > self.textHeight or x < self.textHeight - 10:
                    self.textHeight = x
            else:
                mx = max(self.textHeight, x)
                if mx > self.textHeight or mx < self.textHeight - 10:
                    self.textHeight = mx
            if self.style['autoExpandTextSpace']:
                self._updateHeight()

    def _adjustSize(self):
        if False:
            while True:
                i = 10
        if self.orientation in ['left', 'right']:
            self._updateWidth()
        else:
            self._updateHeight()

    def setHeight(self, h=None):
        if False:
            print('Hello World!')
        'Set the height of this axis reserved for ticks and tick labels.\n        The height of the axis label is automatically added.\n\n        If *height* is None, then the value will be determined automatically\n        based on the size of the tick text.'
        self.fixedHeight = h
        self._updateHeight()

    def _updateHeight(self):
        if False:
            return 10
        if not self.isVisible():
            h = 0
        elif self.fixedHeight is None:
            if not self.style['showValues']:
                h = 0
            elif self.style['autoExpandTextSpace']:
                h = self.textHeight
            else:
                h = self.style['tickTextHeight']
            h += self.style['tickTextOffset'][1] if self.style['showValues'] else 0
            h += max(0, self.style['tickLength'])
            if self.label.isVisible():
                h += self.label.boundingRect().height() * 0.8
        else:
            h = self.fixedHeight
        self.setMaximumHeight(h)
        self.setMinimumHeight(h)
        self.picture = None

    def setWidth(self, w=None):
        if False:
            i = 10
            return i + 15
        'Set the width of this axis reserved for ticks and tick labels.\n        The width of the axis label is automatically added.\n\n        If *width* is None, then the value will be determined automatically\n        based on the size of the tick text.'
        self.fixedWidth = w
        self._updateWidth()

    def _updateWidth(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.isVisible():
            w = 0
        elif self.fixedWidth is None:
            if not self.style['showValues']:
                w = 0
            elif self.style['autoExpandTextSpace']:
                w = self.textWidth
            else:
                w = self.style['tickTextWidth']
            w += self.style['tickTextOffset'][0] if self.style['showValues'] else 0
            w += max(0, self.style['tickLength'])
            if self.label.isVisible():
                w += self.label.boundingRect().height() * 0.8
        else:
            w = self.fixedWidth
        self.setMaximumWidth(w)
        self.setMinimumWidth(w)
        self.picture = None

    def pen(self):
        if False:
            return 10
        if self._pen is None:
            return fn.mkPen(getConfigOption('foreground'))
        return fn.mkPen(self._pen)

    def setPen(self, *args, **kwargs):
        if False:
            return 10
        '\n        Set the pen used for drawing text, axes, ticks, and grid lines.\n        If no arguments are given, the default foreground color will be used\n        (see :func:`setConfigOption <pyqtgraph.setConfigOption>`).\n        '
        self.picture = None
        if args or kwargs:
            self._pen = fn.mkPen(*args, **kwargs)
        else:
            self._pen = fn.mkPen(getConfigOption('foreground'))
        self.labelStyle['color'] = self._pen.color().name()
        self._updateLabel()

    def textPen(self):
        if False:
            return 10
        if self._textPen is None:
            return fn.mkPen(getConfigOption('foreground'))
        return fn.mkPen(self._textPen)

    def setTextPen(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Set the pen used for drawing text.\n        If no arguments are given, the default foreground color will be used.\n        '
        self.picture = None
        if args or kwargs:
            self._textPen = fn.mkPen(*args, **kwargs)
        else:
            self._textPen = fn.mkPen(getConfigOption('foreground'))
        self.labelStyle['color'] = self._textPen.color().name()
        self._updateLabel()

    def tickPen(self):
        if False:
            i = 10
            return i + 15
        if self._tickPen is None:
            return self.pen()
        else:
            return fn.mkPen(self._tickPen)

    def setTickPen(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Set the pen used for drawing tick marks.\n        If no arguments are given, the default pen will be used.\n        '
        self.picture = None
        if args or kwargs:
            self._tickPen = fn.mkPen(*args, **kwargs)
        else:
            self._tickPen = None
        self._updateLabel()

    def setScale(self, scale=None):
        if False:
            return 10
        '\n        Set the value scaling for this axis.\n\n        Setting this value causes the axis to draw ticks and tick labels as if\n        the view coordinate system were scaled. By default, the axis scaling is\n        1.0.\n        '
        if scale != self.scale:
            self.scale = scale
            self._updateLabel()

    def enableAutoSIPrefix(self, enable=True):
        if False:
            print('Hello World!')
        "\n        Enable (or disable) automatic SI prefix scaling on this axis.\n\n        When enabled, this feature automatically determines the best SI prefix\n        to prepend to the label units, while ensuring that axis values are scaled\n        accordingly.\n\n        For example, if the axis spans values from -0.1 to 0.1 and has units set\n        to 'V' then the axis would display values -100 to 100\n        and the units would appear as 'mV'\n\n        This feature is enabled by default, and is only available when a suffix\n        (unit string) is provided to display on the label.\n        "
        self.autoSIPrefix = enable
        self.updateAutoSIPrefix()

    def updateAutoSIPrefix(self):
        if False:
            i = 10
            return i + 15
        if self.label.isVisible():
            if self.logMode:
                _range = 10 ** np.array(self.range)
            else:
                _range = self.range
            (scale, prefix) = fn.siScale(max(abs(_range[0] * self.scale), abs(_range[1] * self.scale)))
            if self.labelUnits == '' and prefix in ['k', 'm']:
                scale = 1.0
                prefix = ''
            self.autoSIPrefixScale = scale
            self.labelUnitPrefix = prefix
        else:
            self.autoSIPrefixScale = 1.0
        self._updateLabel()

    def setRange(self, mn, mx):
        if False:
            while True:
                i = 10
        'Set the range of values displayed by the axis.\n        Usually this is handled automatically by linking the axis to a ViewBox with :func:`linkToView <pyqtgraph.AxisItem.linkToView>`'
        if not isfinite(mn) or not isfinite(mx):
            raise Exception('Not setting range to [%s, %s]' % (str(mn), str(mx)))
        self.range = [mn, mx]
        if self.autoSIPrefix:
            self.updateAutoSIPrefix()
        else:
            self.picture = None
            self.update()

    def linkedView(self):
        if False:
            print('Hello World!')
        'Return the ViewBox this axis is linked to.'
        if self._linkedView is None:
            return None
        else:
            return self._linkedView()

    def _linkToView_internal(self, view):
        if False:
            i = 10
            return i + 15
        self.unlinkFromView()
        self._linkedView = weakref.ref(view)
        if self.orientation in ['right', 'left']:
            view.sigYRangeChanged.connect(self.linkedViewChanged)
        else:
            view.sigXRangeChanged.connect(self.linkedViewChanged)
        view.sigResized.connect(self.linkedViewChanged)

    def linkToView(self, view):
        if False:
            return 10
        'Link this axis to a ViewBox, causing its displayed range to match the visible range of the view.'
        self._linkToView_internal(view)

    def unlinkFromView(self):
        if False:
            i = 10
            return i + 15
        'Unlink this axis from a ViewBox.'
        oldView = self.linkedView()
        self._linkedView = None
        if self.orientation in ['right', 'left']:
            if oldView is not None:
                oldView.sigYRangeChanged.disconnect(self.linkedViewChanged)
        elif oldView is not None:
            oldView.sigXRangeChanged.disconnect(self.linkedViewChanged)
        if oldView is not None:
            oldView.sigResized.disconnect(self.linkedViewChanged)

    def linkedViewChanged(self, view, newRange=None):
        if False:
            i = 10
            return i + 15
        if self.orientation in ['right', 'left']:
            if newRange is None:
                newRange = view.viewRange()[1]
            if view.yInverted():
                self.setRange(*newRange[::-1])
            else:
                self.setRange(*newRange)
        else:
            if newRange is None:
                newRange = view.viewRange()[0]
            if view.xInverted():
                self.setRange(*newRange[::-1])
            else:
                self.setRange(*newRange)

    def boundingRect(self):
        if False:
            while True:
                i = 10
        m = 0
        hide_overlapping_labels = self.style['hideOverlappingLabels']
        if hide_overlapping_labels is True:
            pass
        elif hide_overlapping_labels is False:
            m = 15
        else:
            try:
                m = int(self.style['hideOverlappingLabels'])
            except ValueError:
                pass
        linkedView = self.linkedView()
        if linkedView is None or self.grid is False:
            rect = self.mapRectFromParent(self.geometry())
            tl = self.style['tickLength']
            if self.orientation == 'left':
                rect = rect.adjusted(0, -m, -min(0, tl), m)
            elif self.orientation == 'right':
                rect = rect.adjusted(min(0, tl), -m, 0, m)
            elif self.orientation == 'top':
                rect = rect.adjusted(-m, 0, m, -min(0, tl))
            elif self.orientation == 'bottom':
                rect = rect.adjusted(-m, min(0, tl), m, 0)
            return rect
        else:
            return self.mapRectFromParent(self.geometry()) | linkedView.mapRectToItem(self, linkedView.boundingRect())

    def paint(self, p, opt, widget):
        if False:
            return 10
        profiler = debug.Profiler()
        if self.picture is None:
            try:
                picture = QtGui.QPicture()
                painter = QtGui.QPainter(picture)
                if self.style['tickFont']:
                    painter.setFont(self.style['tickFont'])
                specs = self.generateDrawSpecs(painter)
                profiler('generate specs')
                if specs is not None:
                    self.drawPicture(painter, *specs)
                    profiler('draw picture')
            finally:
                painter.end()
            self.picture = picture
        self.picture.play(p)

    def setTickDensity(self, density=1.0):
        if False:
            i = 10
            return i + 15
        '\n        The default behavior is to show at least two major ticks for axes of up to 300 pixels in length, \n        then add additional major ticks, spacing them out further as the available room increases.\n        (Internally, the targeted number of major ticks grows with the square root of the axes length.)\n\n        Setting a tick density different from the default value of `density = 1.0` scales the number of\n        major ticks that is targeted for display. This only affects the automatic generation of ticks.\n        '
        self._tickDensity = density
        self.picture = None
        self.update()

    def setTicks(self, ticks):
        if False:
            while True:
                i = 10
        'Explicitly determine which ticks to display.\n        This overrides the behavior specified by tickSpacing(), tickValues(), and tickStrings()\n        The format for *ticks* looks like::\n\n            [\n                [ (majorTickValue1, majorTickString1), (majorTickValue2, majorTickString2), ... ],\n                [ (minorTickValue1, minorTickString1), (minorTickValue2, minorTickString2), ... ],\n                ...\n            ]\n\n        The two levels of major and minor ticks are expected. A third tier of additional ticks is optional.\n        If *ticks* is None, then the default tick system will be used instead.\n        '
        self._tickLevels = ticks
        self.picture = None
        self.update()

    def setTickSpacing(self, major=None, minor=None, levels=None):
        if False:
            i = 10
            return i + 15
        '\n        Explicitly determine the spacing of major and minor ticks. This\n        overrides the default behavior of the tickSpacing method, and disables\n        the effect of setTicks(). Arguments may be either *major* and *minor*,\n        or *levels* which is a list of (spacing, offset) tuples for each\n        tick level desired.\n\n        If no arguments are given, then the default behavior of tickSpacing\n        is enabled.\n\n        Examples::\n\n            # two levels, all offsets = 0\n            axis.setTickSpacing(5, 1)\n            # three levels, all offsets = 0\n            axis.setTickSpacing(levels=[(3, 0), (1, 0), (0.25, 0)])\n            # reset to default\n            axis.setTickSpacing()\n        '
        if levels is None:
            if major is None:
                levels = None
            else:
                levels = [(major, 0), (minor, 0)]
        self._tickSpacing = levels
        self.picture = None
        self.update()

    def tickSpacing(self, minVal, maxVal, size):
        if False:
            i = 10
            return i + 15
        'Return values describing the desired spacing and offset of ticks.\n\n        This method is called whenever the axis needs to be redrawn and is a\n        good method to override in subclasses that require control over tick locations.\n\n        The return value must be a list of tuples, one for each set of ticks::\n\n            [\n                (major tick spacing, offset),\n                (minor tick spacing, offset),\n                (sub-minor tick spacing, offset),\n                ...\n            ]\n        '
        if self._tickSpacing is not None:
            return self._tickSpacing
        dif = abs(maxVal - minVal)
        if dif == 0:
            return []
        ref_size = 300.0
        minNumberOfIntervals = max(2.25, 2.25 * self._tickDensity * sqrt(size / ref_size))
        majorMaxSpacing = dif / minNumberOfIntervals
        (mantissa, exp2) = frexp(majorMaxSpacing)
        p10unit = 10.0 ** (floor((exp2 - 1) / 3.32192809488736) - 1)
        if 100.0 * p10unit <= majorMaxSpacing:
            majorScaleFactor = 10
            p10unit *= 10.0
        else:
            for majorScaleFactor in (50, 20, 10):
                if majorScaleFactor * p10unit <= majorMaxSpacing:
                    break
        majorInterval = majorScaleFactor * p10unit
        minorMinSpacing = 2 * dif / size
        if majorScaleFactor == 10:
            trials = (5, 10)
        else:
            trials = (10, 20, 50)
        for minorScaleFactor in trials:
            minorInterval = minorScaleFactor * p10unit
            if minorInterval >= minorMinSpacing:
                break
        levels = [(majorInterval, 0), (minorInterval, 0)]
        if self.style['maxTickLevel'] >= 2:
            if majorScaleFactor == 10:
                trials = (1, 2, 5, 10)
            elif majorScaleFactor == 20:
                trials = (2, 5, 10, 20)
            elif majorScaleFactor == 50:
                trials = (5, 10, 50)
            else:
                trials = ()
                extraInterval = minorInterval
            for extraScaleFactor in trials:
                extraInterval = extraScaleFactor * p10unit
                if extraInterval >= minorMinSpacing or extraInterval == minorInterval:
                    break
            if extraInterval < minorInterval:
                levels.append((extraInterval, 0))
        return levels

    def tickValues(self, minVal, maxVal, size):
        if False:
            return 10
        '\n        Return the values and spacing of ticks to draw::\n\n            [\n                (spacing, [major ticks]),\n                (spacing, [minor ticks]),\n                ...\n            ]\n\n        By default, this method calls tickSpacing to determine the correct tick locations.\n        This is a good method to override in subclasses.\n        '
        (minVal, maxVal) = sorted((minVal, maxVal))
        minVal *= self.scale
        maxVal *= self.scale
        ticks = []
        tickLevels = self.tickSpacing(minVal, maxVal, size)
        allValues = np.array([])
        for i in range(len(tickLevels)):
            (spacing, offset) = tickLevels[i]
            start = ceil((minVal - offset) / spacing) * spacing + offset
            num = int((maxVal - start) / spacing) + 1
            values = (np.arange(num) * spacing + start) / self.scale
            close = np.any(np.isclose(allValues, values[:, np.newaxis], rtol=0, atol=spacing / self.scale * 0.01), axis=-1)
            values = values[~close]
            allValues = np.concatenate([allValues, values])
            ticks.append((spacing / self.scale, values.tolist()))
        if self.logMode:
            return self.logTickValues(minVal, maxVal, size, ticks)
        return ticks

    def logTickValues(self, minVal, maxVal, size, stdTicks):
        if False:
            print('Hello World!')
        ticks = []
        for (spacing, t) in stdTicks:
            if spacing >= 1.0:
                ticks.append((spacing, t))
        if len(ticks) < 3:
            v1 = int(floor(minVal))
            v2 = int(ceil(maxVal))
            minor = []
            for v in range(v1, v2):
                minor.extend(v + np.log10(np.arange(1, 10)))
            minor = [x for x in minor if x > minVal and x < maxVal]
            ticks.append((None, minor))
        return ticks

    def tickStrings(self, values, scale, spacing):
        if False:
            while True:
                i = 10
        "Return the strings that should be placed next to ticks. This method is called\n        when redrawing the axis and is a good method to override in subclasses.\n        The method is called with a list of tick values, a scaling factor (see below), and the\n        spacing between ticks (this is required since, in some instances, there may be only\n        one tick and thus no other way to determine the tick spacing)\n\n        The scale argument is used when the axis label is displaying units which may have an SI scaling prefix.\n        When determining the text to display, use value*scale to correctly account for this prefix.\n        For example, if the axis label's units are set to 'V', then a tick value of 0.001 might\n        be accompanied by a scale value of 1000. This indicates that the label is displaying 'mV', and\n        thus the tick should display 0.001 * 1000 = 1.\n        "
        if self.logMode:
            return self.logTickStrings(values, scale, spacing)
        places = max(0, ceil(-log10(spacing * scale)))
        strings = []
        for v in values:
            vs = v * scale
            if abs(vs) < 0.001 or abs(vs) >= 10000:
                vstr = '%g' % vs
            else:
                vstr = '%%0.%df' % places % vs
            strings.append(vstr)
        return strings

    def logTickStrings(self, values, scale, spacing):
        if False:
            return 10
        estrings = ['%0.1g' % x for x in 10 ** np.array(values).astype(float) * np.array(scale)]
        convdict = {'0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹'}
        dstrings = []
        for e in estrings:
            if e.count('e'):
                (v, p) = e.split('e')
                sign = '⁻' if p[0] == '-' else ''
                pot = ''.join([convdict[pp] for pp in p[1:].lstrip('0')])
                if v == '1':
                    v = ''
                else:
                    v = v + '·'
                dstrings.append(v + '10' + sign + pot)
            else:
                dstrings.append(e)
        return dstrings

    def generateDrawSpecs(self, p):
        if False:
            for i in range(10):
                print('nop')
        '\n        Calls tickValues() and tickStrings() to determine where and how ticks should\n        be drawn, then generates from this a set of drawing commands to be\n        interpreted by drawPicture().\n        '
        profiler = debug.Profiler()
        if self.style['tickFont'] is not None:
            p.setFont(self.style['tickFont'])
        bounds = self.mapRectFromParent(self.geometry())
        linkedView = self.linkedView()
        if linkedView is None or self.grid is False:
            tickBounds = bounds
        else:
            tickBounds = linkedView.mapRectToItem(self, linkedView.boundingRect())
        left_offset = -1.0
        right_offset = 1.0
        top_offset = -1.0
        bottom_offset = 1.0
        if self.orientation == 'left':
            span = (bounds.topRight() + Point(left_offset, top_offset), bounds.bottomRight() + Point(left_offset, bottom_offset))
            tickStart = tickBounds.right()
            tickStop = bounds.right()
            tickDir = -1
            axis = 0
        elif self.orientation == 'right':
            span = (bounds.topLeft() + Point(right_offset, top_offset), bounds.bottomLeft() + Point(right_offset, bottom_offset))
            tickStart = tickBounds.left()
            tickStop = bounds.left()
            tickDir = 1
            axis = 0
        elif self.orientation == 'top':
            span = (bounds.bottomLeft() + Point(left_offset, top_offset), bounds.bottomRight() + Point(right_offset, top_offset))
            tickStart = tickBounds.bottom()
            tickStop = bounds.bottom()
            tickDir = -1
            axis = 1
        elif self.orientation == 'bottom':
            span = (bounds.topLeft() + Point(left_offset, bottom_offset), bounds.topRight() + Point(right_offset, bottom_offset))
            tickStart = tickBounds.top()
            tickStop = bounds.top()
            tickDir = 1
            axis = 1
        else:
            raise ValueError("self.orientation must be in ('left', 'right', 'top', 'bottom')")
        points = list(map(self.mapToDevice, span))
        if None in points:
            return
        lengthInPixels = Point(points[1] - points[0]).length()
        if lengthInPixels == 0:
            return
        if self._tickLevels is None:
            tickLevels = self.tickValues(self.range[0], self.range[1], lengthInPixels)
            tickStrings = None
        else:
            tickLevels = []
            tickStrings = []
            for level in self._tickLevels:
                values = []
                strings = []
                tickLevels.append((None, values))
                tickStrings.append(strings)
                for (val, strn) in level:
                    values.append(val)
                    strings.append(strn)
        dif = self.range[1] - self.range[0]
        if dif == 0:
            xScale = 1
            offset = 0
        elif axis == 0:
            xScale = -bounds.height() / dif
            offset = self.range[0] * xScale - bounds.height()
        else:
            xScale = bounds.width() / dif
            offset = self.range[0] * xScale
        xRange = [x * xScale - offset for x in self.range]
        xMin = min(xRange)
        xMax = max(xRange)
        profiler('init')
        tickPositions = []
        tickSpecs = []
        for i in range(len(tickLevels)):
            tickPositions.append([])
            ticks = tickLevels[i][1]
            tickLength = self.style['tickLength'] / (i * 0.5 + 1.0)
            lineAlpha = self.style['tickAlpha']
            if lineAlpha is None:
                lineAlpha = 255 / (i + 1)
                if self.grid is not False:
                    lineAlpha *= self.grid / 255.0 * fn.clip_scalar(0.05 * lengthInPixels / (len(ticks) + 1), 0.0, 1.0)
            elif isinstance(lineAlpha, float):
                lineAlpha *= 255
                lineAlpha = max(0, int(round(lineAlpha)))
                lineAlpha = min(255, int(round(lineAlpha)))
            elif isinstance(lineAlpha, int):
                if lineAlpha > 255 or lineAlpha < 0:
                    raise ValueError('lineAlpha should be [0..255]')
            else:
                raise TypeError('Line Alpha should be of type None, float or int')
            tickPen = self.tickPen()
            if tickPen.brush().style() == QtCore.Qt.BrushStyle.SolidPattern:
                tickPen = QtGui.QPen(tickPen)
                color = QtGui.QColor(tickPen.color())
                color.setAlpha(int(lineAlpha))
                tickPen.setColor(color)
            for v in ticks:
                x = v * xScale - offset
                if x < xMin or x > xMax:
                    tickPositions[i].append(None)
                    continue
                tickPositions[i].append(x)
                p1 = [x, x]
                p2 = [x, x]
                p1[axis] = tickStart
                p2[axis] = tickStop
                if self.grid is False:
                    p2[axis] += tickLength * tickDir
                tickSpecs.append((tickPen, Point(p1), Point(p2)))
        profiler('compute ticks')
        if self.style['stopAxisAtTick'][0] is True:
            minTickPosition = min(map(min, tickPositions))
            if axis == 0:
                stop = max(span[0].y(), minTickPosition)
                span[0].setY(stop)
            else:
                stop = max(span[0].x(), minTickPosition)
                span[0].setX(stop)
        if self.style['stopAxisAtTick'][1] is True:
            maxTickPosition = max(map(max, tickPositions))
            if axis == 0:
                stop = min(span[1].y(), maxTickPosition)
                span[1].setY(stop)
            else:
                stop = min(span[1].x(), maxTickPosition)
                span[1].setX(stop)
        axisSpec = (self.pen(), span[0], span[1])
        textOffset = self.style['tickTextOffset'][axis]
        textSize2 = 0
        lastTextSize2 = 0
        textRects = []
        textSpecs = []
        if not self.style['showValues']:
            return (axisSpec, tickSpecs, textSpecs)
        for i in range(min(len(tickLevels), self.style['maxTextLevel'] + 1)):
            if tickStrings is None:
                (spacing, values) = tickLevels[i]
                strings = self.tickStrings(values, self.autoSIPrefixScale * self.scale, spacing)
            else:
                strings = tickStrings[i]
            if len(strings) == 0:
                continue
            for j in range(len(strings)):
                if tickPositions[i][j] is None:
                    strings[j] = None
            rects = []
            for s in strings:
                if s is None:
                    rects.append(None)
                else:
                    br = p.boundingRect(QtCore.QRectF(0, 0, 100, 100), QtCore.Qt.AlignmentFlag.AlignCenter, s)
                    br.setHeight(br.height() * 0.8)
                    rects.append(br)
                    textRects.append(rects[-1])
            if len(textRects) > 0:
                if axis == 0:
                    textSize = np.sum([r.height() for r in textRects])
                    textSize2 = np.max([r.width() for r in textRects])
                else:
                    textSize = np.sum([r.width() for r in textRects])
                    textSize2 = np.max([r.height() for r in textRects])
            else:
                textSize = 0
                textSize2 = 0
            if i > 0:
                textFillRatio = float(textSize) / lengthInPixels
                finished = False
                for (nTexts, limit) in self.style['textFillLimits']:
                    if len(textSpecs) >= nTexts and textFillRatio >= limit:
                        finished = True
                        break
                if finished:
                    break
            lastTextSize2 = textSize2
            for j in range(len(strings)):
                vstr = strings[j]
                if vstr is None:
                    continue
                x = tickPositions[i][j]
                textRect = rects[j]
                height = textRect.height()
                width = textRect.width()
                offset = max(0, self.style['tickLength']) + textOffset
                rect = QtCore.QRectF()
                if self.orientation == 'left':
                    alignFlags = QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
                    rect = QtCore.QRectF(tickStop - offset - width, x - height / 2, width, height)
                elif self.orientation == 'right':
                    alignFlags = QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter
                    rect = QtCore.QRectF(tickStop + offset, x - height / 2, width, height)
                elif self.orientation == 'top':
                    alignFlags = QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignBottom
                    rect = QtCore.QRectF(x - width / 2.0, tickStop - offset - height, width, height)
                elif self.orientation == 'bottom':
                    alignFlags = QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignTop
                    rect = QtCore.QRectF(x - width / 2.0, tickStop + offset, width, height)
                textFlags = alignFlags | QtCore.Qt.TextFlag.TextDontClip
                br = self.boundingRect()
                if not br.contains(rect):
                    continue
                textSpecs.append((rect, textFlags, vstr))
        profiler('compute text')
        self._updateMaxTextSize(lastTextSize2)
        return (axisSpec, tickSpecs, textSpecs)

    def drawPicture(self, p, axisSpec, tickSpecs, textSpecs):
        if False:
            for i in range(10):
                print('nop')
        profiler = debug.Profiler()
        p.setRenderHint(p.RenderHint.Antialiasing, False)
        p.setRenderHint(p.RenderHint.TextAntialiasing, True)
        (pen, p1, p2) = axisSpec
        p.setPen(pen)
        p.drawLine(p1, p2)
        for (pen, p1, p2) in tickSpecs:
            p.setPen(pen)
            p.drawLine(p1, p2)
        profiler('draw ticks')
        if self.style['tickFont'] is not None:
            p.setFont(self.style['tickFont'])
        p.setPen(self.textPen())
        bounding = self.boundingRect().toAlignedRect()
        p.setClipRect(bounding)
        for (rect, flags, text) in textSpecs:
            p.drawText(rect, int(flags), text)
        profiler('draw text')

    def show(self):
        if False:
            while True:
                i = 10
        GraphicsWidget.show(self)
        if self.orientation in ['left', 'right']:
            self._updateWidth()
        else:
            self._updateHeight()

    def hide(self):
        if False:
            i = 10
            return i + 15
        GraphicsWidget.hide(self)
        if self.orientation in ['left', 'right']:
            self._updateWidth()
        else:
            self._updateHeight()

    def wheelEvent(self, event):
        if False:
            return 10
        lv = self.linkedView()
        if lv is None:
            return
        if lv.sceneBoundingRect().contains(event.scenePos()):
            event.ignore()
            return
        elif self.orientation in ['left', 'right']:
            lv.wheelEvent(event, axis=1)
        else:
            lv.wheelEvent(event, axis=0)
        event.accept()

    def mouseDragEvent(self, event):
        if False:
            print('Hello World!')
        lv = self.linkedView()
        if lv is None:
            return
        if lv.sceneBoundingRect().contains(event.buttonDownScenePos()):
            event.ignore()
            return
        if self.orientation in ['left', 'right']:
            return lv.mouseDragEvent(event, axis=1)
        else:
            return lv.mouseDragEvent(event, axis=0)

    def mouseClickEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        lv = self.linkedView()
        if lv is None:
            return
        return lv.mouseClickEvent(event)