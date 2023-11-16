"""
Created on 2017年12月18日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: ChartView
@description: 
"""
import json
import os
import chardet
from PyQt5.QtChart import QChart, QChartView, QLineSeries, QCategoryAxis
from PyQt5.QtCore import QMargins, Qt, QEasingCurve
from PyQt5.QtGui import QColor, QBrush, QFont, QPainter, QPen, QPixmap
EasingCurve = dict([(c, getattr(QEasingCurve, n)) for (n, c) in QEasingCurve.__dict__.items() if isinstance(c, QEasingCurve.Type)])
AnimationOptions = {0: QChart.NoAnimation, 1: QChart.GridAxisAnimations, 2: QChart.SeriesAnimations, 3: QChart.AllAnimations}

class ChartView(QChartView):

    def __init__(self, file, parent=None):
        if False:
            while True:
                i = 10
        super(ChartView, self).__init__(parent)
        self._chart = QChart()
        self._chart.setAcceptHoverEvents(True)
        self.setChart(self._chart)
        self.initUi(file)

    def initUi(self, file):
        if False:
            print('Hello World!')
        if isinstance(file, dict):
            return self.__analysis(file)
        if isinstance(file, str):
            if not os.path.isfile(file):
                return self.__analysis(json.loads(file))
            with open(file, 'rb') as fp:
                data = fp.read()
                encoding = chardet.detect(data) or {}
                data = data.decode(encoding.get('encoding') or 'utf-8')
            self.__analysis(json.loads(data))

    def mouseMoveEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        super(ChartView, self).mouseMoveEvent(event)
        (axisX, axisY) = (self._chart.axisX(), self._chart.axisY())
        (min_x, max_x) = (axisX.min(), axisX.max())
        (min_y, max_y) = (axisY.min(), axisY.max())
        x = self._chart.mapToValue(event.pos()).x()
        y = self._chart.mapToValue(event.pos()).y()
        index = round(x)
        print(x, y, index)
        points = [(s.type(), s.at(index)) for s in self._chart.series() if min_x <= x <= max_x and min_y <= y <= max_y]
        print(points)

    def __getColor(self, color=None, default=Qt.white):
        if False:
            return 10
        '\n        :param color: int|str|[r,g,b]|[r,g,b,a]\n        '
        if not color:
            return QColor(default)
        if isinstance(color, QBrush):
            return color
        if isinstance(color, list) and 3 <= len(color) <= 4:
            return QColor(*color)
        else:
            return QColor(color)

    def __getPen(self, pen=None, default=QPen(Qt.white, 1, Qt.SolidLine, Qt.SquareCap, Qt.BevelJoin)):
        if False:
            print('Hello World!')
        '\n        :param pen: pen json\n        '
        if not pen or not isinstance(pen, dict):
            return default
        return QPen(self.__getColor(pen.get('color', None) or default.color()), pen.get('width', 1) or 1, pen.get('style', 0) or 0, pen.get('capStyle', 16) or 16, pen.get('joinStyle', 64) or 64)

    def __getAlignment(self, alignment):
        if False:
            while True:
                i = 10
        '\n        :param alignment: left|top|right|bottom\n        '
        try:
            return getattr(Qt, 'Align' + alignment.capitalize())
        except:
            return Qt.AlignTop

    def __setTitle(self, title=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param title: title json\n        '
        if not title or not isinstance(title, dict):
            return
        self._chart.setTitle(title.get('text', '') or '')
        self._chart.setTitleBrush(self.__getColor(title.get('color', self._chart.titleBrush()) or self._chart.titleBrush()))
        font = QFont(title.get('font', '') or self._chart.titleFont())
        pointSize = title.get('pointSize', -1) or -1
        if pointSize > 0:
            font.setPointSize(pointSize)
        font.setWeight(title.get('weight', -1) or -1)
        font.setItalic(title.get('italic', False) or False)
        self._chart.setTitleFont(font)

    def __setAnimation(self, animation=None):
        if False:
            i = 10
            return i + 15
        '\n        :param value: animation json\n        '
        if not animation or not isinstance(animation, dict):
            return
        self._chart.setAnimationDuration(animation.get('duration', 1000) or 1000)
        self._chart.setAnimationEasingCurve(EasingCurve.get(animation.get('curve', 10) or 10, None) or QEasingCurve.OutQuart)
        self._chart.setAnimationOptions(AnimationOptions.get(animation.get('options', 0) or 0, None) or QChart.NoAnimation)

    def __setBackground(self, background=None):
        if False:
            i = 10
            return i + 15
        '\n        :param background:background json\n        '
        if not background or not isinstance(background, dict):
            return
        self._chart.setBackgroundVisible(background.get('visible', True) or True)
        self._chart.setBackgroundRoundness(background.get('radius', 0) or 0)
        self._chart.setDropShadowEnabled(background.get('dropShadow', True) or True)
        self._chart.setBackgroundPen(self.__getPen(background.get('pen', None), self._chart.backgroundPen()))
        image = background.get('image', None)
        color = background.get('color', None)
        if image:
            self._chart.setBackgroundBrush(QBrush(QPixmap(image)))
        elif color:
            self._chart.setBackgroundBrush(self.__getColor(color, self._chart.backgroundBrush()))

    def __setMargins(self, margins=None):
        if False:
            i = 10
            return i + 15
        '\n        :param margins: margins json\n        '
        if not margins or not isinstance(margins, dict):
            return
        left = margins.get('left', 20) or 20
        top = margins.get('top', 20) or 20
        right = margins.get('right', 20) or 20
        bottom = margins.get('bottom', 20) or 20
        self._chart.setMargins(QMargins(left, top, right, bottom))

    def __setLegend(self, legend=None):
        if False:
            while True:
                i = 10
        '\n        :param legend: legend json\n        '
        if not legend or not isinstance(legend, dict):
            return
        _legend = self._chart.legend()
        _legend.setAlignment(self.__getAlignment(legend.get('alignment', None)))
        _legend.setShowToolTips(legend.get('showToolTips', True) or True)

    def __getSerie(self, serie=None):
        if False:
            i = 10
            return i + 15
        if not serie or not isinstance(serie, dict):
            return None
        types = serie.get('type', '') or ''
        data = serie.get('data', []) or []
        if not data or not isinstance(data, list):
            return None
        if types == 'line':
            _series = QLineSeries(self._chart)
        else:
            return None
        _series.setName(serie.get('name', '') or '')
        for (index, value) in enumerate(data):
            _series.append(index, value if type(value) in (int, float) else 0)
        return _series

    def __setSeries(self, series=None):
        if False:
            i = 10
            return i + 15
        if not series or not isinstance(series, list):
            return
        for serie in series:
            _serie = self.__getSerie(serie)
            if _serie:
                self._chart.addSeries(_serie)
        self._chart.createDefaultAxes()

    def __setAxisX(self, axisx=None):
        if False:
            for i in range(10):
                print('nop')
        if not axisx or not isinstance(axisx, dict):
            return
        series = self._chart.series()
        if not series:
            return
        types = axisx.get('type', None)
        data = axisx.get('data', []) or []
        if not data or not isinstance(data, list):
            return None
        minx = self._chart.axisX().min()
        maxx = self._chart.axisX().max()
        if types == 'category':
            xaxis = QCategoryAxis(self._chart, labelsPosition=QCategoryAxis.AxisLabelsPositionOnValue)
            xaxis.setGridLineVisible(False)
            tickc_d = len(data)
            tickc = tickc_d if tickc_d > 1 else self._chart.axisX().tickCount()
            xaxis.setTickCount(tickc)
            self._chart.axisX().setTickCount(tickc)
            step = (maxx - minx) / (tickc - 1)
            for i in range(min(tickc_d, tickc)):
                xaxis.append(data[i], minx + i * step)
            self._chart.setAxisX(xaxis, series[-1])

    def __analysis(self, datas):
        if False:
            print('Hello World!')
        '\n        analysis json data\n        :param datas: json data\n        '
        self.__setTitle(datas.get('title', None))
        if datas.get('antialiasing', False) or False:
            self.setRenderHint(QPainter.Antialiasing)
        self._chart.setTheme(datas.get('theme', 0) or 0)
        self.__setAnimation(datas.get('animation', None))
        self.__setBackground(datas.get('background', None))
        self.__setMargins(datas.get('margins', None))
        self.__setLegend(datas.get('legend', None))
        self.__setSeries(datas.get('series', None))
        self.__setAxisX(datas.get('axisx', None))