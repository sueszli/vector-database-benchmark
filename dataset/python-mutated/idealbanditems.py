from PyQt5 import QtGui, QtCore, Qt
import pyqtgraph as pg
import numpy

class IdealBandItems(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        pg.setConfigOption('foreground', 'k')
        pg.setConfigOption('background', None)
        pg.setConfigOptions(antialias=True)
        self.win = pg.GraphicsWindow()
        self.plot = self.win.addPlot()
        self.idealbandhcurves = [self.plot.plot() for i in range(4)]
        self.idealbandvcurves = [self.plot.plot() for i in range(4)]
        self.params = ''

    def setLinetype(self):
        if False:
            return 10
        for c in self.idealbandhcurves:
            c.setPen(Qt.QPen(Qt.Qt.red, 1, Qt.Qt.DotLine))
        for c in self.idealbandvcurves:
            c.setPen(Qt.QPen(Qt.Qt.red, 1, Qt.Qt.DotLine))

    def plotIdealCurves(self, ftype, params, plot):
        if False:
            while True:
                i = 10
        self.params = params
        try:
            if ftype == 'Low Pass':
                self.detach_unwantedcurves(plot)
                x = [0, self.params['pbend']]
                y = [20.0 * numpy.log10(self.params['gain'])] * 2
                self.idealbandhcurves[0].setData(x, y)
                x = [self.params['pbend']] * 2
                y = [20.0 * numpy.log10(self.params['gain']), plot.axisScaleDiv(Qwt.QwtPlot.yLeft).lowerBound()]
                self.idealbandvcurves[0].setData(x, y)
                x = [self.params['sbstart'], self.params['fs'] / 2.0]
                y = [-self.params['atten']] * 2
                self.idealbandhcurves[1].setData(x, y)
                x = [self.params['sbstart']] * 2
                y = [-self.params['atten'], plot.axisScaleDiv(Qwt.QwtPlot.yLeft).lowerBound()]
                self.idealbandvcurves[1].setData(x, y)
            elif ftype == 'High Pass':
                self.detach_unwantedcurves(plot)
                x = [self.params['pbstart'], self.params['fs'] / 2.0]
                y = [20.0 * numpy.log10(self.params['gain'])] * 2
                self.idealbandhcurves[0].setData(x, y)
                x = [self.params['pbstart']] * 2
                y = [20.0 * numpy.log10(self.params['gain']), plot.axisScaleDiv(Qwt.QwtPlot.yLeft).lowerBound()]
                self.idealbandvcurves[0].setData(x, y)
                x = [0, self.params['sbend']]
                y = [-self.params['atten']] * 2
                self.idealbandhcurves[1].setData(x, y)
                x = [self.params['sbend']] * 2
                y = [-self.params['atten'], plot.axisScaleDiv(Qwt.QwtPlot.yLeft).lowerBound()]
                self.idealbandvcurves[1].setData(x, y)
            elif ftype == 'Band Notch':
                x = [self.params['sbstart'], self.params['sbend']]
                y = [-self.params['atten']] * 2
                self.idealbandhcurves[0].setData(x, y)
                x = [self.params['sbstart']] * 2
                y = [-self.params['atten'], plot.axisScaleDiv(Qwt.QwtPlot.yLeft).lowerBound()]
                self.idealbandvcurves[0].setData(x, y)
                x = [self.params['sbend']] * 2
                y = [-self.params['atten'], plot.axisScaleDiv(Qwt.QwtPlot.yLeft).lowerBound()]
                self.idealbandvcurves[1].setData(x, y)
                x = [0, self.params['sbstart'] - self.params['tb']]
                y = [20.0 * numpy.log10(self.params['gain'])] * 2
                self.idealbandhcurves[1].setData(x, y)
                x = [self.params['sbstart'] - self.params['tb']] * 2
                y = [20.0 * numpy.log10(self.params['gain']), plot.axisScaleDiv(Qwt.QwtPlot.yLeft).lowerBound()]
                self.idealbandvcurves[2].setData(x, y)
                x = [self.params['sbend'] + self.params['tb'], self.params['fs'] / 2.0]
                y = [20.0 * numpy.log10(self.params['gain'])] * 2
                self.idealbandhcurves[2].setData(x, y)
                x = [self.params['sbend'] + self.params['tb']] * 2
                y = [20.0 * numpy.log10(self.params['gain']), plot.axisScaleDiv(Qwt.QwtPlot.yLeft).lowerBound()]
                self.idealbandvcurves[3].setData(x, y)
            elif ftype == 'Band Pass':
                x = [self.params['pbstart'], self.params['pbend']]
                y = [20.0 * numpy.log10(self.params['gain'])] * 2
                self.idealbandhcurves[0].setData(x, y)
                x = [self.params['pbstart']] * 2
                y = [20.0 * numpy.log10(self.params['gain']), plot.axisScaleDiv(Qwt.QwtPlot.yLeft).lowerBound()]
                self.idealbandvcurves[0].setData(x, y)
                x = [self.params['pbend']] * 2
                y = [20.0 * numpy.log10(self.params['gain']), plot.axisScaleDiv(Qwt.QwtPlot.yLeft).lowerBound()]
                self.idealbandvcurves[1].setData(x, y)
                x = [0, self.params['pbstart'] - self.params['tb']]
                y = [-self.params['atten']] * 2
                self.idealbandhcurves[1].setData(x, y)
                x = [self.params['pbstart'] - self.params['tb']] * 2
                y = [-self.params['atten'], plot.axisScaleDiv(Qwt.QwtPlot.yLeft).lowerBound()]
                self.idealbandvcurves[2].setData(x, y)
                x = [self.params['pbend'] + self.params['tb'], self.params['fs'] / 2.0]
                y = [-self.params['atten']] * 2
                self.idealbandhcurves[2].setData(x, y)
                x = [self.params['pbend'] + self.params['tb']] * 2
                y = [-self.params['atten'], plot.axisScaleDiv(Qwt.QwtPlot.yLeft).lowerBound()]
                self.idealbandvcurves[3].setData(x, y)
            elif ftype == 'Complex Band Pass':
                x = [self.params['pbstart'], self.params['pbend']]
                y = [20.0 * numpy.log10(self.params['gain'])] * 2
                self.idealbandhcurves[0].setData(x, y)
                x = [self.params['pbstart']] * 2
                y = [20.0 * numpy.log10(self.params['gain']), plot.axisScaleDiv(Qwt.QwtPlot.yLeft).lowerBound()]
                self.idealbandvcurves[0].setData(x, y)
                x = [self.params['pbend']] * 2
                y = [20.0 * numpy.log10(self.params['gain']), plot.axisScaleDiv(Qwt.QwtPlot.yLeft).lowerBound()]
                self.idealbandvcurves[1].setData(x, y)
                x = [0, self.params['pbstart'] - self.params['tb']]
                y = [-self.params['atten']] * 2
                self.idealbandhcurves[1].setData(x, y)
                x = [self.params['pbstart'] - self.params['tb']] * 2
                y = [-self.params['atten'], plot.axisScaleDiv(Qwt.QwtPlot.yLeft).lowerBound()]
                self.idealbandvcurves[2].setData(x, y)
                x = [self.params['pbend'] + self.params['tb'], self.params['fs'] / 2.0]
                y = [-self.params['atten']] * 2
                self.idealbandhcurves[2].setData(x, y)
                x = [self.params['pbend'] + self.params['tb']] * 2
                y = [-self.params['atten'], plot.axisScaleDiv(Qwt.QwtPlot.yLeft).lowerBound()]
                self.idealbandvcurves[3].setData(x, y)
            else:
                self.detach_allidealcurves(plot)
        except KeyError:
            print('All parameters not set for ideal band diagram')
            self.detach_allidealcurves(plot)

    def detach_allidealcurves(self, plot):
        if False:
            print('Hello World!')
        ' TODO\n        for c in self.idealbandhcurves:\n            c.detach()\n\n        for c in self.idealbandvcurves:\n            c.detach()\n        '
        plot.replot()

    def detach_unwantedcurves(self, plot):
        if False:
            return 10
        for i in range(2, 4):
            self.idealbandvcurves[i].detach()
            self.idealbandhcurves[i].detach()
        plot.replot()

    def attach_allidealcurves(self, plot):
        if False:
            for i in range(10):
                print('nop')
        ' TODO\n        for c in self.idealbandhcurves:\n            c.attach(plot)\n        for c in self.idealbandvcurves:\n            c.attach(plot)\n        '
        plot.replot()