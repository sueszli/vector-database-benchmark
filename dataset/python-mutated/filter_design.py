import sys
import os
import re
import csv
import copy
import warnings
from optparse import OptionParser
from gnuradio import filter, fft
from .CustomViewBox import CustomViewBox
try:
    import numpy as np
except ImportError:
    raise SystemExit('Please install NumPy to run this script (https://www.np.org/)')
try:
    import numpy.fft as fft_detail
except ImportError:
    raise SystemExit('Could not import fft implementation of numpy')
try:
    from numpy import poly1d
except ImportError:
    raise SystemExit('Please install NumPy to run this script (https://www.np.org)')
try:
    from scipy import signal
except ImportError:
    raise SystemExit('Please install SciPy to run this script (https://www.scipy.org)')
try:
    from PyQt5 import Qt, QtCore, QtWidgets
except ImportError:
    raise SystemExit('Please install PyQt5 to run this script (https://www.riverbankcomputing.com/software/pyqt/download5)')
try:
    import pyqtgraph as pg
except ImportError:
    raise SystemExit('Please install pyqtgraph to run this script (http://www.pyqtgraph.org)')
try:
    from gnuradio.filter.pyqt_filter_stacked import Ui_MainWindow
except ImportError:
    raise SystemExit('Could not import from pyqt_filter_stacked. Please build with "pyuic5 pyqt_filter_stacked.ui -o pyqt_filter_stacked.py"')
try:
    from gnuradio.filter.banditems import *
except ImportError:
    raise SystemExit('Could not import from banditems. Please check whether banditems.py is in the library path')
try:
    from gnuradio.filter.polezero_plot import *
except ImportError:
    raise SystemExit('Could not import from polezero_plot. Please check whether polezero_plot.py is in the library path')
try:
    from gnuradio.filter.api_object import *
except ImportError:
    raise SystemExit('Could not import from api_object. Please check whether api_object.py is in the library path')
try:
    from gnuradio.filter.fir_design import *
except ImportError:
    raise SystemExit('Could not import from fir_design. Please check whether fir_design.py is in the library path')
try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:

    def _fromUtf8(s):
        if False:
            print('Hello World!')
        return s

class gr_plot_filter(QtWidgets.QMainWindow):

    def __init__(self, options, callback=None, restype=''):
        if False:
            while True:
                i = 10
        QtWidgets.QWidget.__init__(self, None)
        self.gui = Ui_MainWindow()
        self.callback = callback
        pg.setConfigOption('foreground', 'k')
        pg.setConfigOption('background', None)
        pg.setConfigOptions(antialias=True)
        self.gui.setupUi(self)
        if restype == 'iir':
            ind = self.gui.fselectComboBox.findText('FIR')
            if ind != -1:
                self.gui.fselectComboBox.removeItem(ind)
        elif restype == 'fir':
            ind = self.gui.fselectComboBox.findText('IIR(scipy)')
            if ind != -1:
                self.gui.fselectComboBox.removeItem(ind)
        self.gui.action_save.triggered.connect(self.action_save_dialog)
        self.gui.action_save.setEnabled(False)
        self.gui.action_open.triggered.connect(self.action_open_dialog)
        self.gui.filterTypeComboBox.currentIndexChanged['const QString&'].connect(self.changed_filter_type)
        self.gui.iirfilterBandComboBox.currentIndexChanged['const QString&'].connect(self.changed_iirfilter_band)
        self.gui.filterDesignTypeComboBox.currentIndexChanged['const QString&'].connect(self.changed_filter_design_type)
        self.gui.fselectComboBox.currentIndexChanged['const QString&'].connect(self.changed_fselect)
        self.gui.iirfilterTypeComboBox.currentIndexChanged['const QString&'].connect(self.set_order)
        self.gui.designButton.released.connect(self.design)
        self.gui.nfftEdit.textEdited['QString'].connect(self.nfft_edit_changed)
        self.gui.actionQuick_Access.triggered.connect(self.action_quick_access)
        self.gui.actionSpec_Widget.triggered.connect(self.action_spec_widget)
        self.gui.actionResponse_Widget.triggered.connect(self.action_response_widget)
        self.gui.actionDesign_Widget.triggered.connect(self.action_design_widget)
        self.gui.actionMagnitude_Response.triggered.connect(self.set_actmagresponse)
        self.gui.actionGrid_2.triggered.connect(self.set_actgrid)
        self.gui.actionPhase_Respone.triggered.connect(self.set_actphase)
        self.gui.actionGroup_Delay.triggered.connect(self.set_actgdelay)
        self.gui.actionFilter_Coefficients.triggered.connect(self.set_actfcoeff)
        self.gui.actionBand_Diagram.triggered.connect(self.set_actband)
        self.gui.actionPole_Zero_Plot_2.triggered.connect(self.set_actpzplot)
        self.gui.actionGridview.triggered.connect(self.set_switchview)
        self.gui.actionPlot_select.triggered.connect(self.set_plotselect)
        self.gui.actionPhase_Delay.triggered.connect(self.set_actpdelay)
        self.gui.actionImpulse_Response.triggered.connect(self.set_actimpres)
        self.gui.actionStep_Response.triggered.connect(self.set_actstepres)
        self.gui.mfmagPush.released.connect(self.set_mfmagresponse)
        self.gui.mfphasePush.released.connect(self.set_mfphaseresponse)
        self.gui.mfgpdlyPush.released.connect(self.set_mfgroupdelay)
        self.gui.mfphdlyPush.released.connect(self.set_mfphasedelay)
        self.gui.mfoverlayPush.clicked.connect(self.set_mfoverlay)
        self.gui.conjPush.clicked.connect(self.set_conj)
        self.gui.mconjPush.clicked.connect(self.set_mconj)
        self.gui.addzeroPush.clicked.connect(self.set_zeroadd)
        self.gui.maddzeroPush.clicked.connect(self.set_mzeroadd)
        self.gui.addpolePush.clicked.connect(self.set_poleadd)
        self.gui.maddpolePush.clicked.connect(self.set_mpoleadd)
        self.gui.delPush.clicked.connect(self.set_delpz)
        self.gui.mdelPush.clicked.connect(self.set_mdelpz)
        self.gui.mttapsPush.clicked.connect(self.set_mttaps)
        self.gui.mtstepPush.clicked.connect(self.set_mtstep)
        self.gui.mtimpPush.clicked.connect(self.set_mtimpulse)
        self.gui.checkGrid.stateChanged['int'].connect(self.set_grid)
        self.gui.checkMagres.stateChanged['int'].connect(self.set_magresponse)
        self.gui.checkGdelay.stateChanged['int'].connect(self.set_gdelay)
        self.gui.checkPhase.stateChanged['int'].connect(self.set_phase)
        self.gui.checkFcoeff.stateChanged['int'].connect(self.set_fcoeff)
        self.gui.checkBand.stateChanged['int'].connect(self.set_band)
        self.gui.checkPzplot.stateChanged['int'].connect(self.set_pzplot)
        self.gui.checkPdelay.stateChanged['int'].connect(self.set_pdelay)
        self.gui.checkImpulse.stateChanged['int'].connect(self.set_impres)
        self.gui.checkStep.stateChanged['int'].connect(self.set_stepres)
        self.gridenable = False
        self.mfoverlay = False
        self.mtoverlay = False
        self.iir = False
        self.mfmagresponse = True
        self.mfphaseresponse = False
        self.mfgroupdelay = False
        self.mfphasedelay = False
        self.mttaps = True
        self.mtstep = False
        self.mtimpulse = False
        self.gui.designButton.setShortcut(QtCore.Qt.Key_Return)
        self.taps = []
        self.a = []
        self.b = []
        self.fftdB = []
        self.fftDeg = []
        self.groupDelay = []
        self.phaseDelay = []
        self.gridview = 0
        self.params = []
        self.nfftpts = int(10000)
        self.gui.nfftEdit.setText(str(self.nfftpts))
        self.firFilters = ('Low Pass', 'Band Pass', 'Complex Band Pass', 'Band Notch', 'High Pass', 'Root Raised Cosine', 'Gaussian', 'Half Band')
        self.optFilters = ('Low Pass', 'Band Pass', 'Complex Band Pass', 'Band Notch', 'High Pass', 'Half Band')
        self.set_windowed()
        self.gui.filterTypeWidget.setCurrentWidget(self.gui.firlpfPage)
        self.gui.iirfilterTypeComboBox.hide()
        self.gui.iirfilterBandComboBox.hide()
        self.gui.adComboBox.hide()
        self.gui.addpolePush.setEnabled(False)
        self.gui.maddpolePush.setEnabled(False)
        self.plots = {'FREQ': None, 'TIME': None, 'PHASE': None, 'GROUP': None, 'IMPRES': None, 'STEPRES': None, 'PDELAY': None}
        self.mplots = {'mFREQ': None, 'mTIME': None}
        self.plots['FREQ'] = self.gui.freqPlot
        self.plots['TIME'] = self.gui.timePlot
        self.plots['PHASE'] = self.gui.phasePlot
        self.plots['GROUP'] = self.gui.groupPlot
        self.plots['IMPRES'] = self.gui.impresPlot
        self.plots['STEPRES'] = self.gui.stepresPlot
        self.plots['PDELAY'] = self.gui.pdelayPlot
        self.mplots['mFREQ'] = self.gui.mfreqPlot
        self.mplots['mTIME'] = self.gui.mtimePlot
        self.labelstyle11b = {'font-family': 'Helvetica', 'font-size': '11pt', 'font-weight': 'bold'}
        self.plots['FREQ'].setLabel('bottom', 'Frequency', units='Hz', **self.labelstyle11b)
        self.plots['FREQ'].setLabel('left', 'Magnitude', units='dB', **self.labelstyle11b)
        self.plots['TIME'].setLabel('bottom', 'Tap number', **self.labelstyle11b)
        self.plots['TIME'].setLabel('left', 'Amplitude', **self.labelstyle11b)
        self.plots['PHASE'].setLabel('bottom', 'Frequency', units='Hz', **self.labelstyle11b)
        self.plots['PHASE'].setLabel('left', 'Phase', units='Radians', **self.labelstyle11b)
        self.plots['GROUP'].setLabel('bottom', 'Frequency', units='Hz', **self.labelstyle11b)
        self.plots['GROUP'].setLabel('left', 'Delay', units='seconds', **self.labelstyle11b)
        self.plots['IMPRES'].setLabel('bottom', 'n', units='Samples', **self.labelstyle11b)
        self.plots['IMPRES'].setLabel('left', 'Amplitude', **self.labelstyle11b)
        self.plots['STEPRES'].setLabel('bottom', 'n', units='Samples', **self.labelstyle11b)
        self.plots['STEPRES'].setLabel('left', 'Amplitude', **self.labelstyle11b)
        self.plots['PDELAY'].setLabel('bottom', 'Frequency', units='Hz', **self.labelstyle11b)
        self.plots['PDELAY'].setLabel('left', 'Phase Delay', units='Radians', **self.labelstyle11b)
        self.labelstyle9b = {'font-family': 'Helvetica', 'font-size': '9pt', 'font-weight': 'bold'}
        self.mplots['mTIME'].setLabel('bottom', 'n', units='Samples/taps', **self.labelstyle9b)
        self.mplots['mTIME'].setLabel('left', 'Amplitude', **self.labelstyle9b)
        for i in self.plots:
            axis = self.plots[i].getAxis('bottom')
            axis.setStyle(tickLength=-10)
            axis = self.plots[i].getAxis('left')
            axis.setStyle(tickLength=-10)
        for i in self.mplots:
            axis = self.mplots[i].getAxis('bottom')
            axis.setStyle(tickLength=-10)
            axis = self.mplots[i].getAxis('left')
            axis.setStyle(tickLength=-10)
        self.rcurve = self.plots['TIME'].plot(title='Real')
        self.icurve = self.plots['TIME'].plot(title='Imag')
        self.mtimecurve = self.mplots['mTIME'].plot(title='PSD')
        self.mtimecurve_stems = self.mplots['mTIME'].plot(connect='pairs', name='Stems')
        self.mtimecurve_i_stems = self.mplots['mTIME'].plot(connect='pairs', name='Stems')
        self.mtimecurve_i = self.mplots['mTIME'].plot(title='Impulse Response Imag')
        self.plots['FREQ'].enableAutoRange(enable=True)
        self.freqcurve = self.plots['FREQ'].plot(title='PSD')
        self.primary_freq_overlay = self.mplots['mFREQ']
        self.mfreqcurve = self.primary_freq_overlay.plot(title='PSD')
        self.secondary_freq_overlay_vb = CustomViewBox()
        self.primary_freq_overlay.scene().addItem(self.secondary_freq_overlay_vb)
        self.primary_freq_overlay.getAxis('right').linkToView(self.secondary_freq_overlay_vb)
        self.mfreqcurve2 = pg.PlotCurveItem()
        self.secondary_freq_overlay_vb.setXLink(self.primary_freq_overlay)
        self.secondary_freq_overlay_vb.addItem(self.mfreqcurve2)
        self.primary_freq_overlay.plotItem.vb.sigResized.connect(self.updateViews)
        self.phasecurve = self.plots['PHASE'].plot(title='Phase')
        self.groupcurve = self.plots['GROUP'].plot(title='Group Delay')
        self.imprescurve_stems = self.plots['IMPRES'].plot(connect='pairs', name='Stems')
        self.imprescurve = self.plots['IMPRES'].plot(title='Impulse Response')
        self.imprescurve_i_stems = self.plots['IMPRES'].plot(connect='pairs', name='Stems')
        self.imprescurve_i = self.plots['IMPRES'].plot(title='Impulse Response Imag')
        self.steprescurve_stems = self.plots['STEPRES'].plot(connect='pairs', name='Stems')
        self.steprescurve = self.plots['STEPRES'].plot(title='Step Response')
        self.steprescurve_i_stems = self.plots['STEPRES'].plot(connect='pairs', name='Stems')
        self.steprescurve_i = self.plots['STEPRES'].plot(title='Step Response Imag')
        self.pdelaycurve = self.plots['PDELAY'].plot(title='Phase Delay')
        self.set_defaultpen()
        self.lpfitems = lpfItems
        self.hpfitems = hpfItems
        self.bpfitems = bpfItems
        self.bnfitems = bnfItems
        self.lpfitems[0].attenChanged.connect(self.set_fatten)
        self.hpfitems[0].attenChanged.connect(self.set_fatten)
        self.bpfitems[0].attenChanged.connect(self.set_fatten)
        self.bnfitems[0].attenChanged.connect(self.set_fatten)
        self.scene = QtWidgets.QGraphicsScene()
        self.scene.setSceneRect(0, 0, 250, 250)
        lightback = QtGui.qRgb(248, 248, 255)
        backbrush = Qt.QBrush(Qt.QColor(lightback))
        self.scene.setBackgroundBrush(backbrush)
        self.gui.bandView.setScene(self.scene)
        self.gui.mbandView.setScene(self.scene)
        self.cpicker = CanvasPicker(self.gui.pzPlot)
        self.cpicker.curveChanged.connect(self.set_curvetaps)
        self.cpicker.mouseposChanged.connect(self.set_statusbar)
        self.cpicker2 = CanvasPicker(self.gui.mpzPlot)
        self.cpicker2.curveChanged.connect(self.set_mcurvetaps)
        self.cpicker2.mouseposChanged.connect(self.set_mstatusbar)
        "\n        self.lpfpassEdit = QtWidgets.QLineEdit()\n        self.lpfpassEdit.setMaximumSize(QtCore.QSize(75,20))\n        self.lpfpassEdit.setText('Not set')\n        self.lpfstartproxy = QtWidgets.QGraphicsProxyWidget()\n        self.lpfstartproxy.setWidget(self.lpfpassEdit)\n        self.lpfstartproxy.setPos(400,30)\n\n        self.lpfstopEdit = QtWidgets.QLineEdit()\n        self.lpfstopEdit.setMaximumSize(QtCore.QSize(75,20))\n        self.lpfstopEdit.setText('Not set')\n        self.lpfstopproxy = QtWidgets.QGraphicsProxyWidget()\n        self.lpfstopproxy.setWidget(self.lpfstopEdit)\n        self.lpfstopproxy.setPos(400,50)\n        self.lpfitems.append(self.lpfstartproxy)\n        self.lpfitems.append(self.lpfstopproxy)\n        "
        self.populate_bandview(self.lpfitems)
        self.intVal = Qt.QIntValidator(None)
        self.dblVal = Qt.QDoubleValidator(None)
        self.gui.nfftEdit.setValidator(self.intVal)
        self.gui.sampleRateEdit.setValidator(self.dblVal)
        self.gui.filterGainEdit.setValidator(self.dblVal)
        self.gui.endofLpfPassBandEdit.setValidator(self.dblVal)
        self.gui.startofLpfStopBandEdit.setValidator(self.dblVal)
        self.gui.lpfStopBandAttenEdit.setValidator(self.dblVal)
        self.gui.lpfPassBandRippleEdit.setValidator(self.dblVal)
        self.gui.startofBpfPassBandEdit.setValidator(self.dblVal)
        self.gui.endofBpfPassBandEdit.setValidator(self.dblVal)
        self.gui.bpfTransitionEdit.setValidator(self.dblVal)
        self.gui.bpfStopBandAttenEdit.setValidator(self.dblVal)
        self.gui.bpfPassBandRippleEdit.setValidator(self.dblVal)
        self.gui.startofBnfStopBandEdit.setValidator(self.dblVal)
        self.gui.endofBnfStopBandEdit.setValidator(self.dblVal)
        self.gui.bnfTransitionEdit.setValidator(self.dblVal)
        self.gui.bnfStopBandAttenEdit.setValidator(self.dblVal)
        self.gui.bnfPassBandRippleEdit.setValidator(self.dblVal)
        self.gui.endofHpfStopBandEdit.setValidator(self.dblVal)
        self.gui.startofHpfPassBandEdit.setValidator(self.dblVal)
        self.gui.hpfStopBandAttenEdit.setValidator(self.dblVal)
        self.gui.hpfPassBandRippleEdit.setValidator(self.dblVal)
        self.gui.rrcSymbolRateEdit.setValidator(self.dblVal)
        self.gui.rrcAlphaEdit.setValidator(self.dblVal)
        self.gui.rrcNumTapsEdit.setValidator(self.dblVal)
        self.gui.gausSymbolRateEdit.setValidator(self.dblVal)
        self.gui.gausBTEdit.setValidator(self.dblVal)
        self.gui.gausNumTapsEdit.setValidator(self.dblVal)
        self.gui.iirendofLpfPassBandEdit.setValidator(self.dblVal)
        self.gui.iirstartofLpfStopBandEdit.setValidator(self.dblVal)
        self.gui.iirLpfPassBandAttenEdit.setValidator(self.dblVal)
        self.gui.iirLpfStopBandRippleEdit.setValidator(self.dblVal)
        self.gui.iirstartofHpfPassBandEdit.setValidator(self.dblVal)
        self.gui.iirendofHpfStopBandEdit.setValidator(self.dblVal)
        self.gui.iirHpfPassBandAttenEdit.setValidator(self.dblVal)
        self.gui.iirHpfStopBandRippleEdit.setValidator(self.dblVal)
        self.gui.iirstartofBpfPassBandEdit.setValidator(self.dblVal)
        self.gui.iirendofBpfPassBandEdit.setValidator(self.dblVal)
        self.gui.iirendofBpfStopBandEdit1.setValidator(self.dblVal)
        self.gui.iirstartofBpfStopBandEdit2.setValidator(self.dblVal)
        self.gui.iirBpfPassBandAttenEdit.setValidator(self.dblVal)
        self.gui.iirBpfStopBandRippleEdit.setValidator(self.dblVal)
        self.gui.iirendofBsfPassBandEdit1.setValidator(self.dblVal)
        self.gui.iirstartofBsfPassBandEdit2.setValidator(self.dblVal)
        self.gui.iirstartofBsfStopBandEdit.setValidator(self.dblVal)
        self.gui.iirendofBsfStopBandEdit.setValidator(self.dblVal)
        self.gui.iirBsfPassBandAttenEdit.setValidator(self.dblVal)
        self.gui.iirBsfStopBandRippleEdit.setValidator(self.dblVal)
        self.gui.besselordEdit.setValidator(self.intVal)
        self.gui.iirbesselcritEdit1.setValidator(self.dblVal)
        self.gui.iirbesselcritEdit2.setValidator(self.dblVal)
        self.gui.nTapsEdit.setText('0')
        self.filterWindows = {'Hamming Window': fft.window.WIN_HAMMING, 'Hann Window': fft.window.WIN_HANN, 'Blackman Window': fft.window.WIN_BLACKMAN, 'Rectangular Window': fft.window.WIN_RECTANGULAR, 'Kaiser Window': fft.window.WIN_KAISER, 'Blackman-harris Window': fft.window.WIN_BLACKMAN_hARRIS}
        self.EQUIRIPPLE_FILT = 6
        self.gui.checkKeepcur.setEnabled(False)
        self.gui.actionIdeal_Band.setEnabled(False)
        self.show()

    def updateViews(self):
        if False:
            i = 10
            return i + 15
        self.secondary_freq_overlay_vb.setGeometry(self.primary_freq_overlay.plotItem.vb.sceneBoundingRect())

    def set_defaultpen(self):
        if False:
            return 10
        blue = QtGui.qRgb(0, 0, 255)
        blueBrush = Qt.QBrush(Qt.QColor(blue))
        red = QtGui.qRgb(255, 0, 0)
        redBrush = Qt.QBrush(Qt.QColor(red))
        self.freqcurve.setPen(pg.mkPen('b', width=1.5))
        self.rcurve.setPen(None)
        self.rcurve.setSymbol('o')
        self.rcurve.setSymbolPen('b')
        self.rcurve.setSymbolBrush(Qt.QBrush(Qt.Qt.gray))
        self.rcurve.setSymbolSize(8)
        self.icurve.setPen(None)
        self.icurve.setSymbol('o')
        self.icurve.setSymbolPen('r')
        self.icurve.setSymbolBrush(Qt.QBrush(Qt.Qt.gray))
        self.icurve.setSymbolSize(8)
        self.imprescurve_stems.setPen(pg.mkPen('b', width=1.5))
        self.imprescurve.setPen(None)
        self.imprescurve.setSymbol('o')
        self.imprescurve.setSymbolPen('b')
        self.imprescurve.setSymbolBrush(Qt.QBrush(Qt.Qt.gray))
        self.imprescurve.setSymbolSize(8)
        self.imprescurve_i_stems.setPen(pg.mkPen('b', width=1.5))
        self.imprescurve_i.setPen(None)
        self.imprescurve_i.setSymbol('o')
        self.imprescurve_i.setSymbolPen('r')
        self.imprescurve_i.setSymbolBrush(Qt.QBrush(Qt.Qt.gray))
        self.imprescurve_i.setSymbolSize(8)
        self.steprescurve_stems.setPen(pg.mkPen('b', width=1.5))
        self.steprescurve.setPen(None)
        self.steprescurve.setSymbol('o')
        self.steprescurve.setSymbolPen('b')
        self.steprescurve.setSymbolBrush(Qt.QBrush(Qt.Qt.gray))
        self.steprescurve.setSymbolSize(8)
        self.steprescurve_i_stems.setPen(pg.mkPen('b', width=1.5))
        self.steprescurve_i.setPen(None)
        self.steprescurve_i.setSymbol('o')
        self.steprescurve_i.setSymbolPen('r')
        self.steprescurve_i.setSymbolBrush(Qt.QBrush(Qt.Qt.gray))
        self.steprescurve_i.setSymbolSize(8)
        self.phasecurve.setPen(pg.mkPen('b', width=1.5))
        self.groupcurve.setPen(pg.mkPen('b', width=1.5))
        self.pdelaycurve.setPen(pg.mkPen('b', width=1.5))
        self.mfreqcurve.setPen(pg.mkPen('b', width=1.5))
        self.mfreqcurve2.setPen(pg.mkPen('r', width=1.5))
        self.mtimecurve.setPen(None)
        self.mtimecurve.setSymbol('o')
        self.mtimecurve.setSymbolPen('b')
        self.mtimecurve.setSymbolBrush(Qt.QBrush(Qt.Qt.gray))
        self.mtimecurve.setSymbolSize(8)
        self.mtimecurve_stems.setPen(pg.mkPen('b', width=1.5))
        self.mtimecurve_i_stems.setPen(pg.mkPen('b', width=1.5))
        self.mtimecurve_i.setPen(None)
        self.mtimecurve_i.setSymbol('o')
        self.mtimecurve_i.setSymbolPen('r')
        self.mtimecurve_i.setSymbolBrush(Qt.QBrush(Qt.Qt.gray))
        self.mtimecurve_i.setSymbolSize(8)

    def changed_fselect(self, ftype):
        if False:
            while True:
                i = 10
        if ftype == 'FIR':
            self.gui.iirfilterTypeComboBox.hide()
            self.gui.iirfilterBandComboBox.hide()
            self.gui.adComboBox.hide()
            self.gui.filterDesignTypeComboBox.show()
            self.gui.globalParamsBox.show()
            self.gui.filterTypeComboBox.show()
            self.gui.filterTypeWidget.setCurrentWidget(self.gui.firlpfPage)
            self.gui.tabGroup.addTab(self.gui.timeTab, _fromUtf8('Filter Taps'))
            self.gui.mttapsPush.setEnabled(True)
            self.gui.addpolePush.setEnabled(False)
            self.gui.maddpolePush.setEnabled(False)
        elif ftype.startswith('IIR'):
            self.gui.filterDesignTypeComboBox.hide()
            self.gui.globalParamsBox.hide()
            self.gui.filterTypeComboBox.hide()
            self.gui.iirfilterTypeComboBox.show()
            self.gui.adComboBox.show()
            self.gui.iirfilterBandComboBox.show()
            self.gui.filterTypeWidget.setCurrentWidget(self.gui.iirlpfPage)
            self.gui.tabGroup.removeTab(self.gui.tabGroup.indexOf(self.gui.timeTab))
            self.gui.mttapsPush.setEnabled(False)
            self.gui.addpolePush.setEnabled(True)
            self.gui.maddpolePush.setEnabled(True)

    def set_order(self, ftype):
        if False:
            return 10
        if ftype == 'Bessel':
            self.gui.filterTypeWidget.setCurrentWidget(self.gui.iirbesselPage)
            self.changed_iirfilter_band(self.gui.iirfilterBandComboBox.currentText())
        else:
            self.changed_iirfilter_band(self.gui.iirfilterBandComboBox.currentText())

    def changed_iirfilter_band(self, ftype):
        if False:
            while True:
                i = 10
        iirftype = self.gui.iirfilterTypeComboBox.currentText()
        if ftype == 'Low Pass':
            if iirftype == 'Bessel':
                self.gui.filterTypeWidget.setCurrentWidget(self.gui.iirbesselPage)
                self.gui.iirbesselcritLabel2.hide()
                self.gui.iirbesselcritEdit2.hide()
            else:
                self.gui.filterTypeWidget.setCurrentWidget(self.gui.iirlpfPage)
        elif ftype == 'Band Pass':
            if iirftype == 'Bessel':
                self.gui.filterTypeWidget.setCurrentWidget(self.gui.iirbesselPage)
                self.gui.iirbesselcritLabel2.show()
                self.gui.iirbesselcritEdit2.show()
            else:
                self.gui.filterTypeWidget.setCurrentWidget(self.gui.iirbpfPage)
        elif ftype == 'Band Stop':
            if iirftype == 'Bessel':
                self.gui.filterTypeWidget.setCurrentWidget(self.gui.iirbesselPage)
                self.gui.iirbesselcritLabel2.show()
                self.gui.iirbesselcritEdit2.show()
            else:
                self.gui.filterTypeWidget.setCurrentWidget(self.gui.iirbsfPage)
        elif ftype == 'High Pass':
            if iirftype == 'Bessel':
                self.gui.filterTypeWidget.setCurrentWidget(self.gui.iirbesselPage)
                self.gui.iirbesselcritLabel2.hide()
                self.gui.iirbesselcritEdit2.hide()
            else:
                self.gui.filterTypeWidget.setCurrentWidget(self.gui.iirhpfPage)

    def changed_filter_type(self, ftype):
        if False:
            print('Hello World!')
        if ftype == 'Low Pass':
            self.gui.filterTypeWidget.setCurrentWidget(self.gui.firlpfPage)
            self.remove_bandview()
            self.populate_bandview(self.lpfitems)
        elif ftype == 'Band Pass':
            self.gui.filterTypeWidget.setCurrentWidget(self.gui.firbpfPage)
            self.remove_bandview()
            self.populate_bandview(self.bpfitems)
        elif ftype == 'Complex Band Pass':
            self.gui.filterTypeWidget.setCurrentWidget(self.gui.firbpfPage)
            self.remove_bandview()
            self.populate_bandview(self.bpfitems)
        elif ftype == 'Band Notch':
            self.gui.filterTypeWidget.setCurrentWidget(self.gui.firbnfPage)
            self.remove_bandview()
            self.populate_bandview(self.bnfitems)
        elif ftype == 'High Pass':
            self.gui.filterTypeWidget.setCurrentWidget(self.gui.firhpfPage)
            self.remove_bandview()
            self.populate_bandview(self.hpfitems)
        elif ftype == 'Root Raised Cosine':
            self.gui.filterTypeWidget.setCurrentWidget(self.gui.rrcPage)
        elif ftype == 'Gaussian':
            self.gui.filterTypeWidget.setCurrentWidget(self.gui.gausPage)
        elif ftype == 'Half Band':
            self.gui.filterTypeWidget.setCurrentWidget(self.gui.firhbPage)

    def changed_filter_design_type(self, design):
        if False:
            print('Hello World!')
        if design == 'Equiripple':
            self.set_equiripple()
        else:
            self.set_windowed()

    def set_equiripple(self):
        if False:
            while True:
                i = 10
        self.gui.filterTypeComboBox.blockSignals(True)
        self.equiripple = True
        self.gui.lpfPassBandRippleLabel.setVisible(True)
        self.gui.lpfPassBandRippleEdit.setVisible(True)
        self.gui.bpfPassBandRippleLabel.setVisible(True)
        self.gui.bpfPassBandRippleEdit.setVisible(True)
        self.gui.bnfPassBandRippleLabel.setVisible(True)
        self.gui.bnfPassBandRippleEdit.setVisible(True)
        self.gui.hpfPassBandRippleLabel.setVisible(True)
        self.gui.hpfPassBandRippleEdit.setVisible(True)
        currenttype = self.gui.filterTypeComboBox.currentText()
        items = self.gui.filterTypeComboBox.count()
        for i in range(items):
            self.gui.filterTypeComboBox.removeItem(0)
        self.gui.filterTypeComboBox.addItems(self.optFilters)
        try:
            index = self.optFilters.index(currenttype)
            self.gui.filterTypeComboBox.setCurrentIndex(index)
        except ValueError:
            pass
        self.gui.filterTypeComboBox.blockSignals(False)

    def set_windowed(self):
        if False:
            for i in range(10):
                print('nop')
        self.gui.filterTypeComboBox.blockSignals(True)
        self.equiripple = False
        self.gui.lpfPassBandRippleLabel.setVisible(False)
        self.gui.lpfPassBandRippleEdit.setVisible(False)
        self.gui.bpfPassBandRippleLabel.setVisible(False)
        self.gui.bpfPassBandRippleEdit.setVisible(False)
        self.gui.bnfPassBandRippleLabel.setVisible(False)
        self.gui.bnfPassBandRippleEdit.setVisible(False)
        self.gui.hpfPassBandRippleLabel.setVisible(False)
        self.gui.hpfPassBandRippleEdit.setVisible(False)
        currenttype = self.gui.filterTypeComboBox.currentText()
        items = self.gui.filterTypeComboBox.count()
        for i in range(items):
            self.gui.filterTypeComboBox.removeItem(0)
        self.gui.filterTypeComboBox.addItems(self.firFilters)
        try:
            index = self.optFilters.index(currenttype)
            self.gui.filterTypeComboBox.setCurrentIndex(index)
        except ValueError:
            pass
        self.gui.filterTypeComboBox.blockSignals(False)

    def design(self):
        if False:
            print('Hello World!')
        ret = True
        (fs, r) = getfloat(self.gui.sampleRateEdit.text())
        ret = r and ret
        (gain, r) = getfloat(self.gui.filterGainEdit.text())
        ret = r and ret
        winstr = self.gui.filterDesignTypeComboBox.currentText()
        ftype = self.gui.filterTypeComboBox.currentText()
        fsel = self.gui.fselectComboBox.currentText()
        if fsel == 'FIR':
            (self.b, self.a) = ([], [])
            if ret:
                self.design_fir(ftype, fs, gain, winstr)
        elif fsel.startswith('IIR'):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                self.design_iir()
                if len(w):
                    reply = QtWidgets.QMessageBox.information(self, 'BadCoefficients', str(w[-1].message), QtWidgets.QMessageBox.Ok)

    def design_fir(self, ftype, fs, gain, winstr):
        if False:
            while True:
                i = 10
        self.iir = False
        self.cpicker.set_iir(False)
        self.cpicker2.set_iir(False)
        if winstr == 'Equiripple':
            designer = {'Low Pass': design_opt_lpf, 'Band Pass': design_opt_bpf, 'Complex Band Pass': design_opt_cbpf, 'Band Notch': design_opt_bnf, 'Half Band': design_opt_hb, 'High Pass': design_opt_hpf}
            (taps, params, r) = designer[ftype](fs, gain, self)
        else:
            designer = {'Low Pass': design_win_lpf, 'Band Pass': design_win_bpf, 'Complex Band Pass': design_win_cbpf, 'Band Notch': design_win_bnf, 'High Pass': design_win_hpf, 'Half Band': design_win_hb, 'Root Raised Cosine': design_win_rrc, 'Gaussian': design_win_gaus}
            wintype = int(self.filterWindows[winstr])
            (taps, params, r) = designer[ftype](fs, gain, wintype, self)
        if r:
            if self.gridview:
                self.params = params
                self.update_fft(taps, params)
                self.set_mfmagresponse()
                self.set_mttaps()
                self.gui.nTapsEdit.setText(str(self.taps.size))
            else:
                self.draw_plots(taps, params)
        zeros = self.get_zeros()
        poles = self.get_poles()
        self.gui.pzPlot.insertZeros(zeros)
        self.gui.pzPlot.insertPoles(poles)
        self.gui.mpzPlot.insertZeros(zeros)
        self.gui.mpzPlot.insertPoles(poles)
        self.update_fcoeff()
        self.gui.action_save.setEnabled(True)
        if self.callback:
            retobj = ApiObject()
            retobj.update_all('fir', self.params, self.taps, 1)
            self.callback(retobj)

    def design_iir(self):
        if False:
            for i in range(10):
                print('nop')
        iirftype = self.gui.iirfilterTypeComboBox.currentText()
        iirbtype = self.gui.iirfilterBandComboBox.currentText()
        atype = self.gui.adComboBox.currentText()
        self.taps = []
        self.iir = True
        ret = True
        params = []
        besselparams = []
        self.cpicker.set_iir(True)
        self.cpicker2.set_iir(True)
        iirft = {'Elliptic': 'ellip', 'Butterworth': 'butter', 'Chebyshev-1': 'cheby1', 'Chebyshev-2': 'cheby2', 'Bessel': 'bessel'}
        sanalog = {'Analog (rad/second)': 1, 'Digital (normalized 0-1)': 0}
        paramtype = {1: 'analog', 0: 'digital'}
        iirabbr = {'Low Pass': 'lpf', 'High Pass': 'hpf', 'Band Pass': 'bpf', 'Band Stop': 'bnf'}
        iirboxes = {'Low Pass': [float(self.gui.iirendofLpfPassBandEdit.text()), float(self.gui.iirstartofLpfStopBandEdit.text()), float(self.gui.iirLpfPassBandAttenEdit.text()), float(self.gui.iirLpfStopBandRippleEdit.text())], 'High Pass': [float(self.gui.iirstartofHpfPassBandEdit.text()), float(self.gui.iirendofHpfStopBandEdit.text()), float(self.gui.iirHpfPassBandAttenEdit.text()), float(self.gui.iirHpfStopBandRippleEdit.text())], 'Band Pass': [float(self.gui.iirstartofBpfPassBandEdit.text()), float(self.gui.iirendofBpfPassBandEdit.text()), float(self.gui.iirendofBpfStopBandEdit1.text()), float(self.gui.iirstartofBpfStopBandEdit2.text()), float(self.gui.iirBpfPassBandAttenEdit.text()), float(self.gui.iirBpfStopBandRippleEdit.text())], 'Band Stop': [float(self.gui.iirendofBsfPassBandEdit1.text()), float(self.gui.iirstartofBsfPassBandEdit2.text()), float(self.gui.iirstartofBsfStopBandEdit.text()), float(self.gui.iirendofBsfStopBandEdit.text()), float(self.gui.iirBsfPassBandAttenEdit.text()), float(self.gui.iirBsfStopBandRippleEdit.text())]}
        for i in range(len(iirboxes[iirbtype])):
            params.append(iirboxes[iirbtype][i])
        if len(iirboxes[iirbtype]) == 6:
            params = [params[:2], params[2:4], params[4], params[5]]
        if iirftype == 'Bessel':
            if iirbtype == 'Low Pass' or iirbtype == 'High Pass':
                besselparams.append(float(self.gui.iirbesselcritEdit1.text()))
            else:
                besselparams.append(float(self.gui.iirbesselcritEdit1.text()))
                besselparams.append(float(self.gui.iirbesselcritEdit2.text()))
            order = int(self.gui.besselordEdit.text())
            try:
                (self.b, self.a) = signal.iirfilter(order, besselparams, btype=iirbtype.replace(' ', '').lower(), analog=sanalog[atype], ftype=iirft[iirftype], output='ba')
            except Exception as e:
                reply = QtWidgets.QMessageBox.information(self, 'IIR design error', e.args[0], QtWidgets.QMessageBox.Ok)
            (self.z, self.p, self.k) = signal.tf2zpk(self.b, self.a)
            iirparams = {'filttype': iirft[iirftype], 'bandtype': iirabbr[iirbtype], 'filtord': order, 'paramtype': paramtype[sanalog[atype]], 'critfreq': besselparams}
        else:
            try:
                (self.b, self.a) = signal.iirdesign(params[0], params[1], params[2], params[3], analog=sanalog[atype], ftype=iirft[iirftype], output='ba')
            except Exception as e:
                reply = QtWidgets.QMessageBox.information(self, 'IIR design error', e.args[0], QtWidgets.QMessageBox.Ok)
            (self.z, self.p, self.k) = signal.tf2zpk(self.b, self.a)
            iirparams = {'filttype': iirft[iirftype], 'bandtype': iirabbr[iirbtype], 'paramtype': paramtype[sanalog[atype]], 'pbedge': params[0], 'sbedge': params[1], 'gpass': params[2], 'gstop': params[3]}
        self.gui.pzPlot.insertZeros(self.z)
        self.gui.pzPlot.insertPoles(self.p)
        self.gui.mpzPlot.insertZeros(self.z)
        self.gui.mpzPlot.insertPoles(self.p)
        self.iir_plot_all(self.z, self.p, self.k)
        self.update_fcoeff()
        self.gui.nTapsEdit.setText('-')
        self.params = iirparams
        self.gui.action_save.setEnabled(True)
        if self.callback:
            retobj = ApiObject()
            retobj.update_all('iir', self.params, (self.b, self.a), 1)
            self.callback(retobj)

    def iir_plot_all(self, z, p, k):
        if False:
            return 10
        (self.b, self.a) = signal.zpk2tf(z, p, k)
        (w, h) = signal.freqz(self.b, self.a)
        self.fftdB = 20 * np.log10(abs(h))
        self.freq = w / max(w)
        self.fftDeg = np.unwrap(np.arctan2(np.imag(h), np.real(h)))
        self.groupDelay = -np.diff(self.fftDeg)
        self.phaseDelay = -self.fftDeg[1:] / self.freq[1:]
        if self.gridview:
            self.set_mfmagresponse()
            self.set_mtimpulse()
        else:
            self.update_freq_curves()
            self.update_phase_curves()
            self.update_group_curves()
            self.update_pdelay_curves()
            self.update_step_curves()
            self.update_imp_curves()

    def nfft_edit_changed(self, nfft):
        if False:
            i = 10
            return i + 15
        (infft, r) = getint(nfft)
        if r and infft != self.nfftpts:
            self.nfftpts = infft
            self.update_freq_curves()

    def get_fft(self, fs, taps, Npts):
        if False:
            i = 10
            return i + 15
        fftpts = fft_detail.fft(taps, Npts)
        self.freq = np.linspace(start=0, stop=fs, num=Npts, endpoint=False)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            self.fftdB = 20.0 * np.log10(abs(fftpts))
            if any(self.fftdB == float('-inf')):
                sys.stderr.write('Filter design failed (taking log10 of 0).\n')
                self.fftdB = np.zeros([len(fftpts)])
        self.fftDeg = np.unwrap(np.angle(fftpts))
        self.groupDelay = -np.diff(self.fftDeg)
        self.phaseDelay = -self.fftDeg[1:] / self.freq[1:]

    def update_time_curves(self):
        if False:
            while True:
                i = 10
        ntaps = len(self.taps)
        if ntaps < 1:
            return
        if type(self.taps[0]) == scipy.complex128:
            self.rcurve.setData(np.arange(ntaps), self.taps.real)
            self.icurve.setData(np.arange(ntaps), self.taps.imag)
        else:
            self.rcurve.setData(np.arange(ntaps), self.taps)
            self.icurve.setData([], [])
        if self.mttaps:
            if type(self.taps[0]) == scipy.complex128:
                self.mtimecurve_stems.setData(np.repeat(np.arange(ntaps), 2), np.dstack((np.zeros(self.taps.real.shape[0], dtype=int), self.taps.real)).flatten())
                self.mtimecurve.setData(np.arange(ntaps), self.taps.real)
                self.mtimecurve_i_stems.setData(np.repeat(np.arange(ntaps), 2), np.dstack((np.zeros(self.taps.imag.shape[0], dtype=int), self.taps.imag)).flatten())
                self.mtimecurve_i.setData(np.arange(ntaps), self.taps.imag)
            else:
                self.mtimecurve.setData(np.arange(ntaps), self.taps)
                self.mtimecurve_stems.setData(np.repeat(np.arange(ntaps), 2), np.dstack((np.zeros(self.taps.shape[0], dtype=int), self.taps)).flatten())
                self.mtimecurve_i_stems.setData([], [])
                self.mtimecurve_i.setData([], [])
        if self.mtoverlay:
            self.mplots['mTIME'].setMouseEnabled(x=True, y=True)
        else:
            self.mplots['mTIME'].setMouseEnabled(x=False, y=False)
            self.mplots['mTIME'].showAxis('right', False)
        self.plot_auto_limit(self.plots['TIME'], xMin=0, xMax=ntaps)
        self.plot_auto_limit(self.mplots['mTIME'], xMin=0, xMax=ntaps)

    def update_step_curves(self):
        if False:
            print('Hello World!')
        ntaps = len(self.taps)
        if ntaps < 1 and (not self.iir):
            return
        if self.iir:
            stepres = self.step_response(self.b, self.a)
            ntaps = 50
        else:
            stepres = self.step_response(self.taps)
        if type(stepres[0]) == np.complex128:
            self.steprescurve_stems.setData(np.repeat(np.arange(ntaps), 2), np.dstack((np.zeros(stepres.real.shape[0], dtype=int), stepres.real)).flatten())
            self.steprescurve.setData(np.arange(ntaps), stepres.real)
            self.steprescurve_i_stems.setData(np.repeat(np.arange(ntaps), 2), np.dstack((np.zeros(stepres.imag.shape[0], dtype=int), stepres.imag)).flatten())
            self.steprescurve_i.setData(np.arange(ntaps), stepres.imag)
        else:
            self.steprescurve_stems.setData(np.repeat(np.arange(ntaps), 2), np.dstack((np.zeros(stepres.shape[0], dtype=int), stepres)).flatten())
            self.steprescurve.setData(np.arange(ntaps), stepres)
            self.steprescurve_i_stems.setData([], [])
            self.steprescurve_i.setData([], [])
        if self.mtstep:
            if type(stepres[0]) == np.complex128:
                self.mtimecurve_stems.setData(np.repeat(np.arange(ntaps), 2), np.dstack((np.zeros(stepres.real.shape[0], dtype=int), stepres.real)).flatten())
                self.mtimecurve.setData(np.arange(ntaps), stepres.real)
                self.mtimecurve_i_stems.setData(np.repeat(np.arange(ntaps), 2), np.dstack((np.zeros(stepres.imag.shape[0], dtype=int), stepres.imag)).flatten())
                self.mtimecurve_i.setData(np.arange(ntaps), stepres.imag)
            else:
                self.mtimecurve_stems.setData(np.repeat(np.arange(ntaps), 2), np.dstack((np.zeros(stepres.shape[0], dtype=int), stepres)).flatten())
                self.mtimecurve.setData(np.arange(ntaps), stepres)
                self.mtimecurve_i_stems.setData([], [])
                self.mtimecurve_i.setData([], [])
        if self.mtoverlay:
            self.mplots['mTIME'].setMouseEnabled(x=True, y=True)
        else:
            self.mplots['mTIME'].setMouseEnabled(x=False, y=False)
            self.mplots['mTIME'].showAxis('right', False)
        self.plot_auto_limit(self.plots['STEPRES'], xMin=0, xMax=ntaps)
        self.plot_auto_limit(self.mplots['mTIME'], xMin=0, xMax=ntaps)

    def update_imp_curves(self):
        if False:
            print('Hello World!')
        ntaps = len(self.taps)
        if ntaps < 1 and (not self.iir):
            return
        if self.iir:
            impres = self.impulse_response(self.b, self.a)
            ntaps = 50
        else:
            impres = self.impulse_response(self.taps)
        if type(impres[0]) == np.complex128:
            self.imprescurve_stems.setData(np.repeat(np.arange(ntaps), 2), np.dstack((np.zeros(impres.real.shape[0], dtype=int), impres.real)).flatten())
            self.imprescurve.setData(np.arange(ntaps), impres.real)
            self.imprescurve_i_stems.setData(np.repeat(np.arange(ntaps), 2), np.dstack((np.zeros(impres.imag.shape[0], dtype=int), impres.imag)).flatten())
            self.imprescurve_i.setData(np.arange(ntaps), impres.imag)
        else:
            self.imprescurve_stems.setData(np.repeat(np.arange(ntaps), 2), np.dstack((np.zeros(impres.shape[0], dtype=int), impres)).flatten())
        if self.mtimpulse:
            if type(impres[0]) == np.complex128:
                self.mtimecurve_stems.setData(np.repeat(np.arange(ntaps), 2), np.dstack((np.zeros(impres.real.shape[0], dtype=int), impres.real)).flatten())
                self.mtimecurve.setData(np.arange(ntaps), impres.real)
                self.mtimecurve_i_stems.setData(np.repeat(np.arange(ntaps), 2), np.dstack((np.zeros(impres.imag.shape[0], dtype=int), impres.imag)).flatten())
                self.mtimecurve_i.setData(np.arange(ntaps), impres.imag)
            else:
                self.mtimecurve_stems.setData(np.repeat(np.arange(ntaps), 2), np.dstack((np.zeros(impres.shape[0], dtype=int), impres)).flatten())
                self.mtimecurve.setData(np.arange(ntaps), impres)
                self.mtimecurve_i_stems.setData([], [])
                self.mtimecurve_i.setData([], [])
        if self.mtoverlay:
            self.mplots['mTIME'].setMouseEnabled(x=True, y=True)
        else:
            self.mplots['mTIME'].setMouseEnabled(x=False, y=False)
            self.mplots['mTIME'].showAxis('right', False)
        self.plot_auto_limit(self.plots['IMPRES'], xMin=0, xMax=ntaps)
        self.plot_auto_limit(self.mplots['mTIME'], xMin=0, xMax=ntaps)

    def plot_secondary(self):
        if False:
            for i in range(10):
                print('nop')
        if self.mfoverlay:
            if self.last_mfreq_plot == 'freq':
                self.mfmagresponse = True
                self.update_freq_curves(True)
            elif self.last_mfreq_plot == 'phase':
                self.mfphaseresponse = True
                self.update_phase_curves(True)
            elif self.last_mfreq_plot == 'group':
                self.mfgroupdelay = True
                self.update_group_curves(True)
            elif self.last_mfreq_plot == 'pdelay':
                self.mfphasedelay = True
                self.update_pdelay_curves(True)
            self.mplots['mFREQ'].showAxis('right', True)
        else:
            self.mplots['mFREQ'].setMouseEnabled(x=False, y=False)
            self.mplots['mFREQ'].showAxis('right', False)
            self.mfreqcurve2.setData([], [])

    def update_freq_curves(self, secondary=False):
        if False:
            return 10
        npts = len(self.fftdB)
        if npts < 1:
            return
        if self.iir:
            self.freqcurve.setData(self.freq[:npts - 1], self.fftdB[:npts - 1])
        else:
            self.freqcurve.setData(self.freq[:int(npts // 2)], self.fftdB[:int(npts // 2)])
        if self.mfmagresponse:
            curve = self.mfreqcurve
            if secondary:
                curve = self.mfreqcurve2
            if self.iir:
                curve.setData(self.freq[:npts - 1], self.fftdB[:npts - 1])
            else:
                curve.setData(self.freq[:int(npts // 2)], self.fftdB[:int(npts // 2)])
        if self.iir:
            xmax = self.freq[npts - 1]
        else:
            xmax = self.freq[npts // 2]
        xmin = self.freq[0]
        self.plot_auto_limit(self.plots['FREQ'], xMin=xmin, xMax=xmax)
        self.plot_auto_limit(self.mplots['mFREQ'], xMin=xmin, xMax=xmax)
        if secondary:
            self.mplots['mFREQ'].setLabel('right', 'Magnitude', units='dB', **self.labelstyle9b)
        else:
            self.mplots['mFREQ'].setLabel('left', 'Magnitude', units='dB', **self.labelstyle9b)
        if not secondary:
            self.plot_secondary()
            self.last_mfreq_plot = 'freq'

    def update_phase_curves(self, secondary=False):
        if False:
            print('Hello World!')
        npts = len(self.fftDeg)
        if npts < 1:
            return
        if self.iir:
            self.phasecurve.setData(self.freq[:npts - 1], self.fftDeg[:npts - 1])
        else:
            self.phasecurve.setData(self.freq[:int(npts // 2)], self.fftDeg[:int(npts // 2)])
        if self.mfphaseresponse:
            curve = self.mfreqcurve
            if secondary:
                curve = self.mfreqcurve2
            if self.iir:
                curve.setData(self.freq[:npts - 1], self.fftDeg[:npts - 1])
            else:
                curve.setData(self.freq[:int(npts // 2)], self.fftDeg[:int(npts // 2)])
        if self.iir:
            xmax = self.freq[npts - 1]
        else:
            xmax = self.freq[npts // 2]
        xmin = self.freq[0]
        self.plot_auto_limit(self.plots['PHASE'], xMin=xmin, xMax=xmax)
        self.plot_auto_limit(self.mplots['mFREQ'], xMin=xmin, xMax=xmax)
        if secondary:
            self.mplots['mFREQ'].setLabel('right', 'Phase', units='Radians', **self.labelstyle9b)
        else:
            self.mplots['mFREQ'].setLabel('left', 'Phase', units='Radians', **self.labelstyle9b)
        if not secondary:
            self.plot_secondary()
            self.last_mfreq_plot = 'phase'

    def update_group_curves(self, secondary=False):
        if False:
            for i in range(10):
                print('nop')
        npts = len(self.groupDelay)
        if npts < 1:
            return
        if self.iir:
            self.groupcurve.setData(self.freq[:npts - 1], self.groupDelay[:npts - 1])
        else:
            self.groupcurve.setData(self.freq[:int(npts // 2)], self.groupDelay[:int(npts // 2)])
        if self.mfgroupdelay:
            curve = self.mfreqcurve
            if secondary:
                curve = self.mfreqcurve2
            if self.iir:
                curve.setData(self.freq[:npts - 1], self.groupDelay[:npts - 1])
            else:
                curve.setData(self.freq[:int(npts // 2)], self.groupDelay[:int(npts // 2)])
        if self.mtoverlay:
            self.mplots['mFREQ'].setMouseEnabled(x=True, y=True)
        else:
            self.mplots['mFREQ'].setMouseEnabled(x=False, y=False)
            self.mplots['mFREQ'].showAxis('right', False)
        if self.iir:
            xmax = self.freq[npts - 1]
        else:
            xmax = self.freq[npts // 2]
        xmin = self.freq[0]
        self.plot_auto_limit(self.plots['GROUP'], xMin=xmin, xMax=xmax)
        self.plot_auto_limit(self.mplots['mFREQ'], xMin=xmin, xMax=xmax)
        if secondary:
            self.mplots['mFREQ'].setLabel('right', 'Delay', units='seconds', **self.labelstyle9b)
        else:
            self.mplots['mFREQ'].setLabel('left', 'Delay', units='seconds', **self.labelstyle9b)
        if not secondary:
            self.plot_secondary()
            self.last_mfreq_plot = 'group'

    def update_pdelay_curves(self, secondary=False):
        if False:
            i = 10
            return i + 15
        npts = len(self.phaseDelay)
        if npts < 1:
            return
        if self.iir:
            self.pdelaycurve.setData(self.freq[:npts - 1], self.phaseDelay[:npts - 1])
        else:
            self.pdelaycurve.setData(self.freq[:int(npts // 2)], self.phaseDelay[:int(npts // 2)])
        if self.mfphasedelay:
            curve = self.mfreqcurve
            if secondary:
                curve = self.mfreqcurve2
            if self.iir:
                curve.setData(self.freq[:npts - 1], self.phaseDelay[:npts - 1])
            else:
                curve.setData(self.freq[:int(npts // 2)], self.phaseDelay[:int(npts // 2)])
        if self.iir:
            xmax = self.freq[npts - 1]
        else:
            xmax = self.freq[npts // 2]
        xmin = self.freq[0]
        self.plot_auto_limit(self.plots['PDELAY'], xMin=xmin, xMax=xmax)
        self.plot_auto_limit(self.mplots['mFREQ'], xMin=xmin, xMax=xmax)
        if secondary:
            self.mplots['mFREQ'].setLabel('right', 'Phase Delay', **self.labelstyle9b)
        else:
            self.mplots['mFREQ'].setLabel('left', 'Phase Delay', **self.labelstyle9b)
        if not secondary:
            self.plot_secondary()
            self.last_mfreq_plot = 'pdelay'

    def plot_auto_limit(self, plot, xMin=None, xMax=None, yMin=None, yMax=None):
        if False:
            print('Hello World!')
        plot.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
        plot.autoRange()
        view = plot.viewRange()
        xmin = view[0][0] if xMin is None else xMin
        xmax = view[0][1] if xMax is None else xMax
        ymin = view[1][0] if yMin is None else yMin
        ymax = view[1][1] if yMax is None else yMax
        plot.setLimits(xMin=xmin, xMax=xmax, yMin=ymin, yMax=ymax)

    def action_quick_access(self):
        if False:
            for i in range(10):
                print('nop')
        if self.gui.quickFrame.isHidden():
            self.gui.quickFrame.show()
        else:
            self.gui.quickFrame.hide()

    def action_spec_widget(self):
        if False:
            return 10
        if self.gui.filterspecView.isHidden():
            self.gui.filterspecView.show()
        else:
            self.gui.filterspecView.hide()

    def action_response_widget(self):
        if False:
            while True:
                i = 10
        if self.gui.tabGroup.isHidden():
            self.gui.tabGroup.show()
        else:
            self.gui.tabGroup.hide()

    def action_design_widget(self):
        if False:
            print('Hello World!')
        if self.gui.filterFrame.isHidden():
            self.gui.filterFrame.show()
        else:
            self.gui.filterFrame.hide()

    def set_grid(self):
        if False:
            while True:
                i = 10
        if self.gui.checkGrid.checkState() == 0:
            self.gridenable = False
            for i in self.plots:
                self.plots[i].showGrid(x=False, y=False)
            for i in self.mplots:
                self.mplots[i].showGrid(x=False, y=False)
        else:
            self.gridenable = True
            if self.gridview:
                for i in self.mplots:
                    self.mplots[i].showGrid(x=True, y=True)
            else:
                for i in self.plots:
                    self.plots[i].showGrid(x=True, y=True)

    def set_actgrid(self):
        if False:
            for i in range(10):
                print('nop')
        if self.gui.actionGrid_2.isChecked() == 0:
            self.gridenable = False
            for i in self.plots:
                self.plots[i].showGrid(x=False, y=False)
            for i in self.mplots:
                self.mplots[i].showGrid(x=False, y=False)
        else:
            self.gridenable = True
            if self.gridview:
                for i in self.mplots:
                    self.mplots[i].showGrid(x=True, y=True)
            else:
                for i in self.plots:
                    self.plots[i].showGrid(x=True, y=True)

    def set_magresponse(self):
        if False:
            print('Hello World!')
        if self.gui.checkMagres.checkState() == 0:
            self.magres = False
            self.gui.tabGroup.removeTab(self.gui.tabGroup.indexOf(self.gui.freqTab))
        else:
            self.magres = True
            self.gui.tabGroup.addTab(self.gui.freqTab, _fromUtf8('Magnitude Response'))
            self.update_freq_curves()

    def set_actmagresponse(self):
        if False:
            while True:
                i = 10
        if self.gui.actionMagnitude_Response.isChecked() == 0:
            self.gui.tabGroup.removeTab(self.gui.tabGroup.indexOf(self.gui.freqTab))
        else:
            self.gui.tabGroup.addTab(self.gui.freqTab, _fromUtf8('Magnitude Response'))
            self.update_freq_curves()

    def set_switchview(self):
        if False:
            while True:
                i = 10
        if self.gui.actionGridview.isChecked() == 0:
            self.gridview = 0
            self.set_defaultpen()
            self.set_actgrid()
            self.gui.stackedWindows.setCurrentIndex(0)
            if self.iir:
                self.iir_plot_all(self.z, self.p, self.k)
            else:
                self.draw_plots(self.taps, self.params)
        else:
            self.gridview = 1
            self.set_actgrid()
            self.gui.stackedWindows.setCurrentIndex(1)
            self.update_freq_curves()
            self.update_time_curves()

    def set_plotselect(self):
        if False:
            print('Hello World!')
        if self.gui.actionPlot_select.isChecked() == 0:
            self.gui.mfgroupBox.hide()
            self.gui.mtgroupBox.hide()
            self.gui.pzgroupBox.hide()
            self.gui.mpzgroupBox.hide()
        else:
            self.gui.mfgroupBox.show()
            self.gui.mtgroupBox.show()
            self.gui.pzgroupBox.show()
            self.gui.mpzgroupBox.show()

    def replot_all(self):
        if False:
            for i in range(10):
                print('nop')
        self.plots['TIME'].replot()
        self.mplots['mTIME'].replot()
        self.plots['FREQ'].replot()
        self.mplots['mFREQ'].replot()
        self.plots['PHASE'].replot()
        self.plots['GROUP'].replot()
        self.plots['IMPRES'].replot()
        self.plots['STEPRES'].replot()
        self.plots['PDELAY'].replot()

    def detach_allgrid(self):
        if False:
            while True:
                i = 10
        for i in self.plots:
            i.showGrid(x=False, y=False)

    def set_mfmagresponse(self):
        if False:
            i = 10
            return i + 15
        if self.mfoverlay:
            self.mfmagresponse = True
        else:
            self.mfmagresponse = not self.mfmagresponse
        self.mfphasedelay = False
        self.mfgroupdelay = False
        self.mfphaseresponse = False
        self.update_freq_curves()

    def set_mfphaseresponse(self):
        if False:
            return 10
        if self.mfoverlay:
            self.mfphaseresponse = True
        else:
            self.mfphaseresponse = not self.mfphaseresponse
        self.mfphasedelay = False
        self.mfgroupdelay = False
        self.mfmagresponse = False
        self.update_phase_curves()

    def set_mfgroupdelay(self):
        if False:
            for i in range(10):
                print('nop')
        if self.mfoverlay:
            self.mfgroupdelay = True
        else:
            self.mfgroupdelay = not self.mfgroupdelay
        self.mfphasedelay = False
        self.mfphaseresponse = False
        self.mfmagresponse = False
        self.update_group_curves()

    def set_mfphasedelay(self):
        if False:
            print('Hello World!')
        if self.mfoverlay:
            self.mfphasedelay = True
        else:
            self.mfphasedelay = not self.mfphasedelay
        self.mfgroupdelay = False
        self.mfphaseresponse = False
        self.mfmagresponse = False
        self.update_pdelay_curves()

    def ifinlist(self, a, dlist):
        if False:
            i = 10
            return i + 15
        for d in dlist:
            if self.compare_instances(a, d):
                return True
        return False

    def compare_instances(self, a, b):
        if False:
            while True:
                i = 10
        if a is b:
            return True
        else:
            return False

    def update_fft(self, taps, params):
        if False:
            print('Hello World!')
        self.params = params
        self.taps = np.array(taps)
        self.get_fft(self.params['fs'], self.taps, self.nfftpts)

    def set_mfoverlay(self):
        if False:
            for i in range(10):
                print('nop')
        self.mfoverlay = not self.mfoverlay

    def set_conj(self):
        if False:
            return 10
        self.cpicker.set_conjugate()

    def set_mconj(self):
        if False:
            print('Hello World!')
        self.cpicker2.set_conjugate()

    def set_zeroadd(self):
        if False:
            return 10
        self.cpicker.add_zero()

    def set_mzeroadd(self):
        if False:
            while True:
                i = 10
        self.cpicker2.add_zero()

    def set_poleadd(self):
        if False:
            print('Hello World!')
        self.cpicker.add_pole()

    def set_mpoleadd(self):
        if False:
            print('Hello World!')
        self.cpicker2.add_pole()

    def set_delpz(self):
        if False:
            while True:
                i = 10
        self.cpicker.delete_pz()

    def set_mdelpz(self):
        if False:
            return 10
        self.cpicker2.delete_pz()

    def set_mttaps(self):
        if False:
            print('Hello World!')
        self.mttaps = not self.mttaps
        if not self.mfoverlay:
            self.mtstep = False
            self.mtimpulse = False
        self.update_time_curves()

    def set_mtstep(self):
        if False:
            while True:
                i = 10
        self.mtstep = not self.mtstep
        if not self.mfoverlay:
            self.mttaps = False
            self.mtimpulse = False
        self.update_step_curves()

    def set_mtimpulse(self):
        if False:
            while True:
                i = 10
        self.mtimpulse = not self.mtimpulse
        if not self.mfoverlay:
            self.mttaps = False
            self.mtstep = False
        self.update_imp_curves()

    def set_gdelay(self):
        if False:
            print('Hello World!')
        if self.gui.checkGdelay.checkState() == 0:
            self.gui.tabGroup.removeTab(self.gui.tabGroup.indexOf(self.gui.groupTab))
        else:
            self.gui.tabGroup.addTab(self.gui.groupTab, _fromUtf8('Group Delay'))
            self.update_freq_curves()

    def set_actgdelay(self):
        if False:
            print('Hello World!')
        if self.gui.actionGroup_Delay.isChecked() == 0:
            self.gui.tabGroup.removeTab(self.gui.tabGroup.indexOf(self.gui.groupTab))
        else:
            self.gui.tabGroup.addTab(self.gui.groupTab, _fromUtf8('Group Delay'))
            self.update_freq_curves()

    def set_phase(self):
        if False:
            print('Hello World!')
        if self.gui.checkPhase.checkState() == 0:
            self.gui.tabGroup.removeTab(self.gui.tabGroup.indexOf(self.gui.phaseTab))
        else:
            self.gui.tabGroup.addTab(self.gui.phaseTab, _fromUtf8('Phase Response'))
            self.update_freq_curves()

    def set_actphase(self):
        if False:
            print('Hello World!')
        if self.gui.actionPhase_Respone.isChecked() == 0:
            self.gui.tabGroup.removeTab(self.gui.tabGroup.indexOf(self.gui.phaseTab))
        else:
            self.gui.tabGroup.addTab(self.gui.phaseTab, _fromUtf8('Phase Response'))
            self.update_freq_curves()

    def set_fcoeff(self):
        if False:
            for i in range(10):
                print('nop')
        if self.gui.checkFcoeff.checkState() == 0:
            self.gui.tabGroup.removeTab(self.gui.tabGroup.indexOf(self.gui.fcTab))
        else:
            self.gui.tabGroup.addTab(self.gui.fcTab, _fromUtf8('Filter Coefficients'))
            self.update_fcoeff()

    def set_actfcoeff(self):
        if False:
            print('Hello World!')
        if self.gui.actionFilter_Coefficients.isChecked() == 0:
            self.gui.tabGroup.removeTab(self.gui.tabGroup.indexOf(self.gui.fcTab))
        else:
            self.gui.tabGroup.addTab(self.gui.fcTab, _fromUtf8('Filter Coefficients'))
            self.update_fcoeff()

    def set_band(self):
        if False:
            while True:
                i = 10
        if self.gui.checkBand.checkState() == 0:
            self.gui.filterspecView.removeTab(self.gui.filterspecView.indexOf(self.gui.bandDiagram))
        else:
            self.gui.filterspecView.addTab(self.gui.bandDiagram, _fromUtf8('Band Diagram'))

    def set_actband(self):
        if False:
            for i in range(10):
                print('nop')
        if self.gui.actionBand_Diagram.isChecked() == 0:
            self.gui.filterspecView.removeTab(self.gui.filterspecView.indexOf(self.gui.bandDiagram))
        else:
            self.gui.filterspecView.addTab(self.gui.bandDiagram, _fromUtf8('Band Diagram'))

    def set_pzplot(self):
        if False:
            print('Hello World!')
        if self.gui.checkPzplot.checkState() == 0:
            self.gui.filterspecView.removeTab(self.gui.filterspecView.indexOf(self.gui.poleZero))
        else:
            self.gui.filterspecView.addTab(self.gui.poleZero, _fromUtf8('Pole-Zero Plot'))

    def set_actpzplot(self):
        if False:
            print('Hello World!')
        if self.gui.actionPole_Zero_Plot_2.isChecked() == 0:
            self.gui.filterspecView.removeTab(self.gui.filterspecView.indexOf(self.gui.poleZero))
        else:
            self.gui.filterspecView.addTab(self.gui.poleZero, _fromUtf8('Pole-Zero Plot'))

    def set_pdelay(self):
        if False:
            while True:
                i = 10
        if self.gui.checkPzplot.checkState() == 0:
            self.gui.tabGroup.removeTab(self.gui.tabGroup.indexOf(self.gui.pdelayTab))
        else:
            self.gui.tabGroup.addTab(self.gui.pdelayTab, _fromUtf8('Phase Delay'))

    def set_actpdelay(self):
        if False:
            print('Hello World!')
        if self.gui.actionPhase_Delay.isChecked() == 0:
            self.gui.tabGroup.removeTab(self.gui.tabGroup.indexOf(self.gui.pdelayTab))
        else:
            self.gui.tabGroup.addTab(self.gui.pdelayTab, _fromUtf8('Phase Delay'))

    def set_impres(self):
        if False:
            i = 10
            return i + 15
        if self.gui.checkImpulse.checkState() == 0:
            self.gui.tabGroup.removeTab(self.gui.tabGroup.indexOf(self.gui.impresTab))
        else:
            self.gui.tabGroup.addTab(self.gui.impresTab, _fromUtf8('Impulse Response'))

    def set_actimpres(self):
        if False:
            i = 10
            return i + 15
        if self.gui.actionImpulse_Response.isChecked() == 0:
            self.gui.tabGroup.removeTab(self.gui.tabGroup.indexOf(self.gui.impresTab))
        else:
            self.gui.tabGroup.addTab(self.gui.impresTab, _fromUtf8('Impulse Response'))

    def set_stepres(self):
        if False:
            i = 10
            return i + 15
        if self.gui.checkStep.checkState() == 0:
            self.gui.tabGroup.removeTab(self.gui.tabGroup.indexOf(self.gui.stepresTab))
        else:
            self.gui.tabGroup.addTab(self.gui.stepresTab, _fromUtf8('Step Response'))

    def set_actstepres(self):
        if False:
            for i in range(10):
                print('nop')
        if self.gui.actionStep_Response.isChecked() == 0:
            self.gui.tabGroup.removeTab(self.gui.tabGroup.indexOf(self.gui.stepresTab))
        else:
            self.gui.tabGroup.addTab(self.gui.stepresTab, _fromUtf8('Step Response'))

    def populate_bandview(self, fitems):
        if False:
            while True:
                i = 10
        for item in fitems:
            if item.isWidgetType():
                self.scene.addWidget(item)
            else:
                self.scene.addItem(item)

    def remove_bandview(self):
        if False:
            i = 10
            return i + 15
        for item in list(self.scene.items()):
            self.scene.removeItem(item)

    def set_fatten(self, atten):
        if False:
            i = 10
            return i + 15
        ftype = self.gui.filterTypeComboBox.currentText()
        if ftype == 'Low Pass':
            (boxatten, r) = getfloat(self.gui.lpfStopBandAttenEdit.text())
            self.gui.lpfStopBandAttenEdit.setText(str(atten + boxatten))
        if ftype == 'High Pass':
            (boxatten, r) = getfloat(self.gui.hpfStopBandAttenEdit.text())
            self.gui.hpfStopBandAttenEdit.setText(str(atten + boxatten))
        if ftype == 'Band Pass':
            (boxatten, r) = getfloat(self.gui.bpfStopBandAttenEdit.text())
            self.gui.bpfStopBandAttenEdit.setText(str(atten + boxatten))
        if ftype == 'Band Notch':
            (boxatten, r) = getfloat(self.gui.bnfStopBandAttenEdit.text())
            self.gui.bnfStopBandAttenEdit.setText(str(atten + boxatten))
        if ftype == 'Complex Band Pass':
            (boxatten, r) = getfloat(self.gui.bpfStopBandAttenEdit.text())
            self.gui.bpfStopBandAttenEdit.setText(str(atten + boxatten))

    def set_curvetaps(self, zeros_poles):
        if False:
            return 10
        (zr, pl) = zeros_poles
        if self.iir:
            self.z = zr
            self.p = pl
            self.iir_plot_all(self.z, self.p, self.k)
            self.gui.mpzPlot.insertZeros(zr)
            self.gui.mpzPlot.insertPoles(pl)
            self.update_fcoeff()
            if self.callback:
                retobj = ApiObject()
                retobj.update_all('iir', self.params, (self.b, self.a), 1)
                self.callback(retobj)
        else:
            hz = poly1d(zr, r=1)
            self.taps = hz.c * self.taps[0]
            self.draw_plots(self.taps, self.params)
            self.update_fcoeff()
            zeros = self.get_zeros()
            poles = self.get_poles()
            self.gui.mpzPlot.insertZeros(zeros)
            self.gui.mpzPlot.insertPoles(poles)
            self.gui.nTapsEdit.setText(str(self.taps.size))
            if self.callback:
                retobj = ApiObject()
                retobj.update_all('fir', self.params, self.taps, 1)
                self.callback(retobj)

    def set_mcurvetaps(self, zeros_poles):
        if False:
            while True:
                i = 10
        (zr, pl) = zeros_poles
        if self.iir:
            self.z = zr
            self.p = pl
            self.iir_plot_all(self.z, self.p, self.k)
            self.gui.pzPlot.insertZeros(zr)
            self.gui.pzPlot.insertPoles(pl)
            self.update_fcoeff()
            if self.callback:
                retobj = ApiObject()
                retobj.update_all('iir', self.params, (self.b, self.a), 1)
                self.callback(retobj)
        else:
            hz = poly1d(zr, r=1)
            self.taps = hz.c * self.taps[0]
            if self.gridview:
                self.update_fft(self.taps, self.params)
                self.set_mfmagresponse()
                self.set_mttaps()
            else:
                self.draw_plots(self.taps, self.params)
            self.update_fcoeff()
            zeros = self.get_zeros()
            poles = self.get_poles()
            self.gui.pzPlot.insertZeros(zeros)
            self.gui.pzPlot.insertPoles(poles)
            self.gui.nTapsEdit.setText(str(self.taps.size))
            if self.callback:
                retobj = ApiObject()
                retobj.update_all('fir', self.params, self.taps, 1)
                self.callback(retobj)

    def set_statusbar(self, point):
        if False:
            i = 10
            return i + 15
        (x, y) = point
        if x == None:
            self.gui.pzstatusBar.showMessage('')
        else:
            self.gui.pzstatusBar.showMessage('X: ' + str(x) + '  Y: ' + str(y))

    def set_mstatusbar(self, point):
        if False:
            for i in range(10):
                print('nop')
        (x, y) = point
        if x == None:
            self.gui.mpzstatusBar.showMessage('')
        else:
            self.gui.mpzstatusBar.showMessage('X: ' + str(x) + '  Y: ' + str(y))

    def get_zeros(self):
        if False:
            for i in range(10):
                print('nop')
        hz = poly1d(self.taps, r=0)
        return hz.r

    def get_poles(self):
        if False:
            print('Hello World!')
        if len(self.taps):
            hp = zeros(len(self.taps) - 1, complex)
            return hp
        else:
            return []

    def impulse_response(self, b, a=1):
        if False:
            while True:
                i = 10
        length = len(b)
        if self.iir:
            length = 50
        impulse = np.repeat(0.0, length)
        impulse[0] = 1.0
        x = np.arange(0, length)
        response = signal.lfilter(b, a, impulse)
        return response

    def step_response(self, b, a=1):
        if False:
            print('Hello World!')
        length = len(b)
        if self.iir:
            length = 50
        impulse = np.repeat(0.0, length)
        impulse[0] = 1.0
        x = np.arange(0, length)
        response = signal.lfilter(b, a, impulse)
        step = np.cumsum(response)
        return step

    def update_fcoeff(self):
        if False:
            while True:
                i = 10
        fcoeff = ''
        if self.iir:
            fcoeff = 'b = ' + ','.join((str(e) for e in self.b)) + '\na = ' + ','.join((str(e) for e in self.a))
        else:
            fcoeff = 'taps = ' + ','.join((str(e) for e in self.taps))
        self.gui.filterCoeff.setText(fcoeff)
        self.gui.mfilterCoeff.setText(fcoeff)

    def action_save_dialog(self):
        if False:
            return 10
        file_dialog_output = QtWidgets.QFileDialog.getSaveFileName(self, 'Save CSV Filter File', '.', '')
        filename = file_dialog_output[0]
        try:
            handle = open(filename, 'w')
        except IOError:
            reply = QtWidgets.QMessageBox.information(self, 'File Name', 'Could not save to file: %s' % filename, QtWidgets.QMessageBox.Ok)
            return
        csvhandle = csv.writer(handle, delimiter=',')
        if self.iir:
            csvhandle.writerow(['restype', 'iir'])
        else:
            csvhandle.writerow(['restype', 'fir'])
        for k in list(self.params.keys()):
            csvhandle.writerow([k, self.params[k]])
        if self.iir:
            csvhandle.writerow(['b'] + list(self.b))
            csvhandle.writerow(['a'] + list(self.a))
        else:
            csvhandle.writerow(['taps'] + list(self.taps))
        handle.close()
        self.gui.action_save.setEnabled(False)
        for window in self.plots.values():
            window.drop_plotdata()
        self.gui.filterCoeff.setText('')
        self.gui.mfilterCoeff.setText('')
        self.gui.pzPlot.clear()
        self.replot_all()

    def action_open_dialog(self):
        if False:
            print('Hello World!')
        file_dialog_output = QtWidgets.QFileDialog.getOpenFileName(self, 'Open CSV Filter File', '.', '')
        if len(file_dialog_output) == 0:
            return
        filename = file_dialog_output[0]
        try:
            handle = open(filename, 'r')
        except IOError:
            reply = QtWidgets.QMessageBox.information(self, 'File Name', 'Could not open file: %s' % filename, QtWidgets.QMessageBox.Ok)
            return
        csvhandle = csv.reader(handle, delimiter=',')
        b_a = {}
        taps = []
        params = {}
        for row in csvhandle:
            if row[0] == 'restype':
                restype = row[1]
            elif row[0] == 'taps':
                testcpx = re.findall('[+-]?\\d+\\.*\\d*[Ee]?[-+]?\\d+j', row[1])
                if len(testcpx) > 0:
                    taps = [complex(r) for r in row[1:]]
                else:
                    taps = [float(r) for r in row[1:]]
            elif row[0] == 'b' or row[0] == 'a':
                testcpx = re.findall('[+-]?\\d+\\.*\\d*[Ee]?[-+]?\\d+j', row[1])
                if len(testcpx) > 0:
                    b_a[row[0]] = [complex(r) for r in row[1:]]
                else:
                    b_a[row[0]] = [float(r) for r in row[1:]]
            else:
                testcpx = re.findall('[+-]?\\d+\\.*\\d*[Ee]?[-+]?\\d+j', row[1])
                if len(testcpx) > 0:
                    params[row[0]] = complex(row[1])
                else:
                    try:
                        params[row[0]] = float(row[1])
                    except ValueError:
                        params[row[0]] = row[1]
        handle.close()
        if restype == 'fir':
            self.iir = False
            self.gui.fselectComboBox.setCurrentIndex(0)
            self.draw_plots(taps, params)
            zeros = self.get_zeros()
            poles = self.get_poles()
            self.gui.pzPlot.insertZeros(zeros)
            self.gui.pzPlot.insertPoles(poles)
            self.gui.mpzPlot.insertZeros(zeros)
            self.gui.mpzPlot.insertPoles(poles)
            self.gui.sampleRateEdit.setText(str(params['fs']))
            self.gui.filterGainEdit.setText(str(params['gain']))
            if params['filttype'] == 'lpf':
                self.gui.filterTypeComboBox.setCurrentIndex(0)
                self.gui.filterDesignTypeComboBox.setCurrentIndex(int(params['wintype']))
                self.gui.endofLpfPassBandEdit.setText(str(params['pbend']))
                self.gui.startofLpfStopBandEdit.setText(str(params['sbstart']))
                self.gui.lpfStopBandAttenEdit.setText(str(params['atten']))
                if params['wintype'] == self.EQUIRIPPLE_FILT:
                    self.gui.lpfPassBandRippleEdit.setText(str(params['ripple']))
            elif params['filttype'] == 'bpf':
                self.gui.filterTypeComboBox.setCurrentIndex(1)
                self.gui.filterDesignTypeComboBox.setCurrentIndex(int(params['wintype']))
                self.gui.startofBpfPassBandEdit.setText(str(params['pbstart']))
                self.gui.endofBpfPassBandEdit.setText(str(params['pbend']))
                self.gui.bpfTransitionEdit.setText(str(params['tb']))
                self.gui.bpfStopBandAttenEdit.setText(str(params['atten']))
                if params['wintype'] == self.EQUIRIPPLE_FILT:
                    self.gui.bpfPassBandRippleEdit.setText(str(params['ripple']))
            elif params['filttype'] == 'cbpf':
                self.gui.filterTypeComboBox.setCurrentIndex(2)
                self.gui.filterDesignTypeComboBox.setCurrentIndex(int(params['wintype']))
                self.gui.startofBpfPassBandEdit.setText(str(params['pbstart']))
                self.gui.endofBpfPassBandEdit.setText(str(params['pbend']))
                self.gui.bpfTransitionEdit.setText(str(params['tb']))
                self.gui.bpfStopBandAttenEdit.setText(str(params['atten']))
                if params['wintype'] == self.EQUIRIPPLE_FILT:
                    self.gui.bpfPassBandRippleEdit.setText(str(params['ripple']))
            elif params['filttype'] == 'bnf':
                self.gui.filterTypeComboBox.setCurrentIndex(3)
                self.gui.filterDesignTypeComboBox.setCurrentIndex(int(params['wintype']))
                self.gui.startofBnfStopBandEdit.setText(str(params['sbstart']))
                self.gui.endofBnfStopBandEdit.setText(str(params['sbend']))
                self.gui.bnfTransitionEdit.setText(str(params['tb']))
                self.gui.bnfStopBandAttenEdit.setText(str(params['atten']))
                if params['wintype'] == self.EQUIRIPPLE_FILT:
                    self.gui.bnfPassBandRippleEdit.setText(str(params['ripple']))
            elif params['filttype'] == 'hpf':
                self.gui.filterTypeComboBox.setCurrentIndex(4)
                self.gui.filterDesignTypeComboBox.setCurrentIndex(int(params['wintype']))
                self.gui.endofHpfStopBandEdit.setText(str(params['sbend']))
                self.gui.startofHpfPassBandEdit.setText(str(params['pbstart']))
                self.gui.hpfStopBandAttenEdit.setText(str(params['atten']))
                if params['wintype'] == self.EQUIRIPPLE_FILT:
                    self.gui.hpfPassBandRippleEdit.setText(str(params['ripple']))
            elif params['filttype'] == 'rrc':
                self.gui.filterTypeComboBox.setCurrentIndex(5)
                self.gui.filterDesignTypeComboBox.setCurrentIndex(int(params['wintype']))
                self.gui.rrcSymbolRateEdit.setText(str(params['srate']))
                self.gui.rrcAlphaEdit.setText(str(params['alpha']))
                self.gui.rrcNumTapsEdit.setText(str(params['ntaps']))
            elif params['filttype'] == 'gaus':
                self.gui.filterTypeComboBox.setCurrentIndex(6)
                self.gui.filterDesignTypeComboBox.setCurrentIndex(int(params['wintype']))
                self.gui.gausSymbolRateEdit.setText(str(params['srate']))
                self.gui.gausBTEdit.setText(str(params['bt']))
                self.gui.gausNumTapsEdit.setText(str(params['ntaps']))
        else:
            self.iir = True
            (self.b, self.a) = (b_a['b'], b_a['a'])
            (self.z, self.p, self.k) = signal.tf2zpk(self.b, self.a)
            self.gui.pzPlot.insertZeros(self.z)
            self.gui.pzPlot.insertPoles(self.p)
            self.gui.mpzPlot.insertZeros(self.z)
            self.gui.mpzPlot.insertPoles(self.p)
            self.iir_plot_all(self.z, self.p, self.k)
            self.update_fcoeff()
            self.gui.nTapsEdit.setText('-')
            self.params = params
            iirft = {'ellip': 0, 'butter': 1, 'cheby1': 2, 'cheby2': 3, 'bessel': 4}
            paramtype = {'analog': 1, 'digital': 0}
            bandpos = {'lpf': 0, 'bpf': 1, 'bnf': 2, 'hpf': 3}
            iirboxes = {'lpf': [self.gui.iirendofLpfPassBandEdit, self.gui.iirstartofLpfStopBandEdit, self.gui.iirLpfPassBandAttenEdit, self.gui.iirLpfStopBandRippleEdit], 'hpf': [self.gui.iirstartofHpfPassBandEdit, self.gui.iirendofHpfStopBandEdit, self.gui.iirHpfPassBandAttenEdit, self.gui.iirHpfStopBandRippleEdit], 'bpf': [self.gui.iirstartofBpfPassBandEdit, self.gui.iirendofBpfPassBandEdit, self.gui.iirendofBpfStopBandEdit1, self.gui.iirstartofBpfStopBandEdit2, self.gui.iirBpfPassBandAttenEdit, self.gui.iirBpfStopBandRippleEdit], 'bnf': [self.gui.iirendofBsfPassBandEdit1, self.gui.iirstartofBsfPassBandEdit2, self.gui.iirstartofBsfStopBandEdit, self.gui.iirendofBsfStopBandEdit, self.gui.iirBsfPassBandAttenEdit, self.gui.iirBsfStopBandRippleEdit]}
            self.gui.fselectComboBox.setCurrentIndex(1)
            self.gui.iirfilterTypeComboBox.setCurrentIndex(iirft[params['filttype']])
            self.gui.iirfilterBandComboBox.setCurrentIndex(bandpos[params['bandtype']])
            if params['filttype'] == 'bessel':
                critfreq = [float(x) for x in params['critfreq'][1:-1].split(',')]
                self.gui.besselordEdit.setText(str(params['filtord']))
                self.gui.iirbesselcritEdit1.setText(str(critfreq[0]))
                self.gui.iirbesselcritEdit2.setText(str(critfreq[1]))
            else:
                self.gui.adComboBox.setCurrentIndex(paramtype[params['paramtype']])
                if len(iirboxes[params['bandtype']]) == 4:
                    sdata = [params['pbedge'], params['sbedge'], params['gpass'], params['gstop']]
                else:
                    pbedge = list(map(float, params['pbedge'][1:-1].split(',')))
                    sbedge = list(map(float, params['sbedge'][1:-1].split(',')))
                    sdata = [pbedge[0], pbedge[1], sbedge[0], sbedge[1], params['gpass'], params['gstop']]
                cboxes = iirboxes[params['bandtype']]
                for i in range(len(cboxes)):
                    cboxes[i].setText(str(sdata[i]))

    def draw_plots(self, taps, params):
        if False:
            return 10
        self.params = params
        self.taps = np.array(taps)
        if self.params:
            self.get_fft(self.params['fs'], self.taps, self.nfftpts)
            self.update_time_curves()
            self.update_freq_curves()
            self.update_phase_curves()
            self.update_group_curves()
            self.update_pdelay_curves()
            self.update_step_curves()
            self.update_imp_curves()
        self.gui.nTapsEdit.setText(str(self.taps.size))

def setup_options():
    if False:
        i = 10
        return i + 15
    usage = '%prog: [options] (input_filename)'
    description = ''
    parser = OptionParser(conflict_handler='resolve', usage=usage, description=description)
    return parser

def launch(args, callback=None, restype=''):
    if False:
        for i in range(10):
            print('nop')
    parser = setup_options()
    (options, args) = parser.parse_args()
    if callback is None:
        app = Qt.QApplication(args)
        gplt = gr_plot_filter(options, callback, restype)
        app.exec_()
        if gplt.iir:
            retobj = ApiObject()
            retobj.update_all('iir', gplt.params, (gplt.b, gplt.a), 1)
            return retobj
        else:
            retobj = ApiObject()
            retobj.update_all('fir', gplt.params, gplt.taps, 1)
            return retobj
    else:
        gplt = gr_plot_filter(options, callback, restype)
        return gplt

def main(args):
    if False:
        i = 10
        return i + 15
    parser = setup_options()
    (options, args) = parser.parse_args()
    app = Qt.QApplication(args)
    gplt = gr_plot_filter(options)
    app.exec_()
    app.deleteLater()
    sys.exit()
if __name__ == '__main__':
    main(sys.argv)