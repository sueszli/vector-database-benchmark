import sys, os, time
import sys, os
sys.path.append(os.path.abspath('..'))
from PyQt4 import QtGui, QtCore, uic
import threading
import tempfile
from core.basemaps import GRIDS, SOURCES, MapService, BBoxRequest, BBoxRequestMZ
from core.lib import shapefile
from core.proj import reprojPts
from xml.etree import ElementTree as etree
import re
(mainForm, mainBase) = uic.loadUiType('QtMapServiceClient.ui')
projSysLst = {2154: 'Lambert 93', 3942: 'Lambert CC42', 3943: 'Lambert CC43', 3944: 'Lambert CC44', 3945: 'Lambert CC45', 3946: 'Lambert CC46', 3947: 'Lambert CC47', 3948: 'Lambert CC48', 3949: 'Lambert CC49', 3950: 'Lambert CC50'}

def getShpExtent(pathShp):
    if False:
        print('Hello World!')
    shp = shapefile.Reader(pathShp)
    shapes = shp.shapes()
    if len(shapes) != 1:
        return
    else:
        extent = shapes[0].bbox
        return extent

def getKmlExtent(kmlFile, crs2):
    if False:
        for i in range(10):
            print('nop')

    def formatCoor(coorText):
        if False:
            print('Hello World!')
        coorText = coorText.strip()
        coordinates = []
        for elem in str(coorText).split(' '):
            coordinates.append(tuple(map(float, elem.split(','))))
        return coordinates

    def namespace(element):
        if False:
            i = 10
            return i + 15
        m = re.match('\\{.*\\}', element.tag)
        return m.group(0) if m else ''
    root = etree.parse(kmlFile).getroot()
    ns = namespace(root)
    polygons = []
    for poly in root.iter(ns + 'Polygon'):
        for attributes in poly.iter(ns + 'coordinates'):
            polygons.append(formatCoor(attributes.text))
    if len(polygons) != 1:
        return
    else:
        pts = polygons[0]
        pts = reprojPts(4326, crs2, pts)
        xmin = min([pt[0] for pt in pts])
        ymin = min([pt[1] for pt in pts])
        xmax = max([pt[0] for pt in pts])
        ymax = max([pt[1] for pt in pts])
        extent = [xmin, ymin, xmax, ymax]
        return list(map(round, extent))

class QtMapServiceClient(QtGui.QMainWindow, mainForm):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        QtGui.QMainWindow.__init__(self)
        self.setupUi(self)
        for (k, v) in SOURCES.items():
            self.cbProvider.addItem(v['name'], k)
        self.extent = None
        self.inCacheFolder.setText(tempfile.gettempdir())
        self.btCacheFolder.clicked.connect(self.setCacheFolder)
        self.btBrowseOutFolder.clicked.connect(self.setInOutFolder)
        self.btOkMosaic.clicked.connect(self.uiDoProcess)
        self.btCancel.clicked.connect(self.uiDoCancelThread)
        self.btExtentShp.clicked.connect(self.uiDoReadShpExtent)
        self.cbProvider.currentIndexChanged.connect(self.uiDoUpdateProvider)
        self.cbLayer.currentIndexChanged.connect(self.uiDoUpdateScales)
        self.cbZoom.currentIndexChanged.connect(self.uiDoUpdateRes)
        self.chkJPG.stateChanged.connect(self.uiUpdateMaskOption)
        self.chkSeedCache.stateChanged.connect(self.uiUpdateSeedOption)
        self.uiDoUpdateProvider()
        self.inVectorFile.setText('*.kml *.shp...')

    @property
    def provider(self):
        if False:
            while True:
                i = 10
        k = self.cbProvider.itemData(self.cbProvider.currentIndex())
        cacheFolder = str(self.inCacheFolder.text())
        return MapService(k, cacheFolder)

    @property
    def layer(self):
        if False:
            i = 10
            return i + 15
        return self.cbLayer.itemData(self.cbLayer.currentIndex())

    @property
    def outProj(self):
        if False:
            for i in range(10):
                print('nop')
        return self.cbOutProj.itemData(self.cbOutProj.currentIndex())

    @property
    def zoom(self):
        if False:
            while True:
                i = 10
        z = self.cbZoom.itemData(self.cbZoom.currentIndex())
        if z is not None:
            return int(z)

    @property
    def rq(self):
        if False:
            print('Hello World!')
        if self.extent is not None and self.zoom is not None:
            rq = self.provider.srcTms.bboxRequest(self.extent, self.zoom)
            return rq

    def uiUpdateMaskOption(self):
        if False:
            for i in range(10):
                print('nop')
        if self.chkJPG.isChecked():
            self.chkMask.setEnabled(True)
        else:
            self.chkMask.setEnabled(False)

    def uiUpdateSeedOption(self):
        if False:
            print('Hello World!')
        if self.chkSeedCache.isChecked():
            self.chkRecurseUpZoomLevels.setEnabled(True)
            self.chkReproj.setEnabled(False)
            self.cbOutProj.setEnabled(False)
            self.chkBuildOverview.setEnabled(False)
            self.chkJPG.setEnabled(False)
            self.chkMask.setEnabled(False)
            self.chkBigtiff.setEnabled(False)
            self.inName.setEnabled(False)
            self.inOutFolder.setEnabled(False)
            self.btBrowseOutFolder.setEnabled(False)
        else:
            self.chkRecurseUpZoomLevels.setEnabled(False)
            self.chkReproj.setEnabled(True)
            self.cbOutProj.setEnabled(True)
            self.chkBuildOverview.setEnabled(True)
            self.chkJPG.setEnabled(True)
            self.chkMask.setEnabled(True)
            self.chkBigtiff.setEnabled(True)
            self.inName.setEnabled(True)
            self.inOutFolder.setEnabled(True)
            self.btBrowseOutFolder.setEnabled(True)

    def uiDoUpdateProvider(self):
        if False:
            while True:
                i = 10
        'Triggered when cbProvider idx change'
        self.cbLayer.clear()
        self.cbOutProj.clear()
        for (layerKey, layer) in self.provider.layers.items():
            self.cbLayer.addItem(layer.name, layerKey)
        for (k, v) in projSysLst.items():
            self.cbOutProj.addItem(v, k)
        self.cbOutProj.setCurrentIndex(self.cbOutProj.findData(2154))
        self.updateExtent()

    def uiDoUpdateScales(self):
        if False:
            print('Hello World!')
        'Triggered when cbLayer idx change'
        if self.layer is not None:
            lay = self.provider.layers[self.layer]
            self.cbZoom.clear()
            for z in range(lay.zmin, lay.zmax):
                self.cbZoom.addItem(str(z), str(z))

    def uiDoUpdateRes(self, zoomLevel):
        if False:
            for i in range(10):
                print('nop')
        'Triggered when cbZoom idx change'
        if self.rq is not None:
            self.lbRes.setText(str(round(self.rq.res, 2)) + ' m/px')
            self.uiDoRequestInfos()

    def uiDoReadShpExtent(self):
        if False:
            return 10
        path = str(self.setOpenFileName('Shapefile (*.shp *.kml)'))
        self.inVectorFile.setText(path)
        self.updateExtent()

    def updateExtent(self):
        if False:
            print('Hello World!')
        path = self.inVectorFile.text()
        if not os.path.exists(path):
            pass
        else:
            ext = path[-3:]
            if ext == 'shp':
                self.extent = getShpExtent(path)
            elif ext == 'kml':
                self.extent = getKmlExtent(path, self.provider.srcTms.CRS)
            if not self.extent:
                QtGui.QMessageBox.information(self, 'Cannot read vector extent file', 'This file must contains only one polygon')
                return
            self.uiDoRequestInfos()
            self.inVectorFile.setText(path)

    def uiDoRequestInfos(self):
        if False:
            print('Hello World!')
        if self.rq is not None:
            tileSize = self.rq.tileSize
            res = self.rq.res
            (cols, rows) = (self.rq.nbTilesX, self.rq.nbTilesY)
            n = self.rq.nbTiles
            (xmin, ymin, xmax, ymax) = self.extent
            dstX = xmax - xmin
            dstY = ymax - ymin
            txtEmprise = str(round(dstX)) + ' x ' + str(round(dstY)) + ' m'
            nbPx = int(cols * tileSize * rows * tileSize)
            if nbPx > 1000000:
                txtNbPx = str(int(nbPx / 1000000)) + ' Mpix'
            else:
                txtNbPx = str(nbPx) + ' pix'
            txtNbTiles = str(n) + ' tile(s)'
            resultStr = txtNbTiles + ' (' + str(cols) + 'x' + str(rows) + ') - ' + txtNbPx + ' - ' + txtEmprise
            self.requestInfos.setText(resultStr)

    def uiDoProcess(self):
        if False:
            print('Hello World!')
        outFolder = str(self.inOutFolder.text())
        nameTemplate = str(self.inName.text())
        cacheFolder = str(self.inCacheFolder.text())
        if not self.chkSeedCache:
            if not os.path.exists(outFolder):
                QtGui.QMessageBox.information(self, 'Error', 'Output folder does not exists')
                return
            if not nameTemplate:
                QtGui.QMessageBox.information(self, 'Error', 'Basename is not defined')
                return
        if not os.path.exists(cacheFolder):
            QtGui.QMessageBox.information(self, 'Error', 'Cache folder does not exists')
            return
        reproj = self.chkReproj.isChecked()
        outProj = self.cbOutProj.itemData(self.cbOutProj.currentIndex())
        reprojOptions = (reproj, outProj)
        buildOvv = self.chkBuildOverview.isChecked()
        jpgInTiff = self.chkJPG.isChecked()
        mask = self.chkMask.isChecked()
        bigTiff = self.chkBigtiff.isChecked()
        self.btOkMosaic.setEnabled(False)
        if self.chkReproj:
            outCRS = self.outProj
        else:
            outCRS = None
        outFile = outFolder + os.sep + nameTemplate + '.tif'
        seedOnly = self.chkSeedCache.isChecked()
        recurseUpZoomLevels = self.chkRecurseUpZoomLevels.isChecked()
        self.thread = DownloadTiles(self.provider, self.layer, self.extent, self.zoom, outFile, outCRS, seedOnly, recurseUpZoomLevels)
        self.thread.finished.connect(self.uiProcessFinished)
        self.thread.terminated.connect(self.uiProcessFinished)
        self.thread.updateBar1.connect(self.uiDoUpdateBar1)
        self.thread.configBar1.connect(self.uiDoConfigBar1)
        self.thread.processInfo.connect(self.updateProcessInfo)
        self.thread.start()

    def uiProcessFinished(self):
        if False:
            return 10
        self.updateUi()
        QtGui.QMessageBox.information(self, 'Info', 'Finished')

    def uiDoCancelThread(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            self.thread.cancel()
        except:
            pass

    def uiSendQuestion(self, titre, msg):
        if False:
            i = 10
            return i + 15
        choice = QtGui.QMessageBox.question(self, titre, msg, QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)
        if choice == QtGui.QMessageBox.Yes:
            return True
        else:
            return False

    def updateUi(self):
        if False:
            i = 10
            return i + 15
        self.btOkMosaic.setEnabled(True)

    def uiDoUpdateBar1(self, num):
        if False:
            print('Hello World!')
        self.pBar1.setValue(num)

    def uiDoConfigBar1(self, nb):
        if False:
            print('Hello World!')
        self.pBar1.setMinimum(0)
        self.pBar1.setMaximum(nb)

    def updateProcessInfo(self, txt):
        if False:
            print('Hello World!')
        self.processInfo.setText(txt)

    def setInOutFolder(self):
        if False:
            print('Hello World!')
        path = self.setExistingDirectory()
        if path:
            self.inOutFolder.setText(path)

    def setCacheFolder(self):
        if False:
            while True:
                i = 10
        path = self.setExistingDirectory()
        if path:
            self.inCacheFolder.setText(path)

    def setInFolder(self):
        if False:
            while True:
                i = 10
        path = self.setExistingDirectory()
        if path:
            self.inVectorFile.setText(path)

    def setOpenFileName(self, filtre):
        if False:
            while True:
                i = 10
        fileName = QtGui.QFileDialog.getOpenFileName(self, 'Select file', QtCore.QDir.rootPath(), filtre)
        return QtCore.QDir.toNativeSeparators(fileName)

    def setExistingDirectory(self):
        if False:
            print('Hello World!')
        directory = QtGui.QFileDialog.getExistingDirectory(self, 'Select directory', QtCore.QDir.rootPath(), QtGui.QFileDialog.ShowDirsOnly)
        return QtCore.QDir.toNativeSeparators(directory)

    def setSaveFileName(self):
        if False:
            return 10
        saveFileName = QtGui.QFileDialog.getSaveFileName(self, 'Save file', QtCore.QDir.rootPath())
        return QtCore.QDir.toNativeSeparators(saveFileName)

class DownloadTiles(QtCore.QThread):
    configBar1 = QtCore.pyqtSignal(int)
    updateBar1 = QtCore.pyqtSignal(int)
    processInfo = QtCore.pyqtSignal(str)

    def __init__(self, srv, layer, extent, zoom, outFile, outCRS, seedOnly, recurseUpZoomLevels):
        if False:
            while True:
                i = 10
        QtCore.QThread.__init__(self, None)
        self.srv = srv
        self.layer = layer
        self.extent = extent
        self.outFile = outFile
        self.outCRS = outCRS
        self.seedOnly = seedOnly
        if recurseUpZoomLevels and seedOnly:
            self.zoom = list(range(self.srv.layers[self.layer].zmin, zoom + 1))
            self.rq = BBoxRequestMZ(self.srv.srcTms, self.extent, self.zoom)
            print(self.rq.nbTiles, self.srv.srcTms.bboxRequest(self.extent, zoom).nbTiles)
        else:
            self.zoom = zoom
            self.rq = self.srv.srcTms.bboxRequest(self.extent, self.zoom)

    def run(self):
        if False:
            while True:
                i = 10
        self.srv.start()
        self.configBar1.emit(self.rq.nbTiles)
        if self.seedOnly:
            thread = threading.Thread(target=self.seedCache)
        else:
            thread = threading.Thread(target=self.getImage)
        thread.start()
        while thread.isAlive():
            time.sleep(0.05)
            self.processInfo.emit(self.srv.report)
            self.updateBar1.emit(self.srv.cptTiles)
        self.srv.stop()

    def seedCache(self):
        if False:
            return 10
        self.srv.seedCache(self.layer, self.extent, self.zoom, toDstGrid=False)

    def getImage(self):
        if False:
            return 10
        self.srv.getImage(self.layer, self.extent, self.zoom, path=self.outFile, bigTiff=True, outCRS=self.outCRS, toDstGrid=False)

    def cancel(self):
        if False:
            for i in range(10):
                print('nop')
        self.srv.stop()
    '\n\t#no need for pausing because downloading tiles are saved in cache,\n\t#so restarting an aborted process will reuse existing tiles\n\tdef pause(self):\n\t\tself.srv.pause()\n\n\tdef resume(self):\n\t\tself.srv.resume()\n\t'
if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    window = QtMapServiceClient()
    window.show()
    sys.exit(app.exec_())