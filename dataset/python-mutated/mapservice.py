import logging
log = logging.getLogger(__name__)
import math
import threading
import queue
import time
import urllib.request
import imghdr
import sys, time, os
from .servicesDefs import GRIDS, SOURCES
from .gpkg import GeoPackage
from ..georaster import NpImage, GeoRef, BigTiffWriter
from ..utils import BBOX
from ..proj.reproj import reprojPt, reprojBbox, reprojImg
from ..proj.ellps import dd2meters, meters2dd
from ..proj.srs import SRS
from .. import settings
USER_AGENT = settings.user_agent
TIMEOUT = 4
MOSAIC_BKG_COLOR = (128, 128, 128, 255)
EMPTY_TILE_COLOR = (255, 192, 203, 255)
CORRUPTED_TILE_COLOR = (255, 0, 0, 255)

class TileMatrix:
    """
	Will inherit attributes from grid source definition
		"CRS" >> epsg code
		"bbox" >> (xmin, ymin, xmax, ymax)
		"bboxCRS" >> epsg code
		"tileSize"
		"originLoc" >> "NW" or SW

		"resFactor"
		"initRes" >> optional
		"nbLevels" >> optional

		or

		"resolutions"

	# Three ways to define a grid:
	# - submit a list of "resolutions" (This parameters override the others)
	# - submit "resFactor" and "initRes"
	# - submit just "resFactor" (initRes will be computed)
	"""
    defaultNbLevels = 24

    def __init__(self, gridDef):
        if False:
            return 10
        for (k, v) in gridDef.items():
            setattr(self, k, v)
        if self.bboxCRS != self.CRS:
            (lonMin, latMin, lonMax, latMax) = self.bbox
            (self.xmin, self.ymax) = self.geoToProj(lonMin, latMax)
            (self.xmax, self.ymin) = self.geoToProj(lonMax, latMin)
        else:
            (self.xmin, self.xmax) = (self.bbox[0], self.bbox[2])
            (self.ymin, self.ymax) = (self.bbox[1], self.bbox[3])
        if not hasattr(self, 'resolutions'):
            if not hasattr(self, 'resFactor'):
                self.resFactor = 2
            if not hasattr(self, 'initRes'):
                dx = abs(self.xmax - self.xmin)
                dy = abs(self.ymax - self.ymin)
                dst = max(dx, dy)
                self.initRes = dst / self.tileSize
            if not hasattr(self, 'nbLevels'):
                self.nbLevels = self.defaultNbLevels
        else:
            self.resolutions.sort(reverse=True)
            self.nbLevels = len(self.resolutions)
        if self.originLoc == 'NW':
            (self.originx, self.originy) = (self.xmin, self.ymax)
        elif self.originLoc == 'SW':
            (self.originx, self.originy) = (self.xmin, self.ymin)
        else:
            raise NotImplementedError
        self.crs = SRS(self.CRS)
        if self.crs.isGeo:
            self.units = 'degrees'
        else:
            self.units = 'meters'

    @property
    def globalbbox(self):
        if False:
            while True:
                i = 10
        return (self.xmin, self.ymin, self.xmax, self.ymax)

    def geoToProj(self, long, lat):
        if False:
            i = 10
            return i + 15
        'convert longitude latitude in decimal degrees to grid crs'
        if self.CRS == 'EPSG:4326':
            return (long, lat)
        else:
            return reprojPt(4326, self.CRS, long, lat)

    def projToGeo(self, x, y):
        if False:
            while True:
                i = 10
        'convert grid crs coords to longitude latitude in decimal degrees'
        if self.CRS == 'EPSG:4326':
            return (x, y)
        else:
            return reprojPt(self.CRS, 4326, x, y)

    def getResList(self):
        if False:
            return 10
        if hasattr(self, 'resolutions'):
            return self.resolutions
        else:
            return [self.initRes / self.resFactor ** zoom for zoom in range(self.nbLevels)]

    def getRes(self, zoom):
        if False:
            print('Hello World!')
        'Resolution (meters/pixel) for given zoom level (measured at Equator)'
        if hasattr(self, 'resolutions'):
            if zoom > len(self.resolutions):
                zoom = len(self.resolutions)
            return self.resolutions[zoom]
        else:
            return self.initRes / self.resFactor ** zoom

    def getNearestZoom(self, res, rule='closer'):
        if False:
            for i in range(10):
                print('nop')
        "\n\t\tReturn the zoom level closest to the submited resolution\n\t\trule in ['closer', 'lower', 'higher']\n\t\tlower return the previous zoom level, higher return the next\n\t\t"
        resLst = self.getResList()
        for (z1, v1) in enumerate(resLst):
            if v1 == res:
                return z1
            if z1 == len(resLst) - 1:
                return z1
            z2 = z1 + 1
            v2 = resLst[z2]
            if v2 == res:
                return z2
            if v1 > res > v2:
                if rule == 'lower':
                    return z1
                elif rule == 'higher':
                    return z2
                else:
                    d1 = v1 - res
                    d2 = res - v2
                    if d1 < d2:
                        return z1
                    else:
                        return z2

    def getPrevResFac(self, z):
        if False:
            for i in range(10):
                print('nop')
        'return res factor to previous zoom level'
        return self.getFromToResFac(z, z - 1)

    def getNextResFac(self, z):
        if False:
            i = 10
            return i + 15
        'return res factor to next zoom level'
        return self.getFromToResFac(z, z + 1)

    def getFromToResFac(self, z1, z2):
        if False:
            while True:
                i = 10
        'return res factor from z1 to z2'
        if z1 == z2:
            return 1
        if z1 < z2:
            if z2 >= self.nbLevels - 1:
                return 1
            else:
                return self.getRes(z2) / self.getRes(z1)
        elif z1 > z2:
            if z2 <= 0:
                return 1
            else:
                return self.getRes(z2) / self.getRes(z1)

    def getTileNumber(self, x, y, zoom):
        if False:
            i = 10
            return i + 15
        'Convert projeted coords to tiles number'
        res = self.getRes(zoom)
        geoTileSize = self.tileSize * res
        dx = x - self.originx
        if self.originLoc == 'NW':
            dy = self.originy - y
        else:
            dy = y - self.originy
        col = dx / geoTileSize
        row = dy / geoTileSize
        col = int(math.floor(col))
        row = int(math.floor(row))
        return (col, row)

    def getTileCoords(self, col, row, zoom):
        if False:
            i = 10
            return i + 15
        '\n\t\tConvert tiles number to projeted coords\n\t\t(top left pixel if matrix origin is NW)\n\t\t'
        res = self.getRes(zoom)
        geoTileSize = self.tileSize * res
        x = self.originx + col * geoTileSize
        if self.originLoc == 'NW':
            y = self.originy - row * geoTileSize
        else:
            y = self.originy + row * geoTileSize
            y += geoTileSize
        return (x, y)

    def getTileBbox(self, col, row, zoom):
        if False:
            print('Hello World!')
        (xmin, ymax) = self.getTileCoords(col, row, zoom)
        xmax = xmin + self.tileSize * self.getRes(zoom)
        ymin = ymax - self.tileSize * self.getRes(zoom)
        return (xmin, ymin, xmax, ymax)

    def bboxRequest(self, bbox, zoom):
        if False:
            i = 10
            return i + 15
        return BBoxRequest(self, bbox, zoom)

class BBoxRequestMZ:
    """Multiple Zoom BBox request"""

    def __init__(self, tm, bbox, zooms):
        if False:
            print('Hello World!')
        self.tm = tm
        self.bboxrequests = {}
        for z in zooms:
            self.bboxrequests[z] = BBoxRequest(tm, bbox, z)

    @property
    def tiles(self):
        if False:
            i = 10
            return i + 15
        tiles = []
        for bboxrequest in self.bboxrequests.values():
            tiles.extend(bboxrequest.tiles)
        return tiles

    @property
    def nbTiles(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.tiles)

    def __getitem__(self, z):
        if False:
            return 10
        return self.bboxrequests[z]

class BBoxRequest:

    def __init__(self, tm, bbox, zoom):
        if False:
            for i in range(10):
                print('nop')
        self.tm = tm
        self.zoom = zoom
        self.tileSize = tm.tileSize
        self.res = tm.getRes(zoom)
        (xmin, ymin, xmax, ymax) = bbox
        (self.firstCol, self.firstRow) = tm.getTileNumber(xmin, ymax, zoom)
        (xmin, ymax) = tm.getTileCoords(self.firstCol, self.firstRow, zoom)
        self.bbox = BBOX(xmin, ymin, xmax, ymax)
        self.nbTilesX = math.ceil((xmax - xmin) / (self.tileSize * self.res))
        self.nbTilesY = math.ceil((ymax - ymin) / (self.tileSize * self.res))

    @property
    def cols(self):
        if False:
            for i in range(10):
                print('nop')
        return [self.firstCol + i for i in range(self.nbTilesX)]

    @property
    def rows(self):
        if False:
            for i in range(10):
                print('nop')
        if self.tm.originLoc == 'NW':
            return [self.firstRow + i for i in range(self.nbTilesY)]
        else:
            return [self.firstRow - i for i in range(self.nbTilesY)]

    @property
    def tiles(self):
        if False:
            for i in range(10):
                print('nop')
        return [(c, r, self.zoom) for c in self.cols for r in self.rows]

    @property
    def nbTiles(self):
        if False:
            return 10
        return self.nbTilesX * self.nbTilesY

class MapService:
    """
	Represent a tile service from source

	Will inherit attributes from source definition
		name
		description
		service >> 'WMS', 'TMS' or 'WMTS'
		grid >> key identifier of the tile matrix used by this source
		matrix >> for WMTS only, name of the matrix as refered in url
		quadTree >> boolean, for TMS only. Flag if tile coords are stord through a quadkey
		layers >> a list layers with the following attributes
			urlkey
			name
			description
			format >> 'jpeg' or 'png'
			style
			zmin & zmax
		urlTemplate
		referer

	Service status code
		0 = no running tasks
		1 = getting cache (create a new db if needed)
		2 = downloading
		3 = building mosaic
		4 = reprojecting
	"""
    RESAMP_ALG = 'BL'

    def __init__(self, srckey, cacheFolder, dstGridKey=None):
        if False:
            print('Hello World!')
        self.srckey = srckey
        source = SOURCES[self.srckey]
        for (k, v) in source.items():
            setattr(self, k, v)

        class Layer:
            pass
        layersObj = {}
        for (layKey, layDict) in self.layers.items():
            lay = Layer()
            for (k, v) in layDict.items():
                setattr(lay, k, v)
            layersObj[layKey] = lay
        self.layers = layersObj
        self.srcGridKey = self.grid
        self.srcTms = TileMatrix(GRIDS[self.srcGridKey])
        self.setDstGrid(dstGridKey)
        self.cacheFolder = cacheFolder
        self.caches = {}
        self.headers = {'Accept': 'image/png,image/*;q=0.8,*/*;q=0.5', 'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.7', 'Accept-Language': 'fr,en-us,en;q=0.5', 'Proxy-Connection': 'keep-alive', 'User-Agent': USER_AGENT, 'Referer': self.referer}
        self.running = False
        self.nbTiles = 0
        self.cptTiles = 0
        self.status = 0
        self.lock = threading.RLock()

    def reportLoop(self):
        if False:
            return 10
        msg = self.report
        while self.running:
            time.sleep(0.05)
            if self.report != msg:
                sys.stdout.write('\x1b[K')
                sys.stdout.flush()
                print(self.report, end='\r')
                msg = self.report

    def start(self):
        if False:
            while True:
                i = 10
        self.running = True
        reporter = threading.Thread(target=self.reportLoop)
        reporter.setDaemon(True)
        reporter.start()

    def stop(self):
        if False:
            for i in range(10):
                print('nop')
        self.running = False

    @property
    def report(self):
        if False:
            for i in range(10):
                print('nop')
        if self.status == 0:
            return ''
        if self.status == 1:
            return 'Get cache database...'
        if self.status == 2:
            return 'Downloading... ' + str(self.cptTiles) + '/' + str(self.nbTiles)
        if self.status == 3:
            return 'Building mosaic...'
        if self.status == 4:
            return 'Reprojecting...'

    def setDstGrid(self, grdkey):
        if False:
            while True:
                i = 10
        'Set destination tile matrix'
        if grdkey is not None and grdkey != self.srcGridKey:
            self.dstGridKey = grdkey
            self.dstTms = TileMatrix(GRIDS[grdkey])
        else:
            self.dstGridKey = None
            self.dstTms = None

    def getCache(self, laykey, useDstGrid):
        if False:
            return 10
        'Return existing cache for requested layer or built it if not exists'
        if useDstGrid:
            if self.dstGridKey is not None:
                grdkey = self.dstGridKey
                tm = self.dstTms
            else:
                raise ValueError('No destination grid defined')
        else:
            grdkey = self.srcGridKey
            tm = self.srcTms
        mapKey = self.srckey + '_' + laykey + '_' + grdkey
        cache = self.caches.get(mapKey)
        if cache is None:
            dbPath = os.path.join(self.cacheFolder, mapKey + '.gpkg')
            self.caches[mapKey] = GeoPackage(dbPath, tm)
            return self.caches[mapKey]
        else:
            return cache

    def getTM(self, dstGrid=False):
        if False:
            while True:
                i = 10
        if dstGrid:
            if self.dstTms is not None:
                return self.dstTms
            else:
                raise ValueError('No destination grid defined')
        else:
            return self.srcTms

    def buildUrl(self, laykey, col, row, zoom):
        if False:
            return 10
        '\n\t\tReceive tiles coords in source tile matrix space and build request url\n\t\t'
        url = self.urlTemplate
        lay = self.layers[laykey]
        tm = self.srcTms
        if self.service == 'TMS':
            url = url.replace('{LAY}', lay.urlKey)
            if not self.quadTree:
                url = url.replace('{X}', str(col))
                url = url.replace('{Y}', str(row))
                url = url.replace('{Z}', str(zoom))
            else:
                quadkey = self.getQuadKey(col, row, zoom)
                url = url.replace('{QUADKEY}', quadkey)
        if self.service == 'WMTS':
            url = self.urlTemplate['BASE_URL']
            if url[-1] != '?':
                url += '?'
            params = ['='.join([k, v]) for (k, v) in self.urlTemplate.items() if k != 'BASE_URL']
            url += '&'.join(params)
            url = url.replace('{LAY}', lay.urlKey)
            url = url.replace('{FORMAT}', lay.format)
            url = url.replace('{STYLE}', lay.style)
            url = url.replace('{MATRIX}', self.matrix)
            url = url.replace('{X}', str(col))
            url = url.replace('{Y}', str(row))
            url = url.replace('{Z}', str(zoom))
        if self.service == 'WMS':
            url = self.urlTemplate['BASE_URL']
            if url[-1] != '?':
                url += '?'
            params = ['='.join([k, v]) for (k, v) in self.urlTemplate.items() if k != 'BASE_URL']
            url += '&'.join(params)
            url = url.replace('{LAY}', lay.urlKey)
            url = url.replace('{FORMAT}', lay.format)
            url = url.replace('{STYLE}', lay.style)
            url = url.replace('{CRS}', str(tm.CRS))
            url = url.replace('{WIDTH}', str(tm.tileSize))
            url = url.replace('{HEIGHT}', str(tm.tileSize))
            (xmin, ymax) = tm.getTileCoords(col, row, zoom)
            xmax = xmin + tm.tileSize * tm.getRes(zoom)
            ymin = ymax - tm.tileSize * tm.getRes(zoom)
            if self.urlTemplate['VERSION'] == '1.3.0' and tm.CRS == 'EPSG:4326':
                bbox = ','.join(map(str, [ymin, xmin, ymax, xmax]))
            else:
                bbox = ','.join(map(str, [xmin, ymin, xmax, ymax]))
            url = url.replace('{BBOX}', bbox)
        return url

    def getQuadKey(self, x, y, z):
        if False:
            return 10
        'Converts TMS tile coordinates to Microsoft QuadTree'
        quadKey = ''
        for i in range(z, 0, -1):
            digit = 0
            mask = 1 << i - 1
            if x & mask != 0:
                digit += 1
            if y & mask != 0:
                digit += 2
            quadKey += str(digit)
        return quadKey

    def isTileInMapsBounds(self, col, row, zoom, tm):
        if False:
            return 10
        'Test if the tile is not out of tile matrix bounds'
        (x, y) = tm.getTileCoords(col, row, zoom)
        if row < 0 or col < 0:
            return False
        elif not tm.xmin <= x < tm.xmax or not tm.ymin < y <= tm.ymax:
            return False
        else:
            return True

    def downloadTile(self, laykey, col, row, zoom):
        if False:
            return 10
        '\n\t\tDownload bytes data of requested tile in source tile matrix space\n\t\tReturn None if unable to download a valid stream\n\t\t'
        url = self.buildUrl(laykey, col, row, zoom)
        log.debug(url)
        try:
            req = urllib.request.Request(url, None, self.headers)
            handle = urllib.request.urlopen(req, timeout=TIMEOUT)
            data = handle.read()
            handle.close()
        except Exception as e:
            log.error("Can't download tile x{} y{}. Error {}".format(col, row, e))
            data = None
        if data is not None:
            format = imghdr.what(None, data)
            if format is None:
                data = None
        if data is None:
            log.debug('Invalid tile data for request {}'.format(url))
        return data

    def tileRequest(self, laykey, col, row, zoom, toDstGrid=True):
        if False:
            return 10
        '\n\t\tReturn bytes data of the requested tile or None if unable to get valid data\n\t\tTile is downloaded from map service and, if needed, reprojected to fit the destination grid\n\t\t'
        tm = self.getTM(toDstGrid)
        if not self.isTileInMapsBounds(col, row, zoom, tm):
            return None
        if not toDstGrid:
            data = self.downloadTile(laykey, col, row, zoom)
        else:
            data = self.buildDstTile(laykey, col, row, zoom)
        return data

    def buildDstTile(self, laykey, col, row, zoom):
        if False:
            for i in range(10):
                print('nop')
        'build a tile that fit the destination tile matrix'
        bbox = self.dstTms.getTileBbox(col, row, zoom)
        (xmin, ymin, xmax, ymax) = bbox
        res = self.dstTms.getRes(zoom)
        if self.dstTms.units == 'degrees' and self.srcTms.units == 'meters':
            res2 = dd2meters(res)
        elif self.srcTms.units == 'degrees' and self.dstTms.units == 'meters':
            res2 = meters2dd(res)
        else:
            res2 = res
        _zoom = self.srcTms.getNearestZoom(res2)
        _res = self.srcTms.getRes(_zoom)
        (crs1, crs2) = (self.srcTms.CRS, self.dstTms.CRS)
        try:
            _bbox = reprojBbox(crs2, crs1, bbox)
        except Exception as e:
            log.warning('Cannot reproj tile bbox - ' + str(e))
            return None
        mosaic = self.getImage(laykey, _bbox, _zoom, toDstGrid=False, nbThread=4, cpt=False)
        if mosaic is None:
            return None
        tileSize = self.dstTms.tileSize
        img = NpImage(reprojImg(crs1, crs2, mosaic.toGDAL(), out_ul=(xmin, ymax), out_size=(tileSize, tileSize), out_res=res, sqPx=True, resamplAlg=self.RESAMP_ALG))
        return img.toBLOB()

    def seedTiles(self, laykey, tiles, toDstGrid=True, nbThread=10, buffSize=5000, cpt=True):
        if False:
            while True:
                i = 10
        '\n\t\tSeed the cache by downloading the requested tiles from map service\n\t\tDownloads are performed through thread to speed up\n\n\t\tbuffSize : maximum number of tiles keeped in memory before put them in cache database\n\t\t'

        def downloading(laykey, tilesQueue, tilesData, toDstGrid):
            if False:
                for i in range(10):
                    print('nop')
            'Worker that process the queue and seed tilesData array [(x,y,z,data)]'
            while not tilesQueue.empty():
                if not self.running:
                    break
                (col, row, zoom) = tilesQueue.get()
                data = self.tileRequest(laykey, col, row, zoom, toDstGrid)
                if data is not None:
                    tilesData.put((col, row, zoom, data))
                if cpt:
                    self.cptTiles += 1
                tilesQueue.task_done()

        def finished():
            if False:
                for i in range(10):
                    print('nop')
            return not any([t.is_alive() for t in threads])

        def putInCache(tilesData, jobs, cache):
            if False:
                while True:
                    i = 10
            while True:
                if tilesData.full() or ((finished() or not self.running) and (not tilesData.empty())):
                    data = [tilesData.get() for i in range(tilesData.qsize())]
                    with self.lock:
                        cache.putTiles(data)
                if finished() and tilesData.empty():
                    break
                if not self.running:
                    break
        if cpt:
            self.nbTiles = len(tiles)
            self.cptTiles = 0
        if cpt:
            self.status = 1
        cache = self.getCache(laykey, toDstGrid)
        missing = cache.listMissingTiles(tiles)
        nMissing = len(missing)
        nExists = self.nbTiles - len(missing)
        log.debug('{} tiles requested, {} already in cache, {} remains to download'.format(self.nbTiles, nExists, nMissing))
        if cpt:
            self.cptTiles += nExists
        if cpt:
            self.status = 2
        if len(missing) > 0:
            tilesData = queue.Queue(maxsize=buffSize)
            jobs = queue.Queue()
            for tile in missing:
                jobs.put(tile)
            threads = []
            for i in range(nbThread):
                t = threading.Thread(target=downloading, args=(laykey, jobs, tilesData, toDstGrid))
                t.setDaemon(True)
                threads.append(t)
                t.start()
            seeder = threading.Thread(target=putInCache, args=(tilesData, jobs, cache))
            seeder.setDaemon(True)
            seeder.start()
            seeder.join()
            for t in threads:
                t.join()
        if cpt:
            self.status = 0
            (self.nbTiles, self.cptTiles) = (0, 0)

    def getTiles(self, laykey, tiles, toDstGrid=True, nbThread=10, cpt=True):
        if False:
            print('Hello World!')
        '\n\t\tReturn bytes data of requested tiles\n\t\tinput: [(x,y,z)] >> output: [(x,y,z,data)]\n\t\tTiles are downloaded from map service or directly pick up from cache database.\n\t\t'
        self.seedTiles(laykey, tiles, toDstGrid=toDstGrid, nbThread=10, cpt=cpt)
        cache = self.getCache(laykey, toDstGrid)
        return cache.getTiles(tiles)

    def getTile(self, laykey, col, row, zoom, toDstGrid=True):
        if False:
            return 10
        return self.getTiles(laykey, [col, row, zoom], toDstGrid)[0]

    def bboxRequest(self, bbox, zoom, dstGrid=True):
        if False:
            print('Hello World!')
        tm = self.getTM(dstGrid)
        return BBoxRequest(tm, bbox, zoom)

    def seedCache(self, laykey, bbox, zoom, toDstGrid=True, nbThread=10, buffSize=5000):
        if False:
            print('Hello World!')
        '\n\t\tSeed the cache with the tiles covering the requested bbox\n\t\t'
        tm = self.getTM(toDstGrid)
        if isinstance(zoom, list):
            rq = BBoxRequestMZ(tm, bbox, zoom)
        else:
            rq = BBoxRequest(tm, bbox, zoom)
        self.seedTiles(laykey, rq.tiles, toDstGrid=toDstGrid, nbThread=10, buffSize=5000)

    def getImage(self, laykey, bbox, zoom, path=None, bigTiff=False, outCRS=None, toDstGrid=True, nbThread=10, cpt=True):
        if False:
            return 10
        '\n\t\tBuild a mosaic of tiles covering the requested bounding box\n\t\t#laykey (str)\n\t\t#bbox\n\t\t#zoom (int)\n\t\t#path (str): if None the function will return a georeferenced NpImage object. If not None, then the resulting output will be\n\t\twriten as geotif file on disk and the function will return None\n\t\t#bigTiff (bool): if true then the raster will be writen by small part with the help of GDAL API. If false the raster will be\n\t\twriten at one, in this case all the tiles must fit in memory otherwise it will raise a memory overflow error\n\t\t#outCRS : destination CRS if a reprojection if expected (require GDAL support)\n\t\t#toDstGrid (bool) : decide if the function will seed the destination tile matrix sets for this MapService instance\n\t\t(different from the source tile matrix set)\n\t\t#nbThread (int) : nimber of threads that will be used for downloading tiles\n\t\t#cpt (bool) : define if the service must report or not tiles downloading count for this request\n\t\t'
        tm = self.getTM(toDstGrid)
        rq = BBoxRequest(tm, bbox, zoom)
        tileSize = rq.tileSize
        res = rq.res
        (cols, rows) = (rq.cols, rq.rows)
        rqTiles = rq.tiles
        self.seedCache(laykey, bbox, zoom, toDstGrid=toDstGrid, nbThread=nbThread, buffSize=5000)
        cache = self.getCache(laykey, toDstGrid)
        if not self.running:
            if cpt:
                self.status = 0
            return
        (img_w, img_h) = (len(cols) * tileSize, len(rows) * tileSize)
        (xmin, ymin, xmax, ymax) = rq.bbox
        georef = GeoRef((img_w, img_h), (res, -res), (xmin, ymax), pxCenter=False, crs=tm.crs)
        if bigTiff and path is None:
            raise ValueError('No output path defined for creating bigTiff')
        if not bigTiff:
            mosaic = NpImage.new(img_w, img_h, bkgColor=MOSAIC_BKG_COLOR, georef=georef)
            chunkSize = rq.nbTiles
        else:
            mosaic = BigTiffWriter(path, img_w, img_h, georef)
            ds = mosaic.ds
            chunkSize = 5
        for i in range(0, rq.nbTiles, chunkSize):
            chunkTiles = rqTiles[i:i + chunkSize]
            tiles = cache.getTiles(chunkTiles)
            if cpt:
                self.status = 3
            for tile in tiles:
                if not self.running:
                    if cpt:
                        self.status = 0
                    return None
                (col, row, z, data) = tile
                if data is None:
                    img = NpImage.new(tileSize, tileSize, bkgColor=EMPTY_TILE_COLOR)
                else:
                    try:
                        img = NpImage(data)
                    except Exception as e:
                        log.error('Corrupted tile on cache', exc_info=True)
                        img = NpImage.new(tileSize, tileSize, bkgColor=CORRUPTED_TILE_COLOR)
                posx = (col - rq.firstCol) * tileSize
                posy = abs(row - rq.firstRow) * tileSize
                mosaic.paste(img, posx, posy)
        if not self.running:
            if cpt:
                self.status = 0
            return None
        if outCRS is not None and outCRS != tm.CRS:
            if cpt:
                self.status = 4
            time.sleep(0.1)
            if not bigTiff:
                mosaic = NpImage(reprojImg(tm.CRS, outCRS, mosaic.toGDAL(), sqPx=True, resamplAlg=self.RESAMP_ALG))
            else:
                outPath = path[:-4] + '_' + str(outCRS) + '.tif'
                ds = reprojImg(tm.CRS, outCRS, mosaic.ds, sqPx=True, resamplAlg=self.RESAMP_ALG, path=outPath)
        if bigTiff:
            ds.BuildOverviews(overviewlist=[2, 4, 8, 16, 32])
            ds = None
        if not bigTiff and path is not None:
            mosaic.save(path)
        if cpt:
            self.status = 0
        if path is None:
            return mosaic
        else:
            return None