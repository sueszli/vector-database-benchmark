"""
NOTE: This is a modified version of whisper.py
For details on the modification, read https://bugs.launchpad.net/graphite/+bug/245835
"""
import os
import struct
import sys
import time
try:
    import fcntl
    CAN_LOCK = True
except ImportError:
    CAN_LOCK = False
if sys.version_info[0] == 3:
    xrange = range
    file = None
LOCK = False
CACHE_HEADERS = False
__headerCache = {}
longFormat = '!L'
longSize = struct.calcsize(longFormat)
floatFormat = '!f'
floatSize = struct.calcsize(floatFormat)
timestampFormat = '!L'
timestampSize = struct.calcsize(timestampFormat)
valueFormat = '!d'
valueSize = struct.calcsize(valueFormat)
pointFormat = '!Ld'
pointSize = struct.calcsize(pointFormat)
metadataFormat = '!2LfL'
metadataSize = struct.calcsize(metadataFormat)
archiveInfoFormat = '!3L'
archiveInfoSize = struct.calcsize(archiveInfoFormat)
debug = startBlock = endBlock = lambda *a, **k: None

def exists(path):
    if False:
        for i in range(10):
            print('nop')
    return os.path.exists(path)

def drop(path):
    if False:
        while True:
            i = 10
    os.remove(path)

def enableMemcache(servers=['127.0.0.1:11211'], min_compress_len=0):
    if False:
        i = 10
        return i + 15
    from StringIO import StringIO
    import memcache
    global open, exists, drop
    MC = memcache.Client(servers)

    class open(StringIO):

        def __init__(self, *args, **kwargs):
            if False:
                print('Hello World!')
            self.name = args[0]
            self.mode = args[1]
            if self.mode == 'r+b' or self.mode == 'rb':
                StringIO.__init__(self, MC.get(self.name))
            else:
                StringIO.__init__(self)

        def close(self):
            if False:
                while True:
                    i = 10
            if self.mode == 'r+b' or self.mode == 'wb':
                MC.set(self.name, self.getvalue(), min_compress_len=min_compress_len)
            StringIO.close(self)

    def exists(path):
        if False:
            for i in range(10):
                print('nop')
        return MC.get(path) is not None

    def drop(path):
        if False:
            i = 10
            return i + 15
        MC.delete(path)

def enableDebug():
    if False:
        for i in range(10):
            print('nop')
    global open, debug, startBlock, endBlock

    class open(file):

        def __init__(self, *args, **kwargs):
            if False:
                return 10
            file.__init__(self, *args, **kwargs)
            self.writeCount = 0
            self.readCount = 0

        def write(self, data):
            if False:
                i = 10
                return i + 15
            self.writeCount += 1
            debug('WRITE %d bytes #%d' % (len(data), self.writeCount))
            return file.write(self, data)

        def read(self, bytes):
            if False:
                return 10
            self.readCount += 1
            debug('READ %d bytes #%d' % (bytes, self.readCount))
            return file.read(self, bytes)

    def debug(message):
        if False:
            return 10
        print('DEBUG :: %s' % message)
    __timingBlocks = {}

    def startBlock(name):
        if False:
            while True:
                i = 10
        __timingBlocks[name] = time.time()

    def endBlock(name):
        if False:
            i = 10
            return i + 15
        debug('%s took %.5f seconds' % (name, time.time() - __timingBlocks.pop(name)))

def __readHeader(fh):
    if False:
        print('Hello World!')
    info = __headerCache.get(fh.name)
    if info:
        return info
    originalOffset = fh.tell()
    fh.seek(0)
    packedMetadata = fh.read(metadataSize)
    (lastUpdate, maxRetention, xff, archiveCount) = struct.unpack(metadataFormat, packedMetadata)
    archives = []
    for i in xrange(archiveCount):
        packedArchiveInfo = fh.read(archiveInfoSize)
        (offset, secondsPerPoint, points) = struct.unpack(archiveInfoFormat, packedArchiveInfo)
        archiveInfo = {'offset': offset, 'secondsPerPoint': secondsPerPoint, 'points': points, 'retention': secondsPerPoint * points, 'size': points * pointSize}
        archives.append(archiveInfo)
    fh.seek(originalOffset)
    info = {'lastUpdate': lastUpdate, 'maxRetention': maxRetention, 'xFilesFactor': xff, 'archives': archives}
    if CACHE_HEADERS:
        __headerCache[fh.name] = info
    return info

def __changeLastUpdate(fh):
    if False:
        while True:
            i = 10
    return
    startBlock('__changeLastUpdate()')
    originalOffset = fh.tell()
    fh.seek(0)
    now = int(time.time())
    packedTime = struct.pack(timestampFormat, now)
    fh.write(packedTime)
    fh.seek(originalOffset)
    endBlock('__changeLastUpdate()')

def create(path, archiveList, xFilesFactor=0.5):
    if False:
        i = 10
        return i + 15
    'create(path,archiveList,xFilesFactor=0.5)\n\npath is a string\narchiveList is a list of archives, each of which is of the form (secondsPerPoint,numberOfPoints)\nxFilesFactor specifies the fraction of data points in a propagation interval that must have known values for a propagation to occur\n'
    assert archiveList, 'You must specify at least one archive configuration!'
    archiveList.sort(key=lambda a: a[0])
    for (i, archive) in enumerate(archiveList):
        if i == len(archiveList) - 1:
            break
        next = archiveList[i + 1]
        assert archive[0] < next[0], 'You cannot configure two archives with the same precision %s,%s' % (archive, next)
        assert next[0] % archive[0] == 0, "Higher precision archives' precision must evenly divide all lower precision archives' precision %s,%s" % (archive[0], next[0])
        retention = archive[0] * archive[1]
        nextRetention = next[0] * next[1]
        assert nextRetention > retention, 'Lower precision archives must cover larger time intervals than higher precision archives %s,%s' % (archive, next)
    assert not exists(path), 'File %s already exists!' % path
    fh = open(path, 'wb')
    if LOCK:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
    lastUpdate = struct.pack(timestampFormat, int(time.time()))
    oldest = sorted([secondsPerPoint * points for (secondsPerPoint, points) in archiveList])[-1]
    maxRetention = struct.pack(longFormat, oldest)
    xFilesFactor = struct.pack(floatFormat, float(xFilesFactor))
    archiveCount = struct.pack(longFormat, len(archiveList))
    packedMetadata = lastUpdate + maxRetention + xFilesFactor + archiveCount
    fh.write(packedMetadata)
    headerSize = metadataSize + archiveInfoSize * len(archiveList)
    archiveOffsetPointer = headerSize
    for (secondsPerPoint, points) in archiveList:
        archiveInfo = struct.pack(archiveInfoFormat, archiveOffsetPointer, secondsPerPoint, points)
        fh.write(archiveInfo)
        archiveOffsetPointer += points * pointSize
    zeroes = '\x00' * (archiveOffsetPointer - headerSize)
    fh.write(zeroes)
    fh.close()

def __propagate(fh, timestamp, xff, higher, lower):
    if False:
        return 10
    lowerIntervalStart = timestamp - timestamp % lower['secondsPerPoint']
    fh.seek(higher['offset'])
    packedPoint = fh.read(pointSize)
    (higherBaseInterval, higherBaseValue) = struct.unpack(pointFormat, packedPoint)
    if higherBaseInterval == 0:
        higherFirstOffset = higher['offset']
    else:
        timeDistance = lowerIntervalStart - higherBaseInterval
        pointDistance = timeDistance / higher['secondsPerPoint']
        byteDistance = pointDistance * pointSize
        higherFirstOffset = higher['offset'] + byteDistance % higher['size']
    higherPoints = lower['secondsPerPoint'] / higher['secondsPerPoint']
    higherSize = higherPoints * pointSize
    higherLastOffset = higherFirstOffset + higherSize % higher['size']
    fh.seek(higherFirstOffset)
    if higherFirstOffset < higherLastOffset:
        seriesString = fh.read(higherLastOffset - higherFirstOffset)
    else:
        higherEnd = higher['offset'] + higher['size']
        seriesString = fh.read(higherEnd - higherFirstOffset)
        fh.seek(higher['offset'])
        seriesString += fh.read(higherLastOffset - higher['offset'])
    (byteOrder, pointTypes) = (pointFormat[0], pointFormat[1:])
    points = len(seriesString) / pointSize
    seriesFormat = byteOrder + pointTypes * points
    unpackedSeries = struct.unpack(seriesFormat, seriesString)
    neighborValues = [None] * points
    currentInterval = lowerIntervalStart
    step = higher['secondsPerPoint']
    for i in xrange(0, len(unpackedSeries), 2):
        pointTime = unpackedSeries[i]
        if pointTime == currentInterval:
            neighborValues[i / 2] = unpackedSeries[i + 1]
        currentInterval += step
    knownValues = [v for v in neighborValues if v is not None]
    knownPercent = float(len(knownValues)) / float(len(neighborValues))
    if knownPercent >= xff:
        aggregateValue = float(sum(knownValues)) / float(len(knownValues))
        myPackedPoint = struct.pack(pointFormat, lowerIntervalStart, aggregateValue)
        fh.seek(lower['offset'])
        packedPoint = fh.read(pointSize)
        (lowerBaseInterval, lowerBaseValue) = struct.unpack(pointFormat, packedPoint)
        if lowerBaseInterval == 0:
            fh.seek(lower['offset'])
            fh.write(myPackedPoint)
        else:
            timeDistance = lowerIntervalStart - lowerBaseInterval
            pointDistance = timeDistance / lower['secondsPerPoint']
            byteDistance = pointDistance * pointSize
            lowerOffset = lower['offset'] + byteDistance % lower['size']
            fh.seek(lowerOffset)
            fh.write(myPackedPoint)
        return True
    else:
        return False

def update(path, value, timestamp=None):
    if False:
        i = 10
        return i + 15
    'update(path,value,timestamp=None)\n\npath is a string\nvalue is a float\ntimestamp is either an int or float\n'
    value = float(value)
    fh = open(path, 'r+b')
    if LOCK:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
    header = __readHeader(fh)
    now = int(time.time())
    if timestamp is None:
        timestamp = now
    timestamp = int(timestamp)
    diff = now - timestamp
    assert diff < header['maxRetention'] and diff >= 0, 'Timestamp not covered by any archives in this database'
    for (i, archive) in enumerate(header['archives']):
        if archive['retention'] < diff:
            continue
        lowerArchives = header['archives'][i + 1:]
        break
    myInterval = timestamp - timestamp % archive['secondsPerPoint']
    myPackedPoint = struct.pack(pointFormat, myInterval, value)
    fh.seek(archive['offset'])
    packedPoint = fh.read(pointSize)
    (baseInterval, baseValue) = struct.unpack(pointFormat, packedPoint)
    if baseInterval == 0:
        fh.seek(archive['offset'])
        fh.write(myPackedPoint)
        (baseInterval, baseValue) = (myInterval, value)
    else:
        timeDistance = myInterval - baseInterval
        pointDistance = timeDistance / archive['secondsPerPoint']
        byteDistance = pointDistance * pointSize
        myOffset = archive['offset'] + byteDistance % archive['size']
        fh.seek(myOffset)
        fh.write(myPackedPoint)
    higher = archive
    for lower in lowerArchives:
        if not __propagate(fh, myInterval, header['xFilesFactor'], higher, lower):
            break
        higher = lower
    __changeLastUpdate(fh)
    fh.close()

def update_many(path, points):
    if False:
        print('Hello World!')
    'update_many(path,points)\n\npath is a string\npoints is a list of (timestamp,value) points\n'
    if not points:
        return
    points = [(int(t), float(v)) for (t, v) in points]
    points.sort(key=lambda p: p[0], reverse=True)
    fh = open(path, 'r+b')
    if LOCK:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
    header = __readHeader(fh)
    now = int(time.time())
    archives = iter(header['archives'])
    currentArchive = next(archives)
    currentPoints = []
    for point in points:
        age = now - point[0]
        while currentArchive['retention'] < age:
            if currentPoints:
                currentPoints.reverse()
                __archive_update_many(fh, header, currentArchive, currentPoints)
                currentPoints = []
            try:
                currentArchive = next(archives)
            except StopIteration:
                currentArchive = None
                break
        if not currentArchive:
            break
        currentPoints.append(point)
    if currentArchive and currentPoints:
        currentPoints.reverse()
        __archive_update_many(fh, header, currentArchive, currentPoints)
    __changeLastUpdate(fh)
    fh.close()

def __archive_update_many(fh, header, archive, points):
    if False:
        return 10
    step = archive['secondsPerPoint']
    alignedPoints = [(timestamp - timestamp % step, value) for (timestamp, value) in points]
    packedStrings = []
    previousInterval = None
    currentString = ''
    for (interval, value) in alignedPoints:
        if not previousInterval or interval == previousInterval + step:
            currentString += struct.pack(pointFormat, interval, value)
            previousInterval = interval
        else:
            numberOfPoints = len(currentString) / pointSize
            startInterval = previousInterval - step * (numberOfPoints - 1)
            packedStrings.append((startInterval, currentString))
            currentString = struct.pack(pointFormat, interval, value)
            previousInterval = interval
    if currentString:
        numberOfPoints = len(currentString) / pointSize
        startInterval = previousInterval - step * (numberOfPoints - 1)
        packedStrings.append((startInterval, currentString))
    fh.seek(archive['offset'])
    packedBasePoint = fh.read(pointSize)
    (baseInterval, baseValue) = struct.unpack(pointFormat, packedBasePoint)
    if baseInterval == 0:
        baseInterval = packedStrings[0][0]
    for (interval, packedString) in packedStrings:
        timeDistance = interval - baseInterval
        pointDistance = timeDistance / step
        byteDistance = pointDistance * pointSize
        myOffset = archive['offset'] + byteDistance % archive['size']
        fh.seek(myOffset)
        archiveEnd = archive['offset'] + archive['size']
        bytesBeyond = myOffset + len(packedString) - archiveEnd
        if bytesBeyond > 0:
            fh.write(packedString[:-bytesBeyond])
            assert fh.tell() == archiveEnd, 'archiveEnd=%d fh.tell=%d bytesBeyond=%d len(packedString)=%d' % (archiveEnd, fh.tell(), bytesBeyond, len(packedString))
            fh.seek(archive['offset'])
            fh.write(packedString[-bytesBeyond:])
        else:
            fh.write(packedString)
    higher = archive
    lowerArchives = [arc for arc in header['archives'] if arc['secondsPerPoint'] > archive['secondsPerPoint']]
    for lower in lowerArchives:
        fit = lambda i: i - i % lower['secondsPerPoint']
        lowerIntervals = [fit(p[0]) for p in alignedPoints]
        uniqueLowerIntervals = set(lowerIntervals)
        propagateFurther = False
        for interval in uniqueLowerIntervals:
            if __propagate(fh, interval, header['xFilesFactor'], higher, lower):
                propagateFurther = True
        if not propagateFurther:
            break
        higher = lower

def info(path):
    if False:
        while True:
            i = 10
    'info(path)\n\npath is a string\n'
    fh = open(path, 'rb')
    info = __readHeader(fh)
    fh.close()
    return info

def fetch(path, fromTime, untilTime=None):
    if False:
        i = 10
        return i + 15
    'fetch(path,fromTime,untilTime=None)\n\npath is a string\nfromTime is an epoch time\nuntilTime is also an epoch time, but defaults to now\n'
    fh = open(path, 'rb')
    header = __readHeader(fh)
    now = int(time.time())
    if untilTime is None or untilTime > now:
        untilTime = now
    if fromTime < now - header['maxRetention']:
        fromTime = now - header['maxRetention']
    assert fromTime < untilTime, 'Invalid time interval'
    diff = now - fromTime
    for archive in header['archives']:
        if archive['retention'] >= diff:
            break
    fromInterval = int(fromTime - fromTime % archive['secondsPerPoint'])
    untilInterval = int(untilTime - untilTime % archive['secondsPerPoint'])
    fh.seek(archive['offset'])
    packedPoint = fh.read(pointSize)
    (baseInterval, baseValue) = struct.unpack(pointFormat, packedPoint)
    if baseInterval == 0:
        step = archive['secondsPerPoint']
        points = (untilInterval - fromInterval) / step
        timeInfo = (fromInterval, untilInterval, step)
        valueList = [None] * points
        return (timeInfo, valueList)
    timeDistance = fromInterval - baseInterval
    pointDistance = timeDistance / archive['secondsPerPoint']
    byteDistance = pointDistance * pointSize
    fromOffset = archive['offset'] + byteDistance % archive['size']
    timeDistance = untilInterval - baseInterval
    pointDistance = timeDistance / archive['secondsPerPoint']
    byteDistance = pointDistance * pointSize
    untilOffset = archive['offset'] + byteDistance % archive['size']
    fh.seek(fromOffset)
    if fromOffset < untilOffset:
        seriesString = fh.read(untilOffset - fromOffset)
    else:
        archiveEnd = archive['offset'] + archive['size']
        seriesString = fh.read(archiveEnd - fromOffset)
        fh.seek(archive['offset'])
        seriesString += fh.read(untilOffset - archive['offset'])
    (byteOrder, pointTypes) = (pointFormat[0], pointFormat[1:])
    points = len(seriesString) / pointSize
    seriesFormat = byteOrder + pointTypes * points
    unpackedSeries = struct.unpack(seriesFormat, seriesString)
    valueList = [None] * points
    currentInterval = fromInterval
    step = archive['secondsPerPoint']
    for i in xrange(0, len(unpackedSeries), 2):
        pointTime = unpackedSeries[i]
        if pointTime == currentInterval:
            pointValue = unpackedSeries[i + 1]
            valueList[i / 2] = pointValue
        currentInterval += step
    fh.close()
    timeInfo = (fromInterval, untilInterval, step)
    return (timeInfo, valueList)