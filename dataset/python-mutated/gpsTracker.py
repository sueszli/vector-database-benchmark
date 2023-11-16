from jnius import autoclass, cast
from plyer import gps
from time import sleep
import os
import datetime
from threading import Thread
import jnius
GPSTRACKER_THREAD = None
TRACES = []
(CURRENT_LAT, CURRENT_LON) = (None, None)

def __getLocation__(**kwargs):
    if False:
        return 10
    '\n    This function is called by configure for setting current GPS location in global variables\n    Info: The on_location and on_status callables might be called from another thread than the thread used for creating the GPS object.\n    See https://plyer.readthedocs.io/en/latest/\n    '
    global CURRENT_LAT
    global CURRENT_LON
    if kwargs is not None:
        CURRENT_LAT = kwargs['lat']
        CURRENT_LON = kwargs['lon']

class GpsTracker(Thread):

    def __init__(self, period=15, inMemory=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        '
        Thread.__init__(self)
        gps.configure(on_location=__getLocation__)
        self.stopFollow = False
        self.period = period
        self.inMemory = inMemory
        self.filename = 'keflfjezomef.csv'
        self.Context = autoclass('android.content.Context')
        self.PythonActivity = autoclass('org.renpy.android.PythonService')
        self.LocationManager = autoclass('android.location.LocationManager')

    def enable(self):
        if False:
            return 10
        '\n        '
        gps.start()

    def disable(self):
        if False:
            while True:
                i = 10
        '\n        '
        gps.stop()

    def stop(self):
        if False:
            print('Hello World!')
        '\n        '
        self.stopFollow = True

    def isGPSenabled(self):
        if False:
            return 10
        '\n        '
        locationManager = cast('android.location.LocationManager', self.PythonActivity.mService.getSystemService(self.Context.LOCATION_SERVICE))
        isGPSEnabled = locationManager.isProviderEnabled(self.LocationManager.GPS_PROVIDER)
        return isGPSEnabled

    def isNetworkProviderEnabled(self):
        if False:
            return 10
        '\n        '
        locationManager = cast('android.location.LocationManager', self.PythonActivity.mService.getSystemService(self.Context.LOCATION_SERVICE))
        isNetworkProviderEnabled = locationManager.isProviderEnabled(self.LocationManager.NETWORK_PROVIDER)
        return isNetworkProviderEnabled

    def getCurrentLocation(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        '
        global CURRENT_LAT
        global CURRENT_LON
        return (CURRENT_LAT, CURRENT_LON)

    def follow(self):
        if False:
            return 10
        global TRACES
        self.enable()
        (lastLat, lastLon) = (None, None)
        if not self.inMemory:
            if not os.path.isfile(self.filename):
                f = open(self.filename, 'w')
                f.write('date,latitude,longitude\n')
                f.close()
        while not self.stopFollow:
            (lat, lon) = self.getCurrentLocation()
            if (lat is not None and lon is not None) and (lastLat != lat or lastLon != lon):
                if not self.inMemory:
                    f = open(self.filename, 'a+')
                    f.write('{0},{1},{2}\n'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M'), lat, lon))
                    f.close()
                else:
                    TRACES.append([datetime.datetime.now().strftime('%Y-%m-%d %H:%M'), lat, lon])
            (lastLat, lastLon) = (lat, lon)
            sleep(self.period)
        self.disable()
        jnius.detach()

    def run(self):
        if False:
            print('Hello World!')
        self.stopFollow = False
        self.follow()

    def isFollowing(self):
        if False:
            i = 10
            return i + 15
        if self.stopFollow:
            return False
        else:
            return True

def startGpsTracker(period):
    if False:
        i = 10
        return i + 15
    '\n    '
    global GPSTRACKER_THREAD
    if GPSTRACKER_THREAD is None or not GPSTRACKER_THREAD.isFollowing():
        gpsTracker = GpsTracker(period=period)
        gpsTracker.start()
        GPSTRACKER_THREAD = gpsTracker
        return True
    else:
        return False

def stopGpsTracker():
    if False:
        return 10
    '\n    '
    global GPSTRACKER_THREAD
    if GPSTRACKER_THREAD is None:
        return False
    if not GPSTRACKER_THREAD.isFollowing():
        return False
    else:
        GPSTRACKER_THREAD.stop()
        GPSTRACKER_THREAD.join()
        return True

def dumpGpsTracker():
    if False:
        for i in range(10):
            print('nop')
    '\n    When inMeory is enabled\n    '
    global TRACES
    return TRACES

def statusGpsTracker():
    if False:
        while True:
            i = 10
    '\n    '
    global GPSTRACKER_THREAD
    if GPSTRACKER_THREAD is None:
        return False
    elif not GPSTRACKER_THREAD.isFollowing():
        return False
    else:
        return True

def deleteFile():
    if False:
        while True:
            i = 10
    '\n    '
    if GPSTRACKER_THREAD is not None and (not GPSTRACKER_THREAD.isFollowing()):
        try:
            os.remove(GPSTRACKER_THREAD.filename)
        except OSError:
            return False
        return True
    else:
        return False