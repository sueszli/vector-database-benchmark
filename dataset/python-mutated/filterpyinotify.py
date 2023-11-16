__author__ = 'Cyril Jaquier, Lee Clemens, Yaroslav Halchenko'
__copyright__ = 'Copyright (c) 2004 Cyril Jaquier, 2011-2012 Lee Clemens, 2012 Yaroslav Halchenko'
__license__ = 'GPL'
import logging
from distutils.version import LooseVersion
import os
from os.path import dirname, sep as pathsep
import pyinotify
from .failmanager import FailManagerEmpty
from .filter import FileFilter
from .mytime import MyTime, time
from .utils import Utils
from ..helpers import getLogger
if not hasattr(pyinotify, '__version__') or LooseVersion(pyinotify.__version__) < '0.8.3':
    raise ImportError('Fail2Ban requires pyinotify >= 0.8.3')
try:
    manager = pyinotify.WatchManager()
    del manager
except Exception as e:
    raise ImportError('Pyinotify is probably not functional on this system: %s' % str(e))
logSys = getLogger(__name__)

def _pyinotify_logger_init():
    if False:
        return 10
    return logSys
pyinotify._logger_init = _pyinotify_logger_init
pyinotify.log = logSys

class FilterPyinotify(FileFilter):

    def __init__(self, jail):
        if False:
            i = 10
            return i + 15
        FileFilter.__init__(self, jail)
        self.__monitor = pyinotify.WatchManager()
        self.__notifier = None
        self.__watchFiles = dict()
        self.__watchDirs = dict()
        self.__pending = dict()
        self.__pendingChkTime = 0
        self.__pendingMinTime = 60
        logSys.debug('Created FilterPyinotify')

    def callback(self, event, origin=''):
        if False:
            for i in range(10):
                print('nop')
        logSys.log(4, '[%s] %sCallback for Event: %s', self.jailName, origin, event)
        path = event.pathname
        isWF = False
        isWD = path in self.__watchDirs
        if not isWD and path in self.__watchFiles:
            isWF = True
        assumeNoDir = False
        if event.mask & (pyinotify.IN_CREATE | pyinotify.IN_MOVED_TO):
            if event.mask & pyinotify.IN_ISDIR:
                logSys.debug('Ignoring creation of directory %s', path)
                return
            if not isWF:
                logSys.debug('Ignoring creation of %s we do not monitor', path)
                return
            self._refreshWatcher(path)
        elif event.mask & (pyinotify.IN_IGNORED | pyinotify.IN_MOVE_SELF | pyinotify.IN_DELETE_SELF):
            assumeNoDir = event.mask & (pyinotify.IN_MOVE_SELF | pyinotify.IN_DELETE_SELF)
            if assumeNoDir and path.endswith('-unknown-path') and (not isWF) and (not isWD):
                path = path[:-len('-unknown-path')]
                isWD = path in self.__watchDirs
            if isWD and (assumeNoDir or not os.path.isdir(path)):
                self._addPending(path, event, isDir=True)
            elif not isWF:
                for logpath in self.__watchDirs:
                    if logpath.startswith(path + pathsep) and (assumeNoDir or not os.path.isdir(logpath)):
                        self._addPending(logpath, event, isDir=True)
        if isWF and (not os.path.isfile(path)):
            self._addPending(path, event)
            return
        if self.idle:
            return
        if not isWF:
            logSys.debug('Ignoring event (%s) of %s we do not monitor', event.maskname, path)
            return
        self._process_file(path)

    def _process_file(self, path):
        if False:
            i = 10
            return i + 15
        'Process a given file\n\n\t\tTODO -- RF:\n\t\tthis is a common logic and must be shared/provided by FileFilter\n\t\t'
        if not self.idle:
            self.getFailures(path)

    def _addPending(self, path, reason, isDir=False):
        if False:
            while True:
                i = 10
        if path not in self.__pending:
            self.__pending[path] = [Utils.DEFAULT_SLEEP_INTERVAL, isDir]
            self.__pendingMinTime = 0
            if isinstance(reason, pyinotify.Event):
                reason = [reason.maskname, reason.pathname]
            logSys.log(logging.MSG, 'Log absence detected (possibly rotation) for %s, reason: %s of %s', path, *reason)

    def _delPending(self, path):
        if False:
            return 10
        try:
            del self.__pending[path]
        except KeyError:
            pass

    def getPendingPaths(self):
        if False:
            for i in range(10):
                print('nop')
        return list(self.__pending.keys())

    def _checkPending(self):
        if False:
            i = 10
            return i + 15
        if not self.__pending:
            return
        ntm = time.time()
        if ntm < self.__pendingChkTime + self.__pendingMinTime:
            return
        found = {}
        minTime = 60
        for (path, (retardTM, isDir)) in list(self.__pending.items()):
            if ntm - self.__pendingChkTime < retardTM:
                if minTime > retardTM:
                    minTime = retardTM
                continue
            chkpath = os.path.isdir if isDir else os.path.isfile
            if not chkpath(path):
                if retardTM < 60:
                    retardTM *= 2
                if minTime > retardTM:
                    minTime = retardTM
                self.__pending[path][0] = retardTM
                continue
            logSys.log(logging.MSG, 'Log presence detected for %s %s', 'directory' if isDir else 'file', path)
            found[path] = isDir
        self.__pendingChkTime = time.time()
        self.__pendingMinTime = minTime
        for (path, isDir) in found.items():
            self._delPending(path)
            if isDir is not None:
                self._refreshWatcher(path, isDir=isDir)
            if isDir:
                for logpath in list(self.__watchFiles):
                    if logpath.startswith(path + pathsep):
                        if not os.path.isfile(logpath):
                            self._addPending(logpath, ('FROM_PARDIR', path))
                        else:
                            self._refreshWatcher(logpath)
                            self._process_file(logpath)
            else:
                self._process_file(path)

    def _refreshWatcher(self, oldPath, newPath=None, isDir=False):
        if False:
            while True:
                i = 10
        if not newPath:
            newPath = oldPath
        if not isDir:
            self._delFileWatcher(oldPath)
            self._addFileWatcher(newPath)
        else:
            self._delDirWatcher(oldPath)
            self._addDirWatcher(newPath)

    def _addFileWatcher(self, path):
        if False:
            while True:
                i = 10
        self._addDirWatcher(dirname(path))
        wd = self.__monitor.add_watch(path, pyinotify.IN_MODIFY)
        self.__watchFiles.update(wd)
        logSys.debug('Added file watcher for %s', path)

    def _delWatch(self, wdInt):
        if False:
            while True:
                i = 10
        m = self.__monitor
        try:
            if m.get_path(wdInt) is not None:
                wd = m.rm_watch(wdInt, quiet=False)
                return True
        except pyinotify.WatchManagerError as e:
            if m.get_path(wdInt) is not None and (not str(e).endswith('(EINVAL)')):
                logSys.debug('Remove watch causes: %s', e)
                raise e
        return False

    def _delFileWatcher(self, path):
        if False:
            i = 10
            return i + 15
        try:
            wdInt = self.__watchFiles.pop(path)
            if not self._delWatch(wdInt):
                logSys.debug('Non-existing file watcher %r for file %s', wdInt, path)
            logSys.debug('Removed file watcher for %s', path)
            return True
        except KeyError:
            pass
        return False

    def _addDirWatcher(self, path_dir):
        if False:
            for i in range(10):
                print('nop')
        if path_dir not in self.__watchDirs:
            self.__watchDirs.update(self.__monitor.add_watch(path_dir, pyinotify.IN_CREATE | pyinotify.IN_MOVED_TO | pyinotify.IN_MOVE_SELF | pyinotify.IN_DELETE_SELF | pyinotify.IN_ISDIR))
            logSys.debug('Added monitor for the parent directory %s', path_dir)

    def _delDirWatcher(self, path_dir):
        if False:
            i = 10
            return i + 15
        try:
            wdInt = self.__watchDirs.pop(path_dir)
            if not self._delWatch(wdInt):
                logSys.debug('Non-existing file watcher %r for directory %s', wdInt, path_dir)
            logSys.debug('Removed monitor for the parent directory %s', path_dir)
        except KeyError:
            pass

    def _addLogPath(self, path):
        if False:
            for i in range(10):
                print('nop')
        self._addFileWatcher(path)
        if self.active:
            self.__pendingMinTime = 0
        self._addPending(path, ('INITIAL', path), isDir=None)

    def _delLogPath(self, path):
        if False:
            i = 10
            return i + 15
        self._delPending(path)
        if not self._delFileWatcher(path):
            logSys.error('Failed to remove watch on path: %s', path)
        path_dir = dirname(path)
        for k in list(self.__watchFiles):
            if k.startswith(path_dir + pathsep):
                path_dir = None
                break
        if path_dir:
            self._delPending(path_dir)
            self._delDirWatcher(path_dir)

    def __process_default(self, event):
        if False:
            i = 10
            return i + 15
        try:
            self.callback(event, origin='Default ')
        except Exception as e:
            logSys.error('Error in FilterPyinotify callback: %s', e, exc_info=logSys.getEffectiveLevel() <= logging.DEBUG)
            self.commonError()
        self.ticks += 1

    @property
    def __notify_maxtout(self):
        if False:
            for i in range(10):
                print('nop')
        return min(self.sleeptime, 0.5, self.__pendingMinTime) * 1000

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        prcevent = pyinotify.ProcessEvent()
        prcevent.process_default = self.__process_default
        self.__notifier = pyinotify.Notifier(self.__monitor, prcevent, timeout=self.__notify_maxtout)
        logSys.debug('[%s] filter started (pyinotifier)', self.jailName)
        while self.active:
            try:
                if self.idle:
                    if Utils.wait_for(lambda : not self.active or not self.idle, min(self.sleeptime * 10, self.__pendingMinTime), min(self.sleeptime, self.__pendingMinTime)):
                        if not self.active:
                            break
                self.__notifier.process_events()

                def __check_events():
                    if False:
                        print('Hello World!')
                    return not self.active or bool(self.__notifier.check_events(timeout=self.__notify_maxtout)) or (self.__pendingMinTime and self.__pending)
                wres = Utils.wait_for(__check_events, min(self.sleeptime, self.__pendingMinTime))
                if wres:
                    if not self.active:
                        break
                    if not isinstance(wres, dict):
                        self.__notifier.read_events()
                self.ticks += 1
                if self.idle:
                    continue
                self._checkPending()
                if self.ticks % 10 == 0:
                    self.performSvc()
            except Exception as e:
                if not self.active:
                    break
                logSys.error('Caught unhandled exception in main cycle: %r', e, exc_info=logSys.getEffectiveLevel() <= logging.DEBUG)
                self.commonError('unhandled', e)
        logSys.debug('[%s] filter exited (pyinotifier)', self.jailName)
        self.__notifier = None
        return True

    def stop(self):
        if False:
            while True:
                i = 10
        super(FilterPyinotify, self).stop()
        try:
            if self.__notifier:
                self.__notifier.stop()
        except AttributeError:
            if self.__notifier:
                raise

    def join(self):
        if False:
            while True:
                i = 10
        self.join = lambda *args: 0
        self.__cleanup()
        super(FilterPyinotify, self).join()
        logSys.debug('[%s] filter terminated (pyinotifier)', self.jailName)

    def __cleanup(self):
        if False:
            return 10
        if self.__notifier:
            if Utils.wait_for(lambda : not self.__notifier, self.sleeptime * 10):
                self.__notifier = None
                self.__monitor = None