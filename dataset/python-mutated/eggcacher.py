import os
import sys
import gc
from panda3d.core import BamCache, ExecutionEnvironment, Filename, Loader, LoaderOptions, ModelPool, TexturePool

class EggCacher:

    def __init__(self, args):
        if False:
            while True:
                i = 10
        maindir = Filename.fromOsSpecific(os.getcwd()).getFullpath()
        ExecutionEnvironment.setEnvironmentVariable('MAIN_DIR', maindir)
        self.bamcache = BamCache.getGlobalPtr()
        self.pandaloader = Loader()
        self.loaderopts = LoaderOptions(LoaderOptions.LF_no_ram_cache)
        if not self.bamcache.getActive():
            print('The model cache is not currently active.')
            print('You must set a model-cache-dir in your config file.')
            sys.exit(1)
        self.parseArgs(args)
        files = self.scanPaths(self.paths)
        self.processFiles(files)

    def parseArgs(self, args):
        if False:
            i = 10
            return i + 15
        self.concise = 0
        self.pzkeep = 0
        while len(args) > 0:
            if args[0] == '--concise':
                self.concise = 1
                args = args[1:]
            elif args[0] == '--pzkeep':
                self.pzkeep = 1
                args = args[1:]
            else:
                break
        if len(args) < 1:
            print('Usage: eggcacher options file-or-directory')
            sys.exit(1)
        self.paths = args

    def scanPath(self, eggs, path):
        if False:
            for i in range(10):
                print('nop')
        if not os.path.exists(path):
            print('No such file or directory: ' + path)
            return
        if os.path.isdir(path):
            for f in os.listdir(path):
                self.scanPath(eggs, os.path.join(path, f))
            return
        if path.endswith('.egg'):
            size = os.path.getsize(path)
            eggs.append((path, size))
            return
        if path.endswith('.egg.pz') or path.endswith('.egg.gz'):
            size = os.path.getsize(path)
            if self.pzkeep:
                eggs.append((path, size))
            else:
                eggs.append((path[:-3], size))

    def scanPaths(self, paths):
        if False:
            print('Hello World!')
        eggs = []
        for path in paths:
            self.scanPath(eggs, path)
        return eggs

    def processFiles(self, files):
        if False:
            while True:
                i = 10
        total = 0
        for (path, size) in files:
            total += size
        progress = 0
        for (path, size) in files:
            fn = Filename.fromOsSpecific(path)
            cached = self.bamcache.lookup(fn, 'bam')
            percent = progress * 100 / total
            report = path
            if self.concise:
                report = os.path.basename(report)
            print('Preprocessing Models %2d%% %s' % (percent, report))
            sys.stdout.flush()
            if cached and (not cached.hasData()):
                self.pandaloader.loadSync(fn, self.loaderopts)
            gc.collect()
            ModelPool.releaseAllModels()
            TexturePool.releaseAllTextures()
            progress += size

def main(args=None):
    if False:
        print('Hello World!')
    if args is None:
        args = sys.argv[1:]
    cacher = EggCacher(args)
    return 0
if __name__ == '__main__':
    sys.exit(main())