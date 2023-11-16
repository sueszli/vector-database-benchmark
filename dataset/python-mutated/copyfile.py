import glob
import os
import shutil
import sys

def main():
    if False:
        while True:
            i = 10
    src = sys.argv[1]
    dst = sys.argv[2]
    if os.path.isdir(src):
        pathList = os.path.split(src)
        dst = os.path.join(dst, pathList[-1])
        if not os.path.exists(dst):
            shutil.copytree(src, dst)
            print(f'first copy directory: {src} --->>> {dst}')
        else:
            shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f'overwritten copy directory: {src} --->>> {dst}')
    else:
        if not os.path.exists(dst):
            os.makedirs(dst)
        srcFiles = glob.glob(src)
        for srcFile in srcFiles:
            shutil.copy(srcFile, dst)
            print(f'copy file: {srcFile} --->>> {dst}')
if __name__ == '__main__':
    main()