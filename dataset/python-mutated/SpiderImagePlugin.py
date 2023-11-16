import os
import struct
import sys
from . import Image, ImageFile

def isInt(f):
    if False:
        i = 10
        return i + 15
    try:
        i = int(f)
        if f - i == 0:
            return 1
        else:
            return 0
    except (ValueError, OverflowError):
        return 0
iforms = [1, 3, -11, -12, -21, -22]

def isSpiderHeader(t):
    if False:
        print('Hello World!')
    h = (99,) + t
    for i in [1, 2, 5, 12, 13, 22, 23]:
        if not isInt(h[i]):
            return 0
    iform = int(h[5])
    if iform not in iforms:
        return 0
    labrec = int(h[13])
    labbyt = int(h[22])
    lenbyt = int(h[23])
    if labbyt != labrec * lenbyt:
        return 0
    return labbyt

def isSpiderImage(filename):
    if False:
        for i in range(10):
            print('nop')
    with open(filename, 'rb') as fp:
        f = fp.read(92)
    t = struct.unpack('>23f', f)
    hdrlen = isSpiderHeader(t)
    if hdrlen == 0:
        t = struct.unpack('<23f', f)
        hdrlen = isSpiderHeader(t)
    return hdrlen

class SpiderImageFile(ImageFile.ImageFile):
    format = 'SPIDER'
    format_description = 'Spider 2D image'
    _close_exclusive_fp_after_loading = False

    def _open(self):
        if False:
            i = 10
            return i + 15
        n = 27 * 4
        f = self.fp.read(n)
        try:
            self.bigendian = 1
            t = struct.unpack('>27f', f)
            hdrlen = isSpiderHeader(t)
            if hdrlen == 0:
                self.bigendian = 0
                t = struct.unpack('<27f', f)
                hdrlen = isSpiderHeader(t)
            if hdrlen == 0:
                msg = 'not a valid Spider file'
                raise SyntaxError(msg)
        except struct.error as e:
            msg = 'not a valid Spider file'
            raise SyntaxError(msg) from e
        h = (99,) + t
        iform = int(h[5])
        if iform != 1:
            msg = 'not a Spider 2D image'
            raise SyntaxError(msg)
        self._size = (int(h[12]), int(h[2]))
        self.istack = int(h[24])
        self.imgnumber = int(h[27])
        if self.istack == 0 and self.imgnumber == 0:
            offset = hdrlen
            self._nimages = 1
        elif self.istack > 0 and self.imgnumber == 0:
            self.imgbytes = int(h[12]) * int(h[2]) * 4
            self.hdrlen = hdrlen
            self._nimages = int(h[26])
            offset = hdrlen * 2
            self.imgnumber = 1
        elif self.istack == 0 and self.imgnumber > 0:
            offset = hdrlen + self.stkoffset
            self.istack = 2
        else:
            msg = 'inconsistent stack header values'
            raise SyntaxError(msg)
        if self.bigendian:
            self.rawmode = 'F;32BF'
        else:
            self.rawmode = 'F;32F'
        self._mode = 'F'
        self.tile = [('raw', (0, 0) + self.size, offset, (self.rawmode, 0, 1))]
        self._fp = self.fp

    @property
    def n_frames(self):
        if False:
            return 10
        return self._nimages

    @property
    def is_animated(self):
        if False:
            i = 10
            return i + 15
        return self._nimages > 1

    def tell(self):
        if False:
            print('Hello World!')
        if self.imgnumber < 1:
            return 0
        else:
            return self.imgnumber - 1

    def seek(self, frame):
        if False:
            return 10
        if self.istack == 0:
            msg = 'attempt to seek in a non-stack file'
            raise EOFError(msg)
        if not self._seek_check(frame):
            return
        self.stkoffset = self.hdrlen + frame * (self.hdrlen + self.imgbytes)
        self.fp = self._fp
        self.fp.seek(self.stkoffset)
        self._open()

    def convert2byte(self, depth=255):
        if False:
            for i in range(10):
                print('nop')
        (minimum, maximum) = self.getextrema()
        m = 1
        if maximum != minimum:
            m = depth / (maximum - minimum)
        b = -m * minimum
        return self.point(lambda i, m=m, b=b: i * m + b).convert('L')

    def tkPhotoImage(self):
        if False:
            print('Hello World!')
        from . import ImageTk
        return ImageTk.PhotoImage(self.convert2byte(), palette=256)

def loadImageSeries(filelist=None):
    if False:
        i = 10
        return i + 15
    'create a list of :py:class:`~PIL.Image.Image` objects for use in a montage'
    if filelist is None or len(filelist) < 1:
        return
    imglist = []
    for img in filelist:
        if not os.path.exists(img):
            print(f'unable to find {img}')
            continue
        try:
            with Image.open(img) as im:
                im = im.convert2byte()
        except Exception:
            if not isSpiderImage(img):
                print(img + ' is not a Spider image file')
            continue
        im.info['filename'] = img
        imglist.append(im)
    return imglist

def makeSpiderHeader(im):
    if False:
        print('Hello World!')
    (nsam, nrow) = im.size
    lenbyt = nsam * 4
    labrec = int(1024 / lenbyt)
    if 1024 % lenbyt != 0:
        labrec += 1
    labbyt = labrec * lenbyt
    nvalues = int(labbyt / 4)
    if nvalues < 23:
        return []
    hdr = []
    for i in range(nvalues):
        hdr.append(0.0)
    hdr[1] = 1.0
    hdr[2] = float(nrow)
    hdr[3] = float(nrow)
    hdr[5] = 1.0
    hdr[12] = float(nsam)
    hdr[13] = float(labrec)
    hdr[22] = float(labbyt)
    hdr[23] = float(lenbyt)
    hdr = hdr[1:]
    hdr.append(0.0)
    return [struct.pack('f', v) for v in hdr]

def _save(im, fp, filename):
    if False:
        for i in range(10):
            print('nop')
    if im.mode[0] != 'F':
        im = im.convert('F')
    hdr = makeSpiderHeader(im)
    if len(hdr) < 256:
        msg = 'Error creating Spider header'
        raise OSError(msg)
    fp.writelines(hdr)
    rawmode = 'F;32NF'
    ImageFile._save(im, fp, [('raw', (0, 0) + im.size, 0, (rawmode, 0, 1))])

def _save_spider(im, fp, filename):
    if False:
        for i in range(10):
            print('nop')
    ext = os.path.splitext(filename)[1]
    Image.register_extension(SpiderImageFile.format, ext)
    _save(im, fp, filename)
Image.register_open(SpiderImageFile.format, SpiderImageFile)
Image.register_save(SpiderImageFile.format, _save_spider)
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Syntax: python3 SpiderImagePlugin.py [infile] [outfile]')
        sys.exit()
    filename = sys.argv[1]
    if not isSpiderImage(filename):
        print('input image must be in Spider format')
        sys.exit()
    with Image.open(filename) as im:
        print('image: ' + str(im))
        print('format: ' + str(im.format))
        print('size: ' + str(im.size))
        print('mode: ' + str(im.mode))
        print('max, min: ', end=' ')
        print(im.getextrema())
        if len(sys.argv) > 2:
            outfile = sys.argv[2]
            im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            print(f'saving a flipped version of {os.path.basename(filename)} as {outfile} ')
            im.save(outfile, SpiderImageFile.format)