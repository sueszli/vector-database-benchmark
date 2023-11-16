import itertools
import os
import struct
from . import ExifTags, Image, ImageFile, ImageSequence, JpegImagePlugin, TiffImagePlugin
from ._binary import i16be as i16
from ._binary import o32le

def _save(im, fp, filename):
    if False:
        while True:
            i = 10
    JpegImagePlugin._save(im, fp, filename)

def _save_all(im, fp, filename):
    if False:
        return 10
    append_images = im.encoderinfo.get('append_images', [])
    if not append_images:
        try:
            animated = im.is_animated
        except AttributeError:
            animated = False
        if not animated:
            _save(im, fp, filename)
            return
    mpf_offset = 28
    offsets = []
    for imSequence in itertools.chain([im], append_images):
        for im_frame in ImageSequence.Iterator(imSequence):
            if not offsets:
                im_frame.encoderinfo['extra'] = b'\xff\xe2' + struct.pack('>H', 6 + 82) + b'MPF\x00' + b' ' * 82
                exif = im_frame.encoderinfo.get('exif')
                if isinstance(exif, Image.Exif):
                    exif = exif.tobytes()
                    im_frame.encoderinfo['exif'] = exif
                if exif:
                    mpf_offset += 4 + len(exif)
                JpegImagePlugin._save(im_frame, fp, filename)
                offsets.append(fp.tell())
            else:
                im_frame.save(fp, 'JPEG')
                offsets.append(fp.tell() - offsets[-1])
    ifd = TiffImagePlugin.ImageFileDirectory_v2()
    ifd[45056] = b'0100'
    ifd[45057] = len(offsets)
    mpentries = b''
    data_offset = 0
    for (i, size) in enumerate(offsets):
        if i == 0:
            mptype = 196608
        else:
            mptype = 0
        mpentries += struct.pack('<LLLHH', mptype, size, data_offset, 0, 0)
        if i == 0:
            data_offset -= mpf_offset
        data_offset += size
    ifd[45058] = mpentries
    fp.seek(mpf_offset)
    fp.write(b'II*\x00' + o32le(8) + ifd.tobytes(8))
    fp.seek(0, os.SEEK_END)

class MpoImageFile(JpegImagePlugin.JpegImageFile):
    format = 'MPO'
    format_description = 'MPO (CIPA DC-007)'
    _close_exclusive_fp_after_loading = False

    def _open(self):
        if False:
            return 10
        self.fp.seek(0)
        JpegImagePlugin.JpegImageFile._open(self)
        self._after_jpeg_open()

    def _after_jpeg_open(self, mpheader=None):
        if False:
            return 10
        self._initial_size = self.size
        self.mpinfo = mpheader if mpheader is not None else self._getmp()
        self.n_frames = self.mpinfo[45057]
        self.__mpoffsets = [mpent['DataOffset'] + self.info['mpoffset'] for mpent in self.mpinfo[45058]]
        self.__mpoffsets[0] = 0
        assert self.n_frames == len(self.__mpoffsets)
        del self.info['mpoffset']
        self.is_animated = self.n_frames > 1
        self._fp = self.fp
        self._fp.seek(self.__mpoffsets[0])
        self.__frame = 0
        self.offset = 0
        self.readonly = 1

    def load_seek(self, pos):
        if False:
            while True:
                i = 10
        self._fp.seek(pos)

    def seek(self, frame):
        if False:
            while True:
                i = 10
        if not self._seek_check(frame):
            return
        self.fp = self._fp
        self.offset = self.__mpoffsets[frame]
        self.fp.seek(self.offset + 2)
        segment = self.fp.read(2)
        if not segment:
            msg = 'No data found for frame'
            raise ValueError(msg)
        self._size = self._initial_size
        if i16(segment) == 65505:
            n = i16(self.fp.read(2)) - 2
            self.info['exif'] = ImageFile._safe_read(self.fp, n)
            self._reload_exif()
            mptype = self.mpinfo[45058][frame]['Attribute']['MPType']
            if mptype.startswith('Large Thumbnail'):
                exif = self.getexif().get_ifd(ExifTags.IFD.Exif)
                if 40962 in exif and 40963 in exif:
                    self._size = (exif[40962], exif[40963])
        elif 'exif' in self.info:
            del self.info['exif']
            self._reload_exif()
        self.tile = [('jpeg', (0, 0) + self.size, self.offset, (self.mode, ''))]
        self.__frame = frame

    def tell(self):
        if False:
            print('Hello World!')
        return self.__frame

    @staticmethod
    def adopt(jpeg_instance, mpheader=None):
        if False:
            while True:
                i = 10
        '\n        Transform the instance of JpegImageFile into\n        an instance of MpoImageFile.\n        After the call, the JpegImageFile is extended\n        to be an MpoImageFile.\n\n        This is essentially useful when opening a JPEG\n        file that reveals itself as an MPO, to avoid\n        double call to _open.\n        '
        jpeg_instance.__class__ = MpoImageFile
        jpeg_instance._after_jpeg_open(mpheader)
        return jpeg_instance
Image.register_save(MpoImageFile.format, _save)
Image.register_save_all(MpoImageFile.format, _save_all)
Image.register_extension(MpoImageFile.format, '.mpo')
Image.register_mime(MpoImageFile.format, 'image/mpo')