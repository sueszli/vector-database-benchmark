__license__ = 'GPL v3'
__copyright__ = '2010, Li Fanxi <lifanxi@freemindworld.com>'
__docformat__ = 'restructuredtext en'
import sys, struct, zlib, bz2, os
from calibre import guess_type

class FileStream:

    def IsBinary(self):
        if False:
            return 10
        return self.attr & 1090519040 != 1090519040

class BlockData:
    pass

class SNBFile:
    MAGIC = b'SNBP000B'
    REV80 = 32768
    REVA3 = 10724259
    REVZ1 = 0
    REVZ2 = 0

    def __init__(self, inputFile=None):
        if False:
            for i in range(10):
                print('nop')
        self.files = []
        self.blocks = []
        if inputFile is not None:
            self.Open(inputFile)

    def Open(self, inputFile):
        if False:
            for i in range(10):
                print('nop')
        self.fileName = inputFile
        with open(self.fileName, 'rb') as f:
            f.seek(0)
            self.Parse(f)

    def Parse(self, snbFile, metaOnly=False):
        if False:
            return 10
        vmbr = snbFile.read(44)
        (self.magic, self.rev80, self.revA3, self.revZ1, self.fileCount, self.vfatSize, self.vfatCompressed, self.binStreamSize, self.plainStreamSizeUncompressed, self.revZ2) = struct.unpack('>8siiiiiiiii', vmbr)
        self.vfat = zlib.decompress(snbFile.read(self.vfatCompressed))
        self.ParseFile(self.vfat, self.fileCount)
        snbFile.seek(-16, os.SEEK_END)
        tailblock = snbFile.read(16)
        (self.tailSize, self.tailOffset, self.tailMagic) = struct.unpack('>ii8s', tailblock)
        snbFile.seek(self.tailOffset)
        self.vTailUncompressed = zlib.decompress(snbFile.read(self.tailSize))
        self.tailSizeUncompressed = len(self.vTailUncompressed)
        self.ParseTail(self.vTailUncompressed, self.fileCount)
        binPos = 0
        plainPos = 0
        uncompressedData = None
        for f in self.files:
            if f.attr & 1090519040 == 1090519040:
                if uncompressedData is None:
                    uncompressedData = b''
                    for i in range(self.plainBlock):
                        bzdc = bz2.BZ2Decompressor()
                        if i < self.plainBlock - 1:
                            bSize = self.blocks[self.binBlock + i + 1].Offset - self.blocks[self.binBlock + i].Offset
                        else:
                            bSize = self.tailOffset - self.blocks[self.binBlock + i].Offset
                        snbFile.seek(self.blocks[self.binBlock + i].Offset)
                        try:
                            data = snbFile.read(bSize)
                            if len(data) < 32768:
                                uncompressedData += bzdc.decompress(data)
                            else:
                                uncompressedData += data
                        except Exception:
                            import traceback
                            print(traceback.print_exc())
                if len(uncompressedData) != self.plainStreamSizeUncompressed:
                    raise Exception()
                f.fileBody = uncompressedData[plainPos:plainPos + f.fileSize]
                plainPos += f.fileSize
            elif f.attr & 16777216 == 16777216:
                snbFile.seek(44 + self.vfatCompressed + binPos)
                f.fileBody = snbFile.read(f.fileSize)
                binPos += f.fileSize
            else:
                raise ValueError(f'Invalid file: {f.attr} {f.fileName}')

    def ParseFile(self, vfat, fileCount):
        if False:
            print('Hello World!')
        fileNames = vfat[fileCount * 12:].split(b'\x00')
        for i in range(fileCount):
            f = FileStream()
            (f.attr, f.fileNameOffset, f.fileSize) = struct.unpack('>iii', vfat[i * 12:(i + 1) * 12])
            f.fileName = fileNames[i]
            self.files.append(f)

    def ParseTail(self, vtail, fileCount):
        if False:
            while True:
                i = 10
        self.binBlock = (self.binStreamSize + 32768 - 1) // 32768
        self.plainBlock = (self.plainStreamSizeUncompressed + 32768 - 1) // 32768
        for i in range(self.binBlock + self.plainBlock):
            block = BlockData()
            (block.Offset,) = struct.unpack('>i', vtail[i * 4:(i + 1) * 4])
            self.blocks.append(block)
        for i in range(fileCount):
            (self.files[i].blockIndex, self.files[i].contentOffset) = struct.unpack('>ii', vtail[(self.binBlock + self.plainBlock) * 4 + i * 8:(self.binBlock + self.plainBlock) * 4 + (i + 1) * 8])

    def IsValid(self):
        if False:
            print('Hello World!')
        if self.magic != SNBFile.MAGIC:
            return False
        if self.rev80 != SNBFile.REV80:
            return False
        if self.revZ1 != SNBFile.REVZ1:
            return False
        if self.revZ2 != SNBFile.REVZ2:
            return False
        if self.vfatSize != len(self.vfat):
            return False
        if self.fileCount != len(self.files):
            return False
        if (self.binBlock + self.plainBlock) * 4 + self.fileCount * 8 != self.tailSizeUncompressed:
            return False
        if self.tailMagic != SNBFile.MAGIC:
            print(self.tailMagic)
            return False
        return True

    def FromDir(self, tdir):
        if False:
            for i in range(10):
                print('nop')
        for (root, dirs, files) in os.walk(tdir):
            for name in files:
                (p, ext) = os.path.splitext(name)
                if ext in ['.snbf', '.snbc']:
                    self.AppendPlain(os.path.relpath(os.path.join(root, name), tdir), tdir)
                else:
                    self.AppendBinary(os.path.relpath(os.path.join(root, name), tdir), tdir)

    def AppendPlain(self, fileName, tdir):
        if False:
            return 10
        f = FileStream()
        f.attr = 1090519040
        f.fileSize = os.path.getsize(os.path.join(tdir, fileName))
        with open(os.path.join(tdir, fileName), 'rb') as data:
            f.fileBody = data.read()
        f.fileName = fileName.replace(os.sep, '/')
        if isinstance(f.fileName, str):
            f.fileName = f.fileName.encode('ascii', 'ignore')
        self.files.append(f)

    def AppendBinary(self, fileName, tdir):
        if False:
            while True:
                i = 10
        f = FileStream()
        f.attr = 16777216
        f.fileSize = os.path.getsize(os.path.join(tdir, fileName))
        with open(os.path.join(tdir, fileName), 'rb') as data:
            f.fileBody = data.read()
        f.fileName = fileName.replace(os.sep, '/')
        if isinstance(f.fileName, str):
            f.fileName = f.fileName.encode('ascii', 'ignore')
        self.files.append(f)

    def GetFileStream(self, fileName):
        if False:
            for i in range(10):
                print('nop')
        for file in self.files:
            if file.fileName == fileName:
                return file.fileBody
        return None

    def OutputImageFiles(self, path):
        if False:
            print('Hello World!')
        fileNames = []
        for f in self.files:
            fname = os.path.basename(f.fileName)
            (root, ext) = os.path.splitext(fname)
            if ext in ['.jpeg', '.jpg', '.gif', '.svg', '.png']:
                with open(os.path.join(path, fname), 'wb') as outfile:
                    outfile.write(f.fileBody)
                fileNames.append((fname, guess_type('a' + ext)[0]))
        return fileNames

    def Output(self, outputFile):
        if False:
            print('Hello World!')
        self.files.sort(key=lambda x: x.fileName)
        outputFile = open(outputFile, 'wb')
        vmbrp1 = struct.pack('>8siiii', SNBFile.MAGIC, SNBFile.REV80, SNBFile.REVA3, SNBFile.REVZ1, len(self.files))
        vfat = b''
        fileNameTable = b''
        plainStream = b''
        binStream = b''
        for f in self.files:
            vfat += struct.pack('>iii', f.attr, len(fileNameTable), f.fileSize)
            fileNameTable += f.fileName + b'\x00'
            if f.attr & 1090519040 == 1090519040:
                f.contentOffset = len(plainStream)
                plainStream += f.fileBody
            elif f.attr & 16777216 == 16777216:
                f.contentOffset = len(binStream)
                binStream += f.fileBody
            else:
                raise Exception(f'Unknown file type: {f.attr} {f.fileName}')
        vfatCompressed = zlib.compress(vfat + fileNameTable)
        vmbrp2 = struct.pack('>iiiii', len(vfat + fileNameTable), len(vfatCompressed), len(binStream), len(plainStream), SNBFile.REVZ2)
        outputFile.write(vmbrp1 + vmbrp2)
        outputFile.write(vfatCompressed)
        binBlockOffset = 44 + len(vfatCompressed)
        plainBlockOffset = binBlockOffset + len(binStream)
        binBlock = (len(binStream) + 32768 - 1) // 32768
        offset = 0
        tailBlock = b''
        for i in range(binBlock):
            tailBlock += struct.pack('>i', binBlockOffset + offset)
            offset += 32768
        tailRec = b''
        for f in self.files:
            t = 0
            if f.IsBinary():
                t = 0
            else:
                t = binBlock
            tailRec += struct.pack('>ii', f.contentOffset // 32768 + t, f.contentOffset % 32768)
        outputFile.write(binStream)
        pos = 0
        offset = 0
        while pos < len(plainStream):
            tailBlock += struct.pack('>i', plainBlockOffset + offset)
            block = plainStream[pos:pos + 32768]
            compressed = bz2.compress(block)
            outputFile.write(compressed)
            offset += len(compressed)
            pos += 32768
        compressedTail = zlib.compress(tailBlock + tailRec)
        outputFile.write(compressedTail)
        veom = struct.pack('>ii', len(compressedTail), plainBlockOffset + offset)
        outputFile.write(veom)
        outputFile.write(SNBFile.MAGIC)
        outputFile.close()
        return

    def Dump(self):
        if False:
            print('Hello World!')
        if self.fileName:
            print('File Name:\t', self.fileName)
        print('File Count:\t', self.fileCount)
        print('VFAT Size(Compressed):\t%d(%d)' % (self.vfatSize, self.vfatCompressed))
        print('Binary Stream Size:\t', self.binStreamSize)
        print('Plain Stream Uncompressed Size:\t', self.plainStreamSizeUncompressed)
        print('Binary Block Count:\t', self.binBlock)
        print('Plain Block Count:\t', self.plainBlock)
        for i in range(self.fileCount):
            print('File ', i)
            f = self.files[i]
            print('File Name: ', f.fileName)
            print('File Attr: ', f.attr)
            print('File Size: ', f.fileSize)
            print('Block Index: ', f.blockIndex)
            print('Content Offset: ', f.contentOffset)
            with open('/tmp/' + f.fileName, 'wb') as tempFile:
                tempFile.write(f.fileBody)

def usage():
    if False:
        i = 10
        return i + 15
    print('This unit test is for INTERNAL usage only!')
    print('This unit test accept two parameters.')
    print('python snbfile.py <INPUTFILE> <DESTFILE>')
    print('The input file will be extracted and write to dest file. ')
    print('Meta data of the file will be shown during this process.')

def main():
    if False:
        while True:
            i = 10
    if len(sys.argv) != 3:
        usage()
        sys.exit(0)
    inputFile = sys.argv[1]
    outputFile = sys.argv[2]
    print('Input file: ', inputFile)
    print('Output file: ', outputFile)
    snbFile = SNBFile(inputFile)
    if snbFile.IsValid():
        snbFile.Dump()
        snbFile.Output(outputFile)
    else:
        print('The input file is invalid.')
        return 1
    return 0
if __name__ == '__main__':
    'SNB file unit test'
    sys.exit(main())