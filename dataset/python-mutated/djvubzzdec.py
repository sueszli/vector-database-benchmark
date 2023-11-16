__license__ = 'GPL v3'
__copyright__ = '2011, Anthon van der Neut <A.van.der.Neut@ruamel.eu>'
original_copyright_notice = '\n//C- -------------------------------------------------------------------\n//C- DjVuLibre-3.5\n//C- Copyright (c) 2002  Leon Bottou and Yann Le Cun.\n//C- Copyright (c) 2001  AT&T\n//C-\n//C- This software is subject to, and may be distributed under, the\n//C- GNU General Public License, either Version 2 of the license,\n//C- or (at your option) any later version. The license should have\n//C- accompanied the software or you may obtain a copy of the license\n//C- from the Free Software Foundation at http://www.fsf.org .\n//C-\n//C- This program is distributed in the hope that it will be useful,\n//C- but WITHOUT ANY WARRANTY; without even the implied warranty of\n//C- MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n//C- GNU General Public License for more details.\n//C-\n//C- DjVuLibre-3.5 is derived from the DjVu(r) Reference Library from\n//C- Lizardtech Software.  Lizardtech Software has authorized us to\n//C- replace the original DjVu(r) Reference Library notice by the following\n//C- text (see doc/lizard2002.djvu and doc/lizardtech2007.djvu):\n//C-\n//C-  ------------------------------------------------------------------\n//C- | DjVu (r) Reference Library (v. 3.5)\n//C- | Copyright (c) 1999-2001 LizardTech, Inc. All Rights Reserved.\n//C- | The DjVu Reference Library is protected by U.S. Pat. No.\n//C- | 6,058,214 and patents pending.\n//C- |\n//C- | This software is subject to, and may be distributed under, the\n//C- | GNU General Public License, either Version 2 of the license,\n//C- | or (at your option) any later version. The license should have\n//C- | accompanied the software or you may obtain a copy of the license\n//C- | from the Free Software Foundation at http://www.fsf.org .\n//C- |\n//C- | The computer code originally released by LizardTech under this\n//C- | license and unmodified by other parties is deemed "the LIZARDTECH\n//C- | ORIGINAL CODE."  Subject to any third party intellectual property\n//C- | claims, LizardTech grants recipient a worldwide, royalty-free,\n//C- | non-exclusive license to make, use, sell, or otherwise dispose of\n//C- | the LIZARDTECH ORIGINAL CODE or of programs derived from the\n//C- | LIZARDTECH ORIGINAL CODE in compliance with the terms of the GNU\n//C- | General Public License.   This grant only confers the right to\n//C- | infringe patent claims underlying the LIZARDTECH ORIGINAL CODE to\n//C- | the extent such infringement is reasonably necessary to enable\n//C- | recipient to make, have made, practice, sell, or otherwise dispose\n//C- | of the LIZARDTECH ORIGINAL CODE (or portions thereof) and not to\n//C- | any greater extent that may be necessary to utilize further\n//C- | modifications or combinations.\n//C- |\n//C- | The LIZARDTECH ORIGINAL CODE is provided "AS IS" WITHOUT WARRANTY\n//C- | OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED\n//C- | TO ANY WARRANTY OF NON-INFRINGEMENT, OR ANY IMPLIED WARRANTY OF\n//C- | MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.\n//C- +------------------------------------------------------------------\n//\n// $Id: BSByteStream.cpp,v 1.9 2007/03/25 20:48:29 leonb Exp $\n// $Name: release_3_5_23 $\n'
MAXBLOCK = 4096
FREQMAX = 4
CTXIDS = 3
MAXLEN = 1024 ** 2

class BZZDecoderError(Exception):
    """This exception is raised when BZZDecode runs into trouble
    """

    def __init__(self, msg):
        if False:
            return 10
        self.msg = msg

    def __str__(self):
        if False:
            print('Hello World!')
        return 'BZZDecoderError: %s' % self.msg
default_ztable = [(32768, 0, 84, 145), (32768, 0, 3, 4), (32768, 0, 4, 3), (27581, 4261, 5, 1), (27581, 4261, 6, 2), (23877, 7976, 7, 3), (23877, 7976, 8, 4), (20921, 11219, 9, 5), (20921, 11219, 10, 6), (18451, 14051, 11, 7), (18451, 14051, 12, 8), (16341, 16524, 13, 9), (16341, 16524, 14, 10), (14513, 18685, 15, 11), (14513, 18685, 16, 12), (12917, 20573, 17, 13), (12917, 20573, 18, 14), (11517, 22224, 19, 15), (11517, 22224, 20, 16), (10277, 23665, 21, 17), (10277, 23665, 22, 18), (9131, 24923, 23, 19), (9131, 24923, 24, 20), (8071, 26021, 25, 21), (8071, 26021, 26, 22), (7099, 26978, 27, 23), (7099, 26978, 28, 24), (6213, 27810, 29, 25), (6213, 27810, 30, 26), (5411, 28532, 31, 27), (5411, 28532, 32, 28), (4691, 29158, 33, 29), (4691, 29158, 34, 30), (4047, 29700, 35, 31), (4047, 29700, 36, 32), (3477, 30166, 37, 33), (3477, 30166, 38, 34), (2973, 30568, 39, 35), (2973, 30568, 40, 36), (2531, 30914, 41, 37), (2531, 30914, 42, 38), (2145, 31210, 43, 39), (2145, 31210, 44, 40), (1809, 31463, 45, 41), (1809, 31463, 46, 42), (1521, 31678, 47, 43), (1521, 31678, 48, 44), (1273, 31861, 49, 45), (1273, 31861, 50, 46), (1061, 32015, 51, 47), (1061, 32015, 52, 48), (881, 32145, 53, 49), (881, 32145, 54, 50), (729, 32254, 55, 51), (729, 32254, 56, 52), (601, 32346, 57, 53), (601, 32346, 58, 54), (493, 32422, 59, 55), (493, 32422, 60, 56), (403, 32486, 61, 57), (403, 32486, 62, 58), (329, 32538, 63, 59), (329, 32538, 64, 60), (267, 32581, 65, 61), (267, 32581, 66, 62), (213, 32619, 67, 63), (213, 32619, 68, 64), (165, 32653, 69, 65), (165, 32653, 70, 66), (123, 32682, 71, 67), (123, 32682, 72, 68), (87, 32707, 73, 69), (87, 32707, 74, 70), (59, 32727, 75, 71), (59, 32727, 76, 72), (35, 32743, 77, 73), (35, 32743, 78, 74), (19, 32754, 79, 75), (19, 32754, 80, 76), (7, 32762, 81, 77), (7, 32762, 82, 78), (1, 32767, 81, 79), (1, 32767, 82, 80), (22165, 0, 9, 85), (9454, 0, 86, 226), (32768, 0, 5, 6), (3376, 0, 88, 176), (18458, 0, 89, 143), (1153, 0, 90, 138), (13689, 0, 91, 141), (378, 0, 92, 112), (9455, 0, 93, 135), (123, 0, 94, 104), (6520, 0, 95, 133), (40, 0, 96, 100), (4298, 0, 97, 129), (13, 0, 82, 98), (2909, 0, 99, 127), (52, 0, 76, 72), (1930, 0, 101, 125), (160, 0, 70, 102), (1295, 0, 103, 123), (279, 0, 66, 60), (856, 0, 105, 121), (490, 0, 106, 110), (564, 0, 107, 119), (324, 0, 66, 108), (371, 0, 109, 117), (564, 0, 60, 54), (245, 0, 111, 115), (851, 0, 56, 48), (161, 0, 69, 113), (1477, 0, 114, 134), (282, 0, 65, 59), (975, 0, 116, 132), (426, 0, 61, 55), (645, 0, 118, 130), (646, 0, 57, 51), (427, 0, 120, 128), (979, 0, 53, 47), (282, 0, 122, 126), (1477, 0, 49, 41), (186, 0, 124, 62), (2221, 0, 43, 37), (122, 0, 72, 66), (3276, 0, 39, 31), (491, 0, 60, 54), (4866, 0, 33, 25), (742, 0, 56, 50), (7041, 0, 29, 131), (1118, 0, 52, 46), (9455, 0, 23, 17), (1680, 0, 48, 40), (10341, 0, 23, 15), (2526, 0, 42, 136), (14727, 0, 137, 7), (3528, 0, 38, 32), (11417, 0, 21, 139), (4298, 0, 140, 172), (15199, 0, 15, 9), (2909, 0, 142, 170), (22165, 0, 9, 85), (1930, 0, 144, 168), (32768, 0, 141, 248), (1295, 0, 146, 166), (9454, 0, 147, 247), (856, 0, 148, 164), (3376, 0, 149, 197), (564, 0, 150, 162), (1153, 0, 151, 95), (371, 0, 152, 160), (378, 0, 153, 173), (245, 0, 154, 158), (123, 0, 155, 165), (161, 0, 70, 156), (40, 0, 157, 161), (282, 0, 66, 60), (13, 0, 81, 159), (426, 0, 62, 56), (52, 0, 75, 71), (646, 0, 58, 52), (160, 0, 69, 163), (979, 0, 54, 48), (279, 0, 65, 59), (1477, 0, 50, 42), (490, 0, 167, 171), (2221, 0, 44, 38), (324, 0, 65, 169), (3276, 0, 40, 32), (564, 0, 59, 53), (4866, 0, 34, 26), (851, 0, 55, 47), (7041, 0, 30, 174), (1477, 0, 175, 193), (9455, 0, 24, 18), (975, 0, 177, 191), (11124, 0, 178, 222), (645, 0, 179, 189), (8221, 0, 180, 218), (427, 0, 181, 187), (5909, 0, 182, 216), (282, 0, 183, 185), (4023, 0, 184, 214), (186, 0, 69, 61), (2663, 0, 186, 212), (491, 0, 59, 53), (1767, 0, 188, 210), (742, 0, 55, 49), (1174, 0, 190, 208), (1118, 0, 51, 45), (781, 0, 192, 206), (1680, 0, 47, 39), (518, 0, 194, 204), (2526, 0, 41, 195), (341, 0, 196, 202), (3528, 0, 37, 31), (225, 0, 198, 200), (11124, 0, 199, 243), (148, 0, 72, 64), (8221, 0, 201, 239), (392, 0, 62, 56), (5909, 0, 203, 237), (594, 0, 58, 52), (4023, 0, 205, 235), (899, 0, 54, 48), (2663, 0, 207, 233), (1351, 0, 50, 44), (1767, 0, 209, 231), (2018, 0, 46, 38), (1174, 0, 211, 229), (3008, 0, 40, 34), (781, 0, 213, 227), (4472, 0, 36, 28), (518, 0, 215, 225), (6618, 0, 30, 22), (341, 0, 217, 223), (9455, 0, 26, 16), (225, 0, 219, 221), (12814, 0, 20, 220), (148, 0, 71, 63), (17194, 0, 14, 8), (392, 0, 61, 55), (17533, 0, 14, 224), (594, 0, 57, 51), (24270, 0, 8, 2), (899, 0, 53, 47), (32768, 0, 228, 87), (1351, 0, 49, 43), (18458, 0, 230, 246), (2018, 0, 45, 37), (13689, 0, 232, 244), (3008, 0, 39, 33), (9455, 0, 234, 238), (4472, 0, 35, 27), (6520, 0, 138, 236), (6618, 0, 29, 21), (10341, 0, 24, 16), (9455, 0, 25, 15), (14727, 0, 240, 8), (12814, 0, 19, 241), (11417, 0, 22, 242), (17194, 0, 13, 7), (15199, 0, 16, 10), (17533, 0, 13, 245), (22165, 0, 10, 2), (24270, 0, 7, 1), (32768, 0, 244, 83), (32768, 0, 249, 250), (22165, 0, 10, 2), (18458, 0, 89, 143), (18458, 0, 230, 246), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)]
xmtf = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255)

class BZZDecoder:

    def __init__(self, infile, outfile):
        if False:
            while True:
                i = 10
        self.instream = infile
        self.inptr = 0
        self.outf = outfile
        self.ieof = False
        self.bptr = None
        self.xsize = None
        self.outbuf = [0] * (MAXBLOCK * 1024)
        self.byte = None
        self.scount = 0
        self.delay = 25
        self.a = 0
        self.code = 0
        self.bufint = 0
        self.ctx = [0] * 300
        self.p = [0] * 256
        self.m = [0] * 256
        self.up = [0] * 256
        self.dn = [0] * 256
        self.ffzt = [0] * 256
        for i in range(256):
            j = i
            while j & 128:
                self.ffzt[i] += 1
                j <<= 1
        self.newtable(default_ztable)
        if not self.read_byte():
            self.byte = 255
        self.code = self.byte << 8
        if not self.read_byte():
            self.byte = 255
        self.code = self.code | self.byte
        self.preload()
        self.fence = self.code
        if self.code >= 32768:
            self.fence = 32767

    def convert(self, sz):
        if False:
            for i in range(10):
                print('nop')
        if self.ieof:
            return 0
        copied = 0
        while sz > 0 and (not self.ieof):
            if not self.xsize:
                self.bptr = 0
                if not self.decode():
                    self.xsize = 1
                    self.ieof = True
                self.xsize -= 1
            remaining = min(sz, self.xsize)
            if remaining > 0:
                self.outf.extend(self.outbuf[self.bptr:self.bptr + remaining])
            self.xsize -= remaining
            self.bptr += remaining
            sz -= remaining
            copied += remaining
        return copied

    def preload(self):
        if False:
            i = 10
            return i + 15
        while self.scount <= 24:
            if not self.read_byte():
                self.byte = 255
                self.delay -= 1
                if self.delay < 1:
                    raise BZZDecoderError('BiteStream EOF')
            self.bufint = self.bufint << 8 | self.byte
            self.scount += 8

    def newtable(self, table):
        if False:
            for i in range(10):
                print('nop')
        for i in range(256):
            self.p[i] = table[i][0]
            self.m[i] = table[i][1]
            self.up[i] = table[i][2]
            self.dn[i] = table[i][3]

    def decode(self):
        if False:
            print('Hello World!')
        outbuf = self.outbuf
        self.xsize = self.decode_raw(24)
        if not self.xsize:
            return 0
        if self.xsize > MAXBLOCK * 1024:
            raise BZZDecoderError('BiteStream.corrupt')
        fshift = 0
        if self.zpcodec_decoder():
            fshift += 1
            if self.zpcodec_decoder():
                fshift += 1
        mtf = list(xmtf)
        freq = [0] * FREQMAX
        fadd = 4
        mtfno = 3
        markerpos = -1

        def zc(i):
            if False:
                for i in range(10):
                    print('nop')
            return self.zpcodec_decode(self.ctx, i)

        def dc(i, bits):
            if False:
                while True:
                    i = 10
            return self.decode_binary(self.ctx, i, bits)
        for i in range(self.xsize):
            ctxid = CTXIDS - 1
            if ctxid > mtfno:
                ctxid = mtfno
            if zc(ctxid):
                mtfno = 0
                outbuf[i] = mtf[mtfno]
            elif zc(ctxid + CTXIDS):
                mtfno = 1
                outbuf[i] = mtf[mtfno]
            elif zc(2 * CTXIDS):
                mtfno = 2 + dc(2 * CTXIDS + 1, 1)
                outbuf[i] = mtf[mtfno]
            elif zc(2 * CTXIDS + 2):
                mtfno = 4 + dc(2 * CTXIDS + 2 + 1, 2)
                outbuf[i] = mtf[mtfno]
            elif zc(2 * CTXIDS + 6):
                mtfno = 8 + dc(2 * CTXIDS + 6 + 1, 3)
                outbuf[i] = mtf[mtfno]
            elif zc(2 * CTXIDS + 14):
                mtfno = 16 + dc(2 * CTXIDS + 14 + 1, 4)
                outbuf[i] = mtf[mtfno]
            elif zc(2 * CTXIDS + 30):
                mtfno = 32 + dc(2 * CTXIDS + 30 + 1, 5)
                outbuf[i] = mtf[mtfno]
            elif zc(2 * CTXIDS + 62):
                mtfno = 64 + dc(2 * CTXIDS + 62 + 1, 6)
                outbuf[i] = mtf[mtfno]
            elif zc(2 * CTXIDS + 126):
                mtfno = 128 + dc(2 * CTXIDS + 126 + 1, 7)
                outbuf[i] = mtf[mtfno]
            else:
                mtfno = 256
                outbuf[i] = 0
                markerpos = i
                continue
            fadd = fadd + (fadd >> fshift)
            if fadd > 268435456:
                fadd >>= 24
                freq[0] >>= 24
                freq[1] >>= 24
                freq[2] >>= 24
                freq[3] >>= 24
                for k in range(4, FREQMAX):
                    freq[k] = freq[k] >> 24
            fc = fadd
            if mtfno < FREQMAX:
                fc += freq[mtfno]
            k = mtfno
            while k >= FREQMAX:
                mtf[k] = mtf[k - 1]
                k -= 1
            while k > 0 and fc >= freq[k - 1]:
                mtf[k] = mtf[k - 1]
                freq[k] = freq[k - 1]
                k -= 1
            mtf[k] = outbuf[i]
            freq[k] = fc
        if markerpos < 1 or markerpos >= self.xsize:
            raise BZZDecoderError('BiteStream.corrupt')
        posn = [0] * self.xsize
        count = [0] * 256
        for i in range(markerpos):
            c = outbuf[i]
            posn[i] = c << 24 | count[c] & 16777215
            count[c] += 1
        for i in range(markerpos + 1, self.xsize):
            c = outbuf[i]
            posn[i] = c << 24 | count[c] & 16777215
            count[c] += 1
        last = 1
        for i in range(256):
            tmp = count[i]
            count[i] = last
            last += tmp
        i = 0
        last = self.xsize - 1
        while last > 0:
            n = posn[i]
            c = posn[i] >> 24
            last -= 1
            outbuf[last] = c
            i = count[c] + (n & 16777215)
        if i != markerpos:
            raise BZZDecoderError('BiteStream.corrupt')
        return self.xsize

    def decode_raw(self, bits):
        if False:
            return 10
        n = 1
        m = 1 << bits
        while n < m:
            b = self.zpcodec_decoder()
            n = n << 1 | b
        return n - m

    def decode_binary(self, ctx, index, bits):
        if False:
            print('Hello World!')
        n = 1
        m = 1 << bits
        while n < m:
            b = self.zpcodec_decode(ctx, index + n - 1)
            n = n << 1 | b
        return n - m

    def zpcodec_decoder(self):
        if False:
            i = 10
            return i + 15
        return self.decode_sub_simple(0, 32768 + (self.a >> 1))

    def decode_sub_simple(self, mps, z):
        if False:
            return 10
        if z > self.code:
            z = 65536 - z
            self.a += +z
            self.code = self.code + z
            shift = self.ffz()
            self.scount -= shift
            self.a = self.a << shift
            self.a &= 65535
            self.code = self.code << shift | self.bufint >> self.scount & (1 << shift) - 1
            self.code &= 65535
            if self.scount < 16:
                self.preload()
            self.fence = self.code
            if self.code >= 32768:
                self.fence = 32767
            result = mps ^ 1
        else:
            self.scount -= 1
            self.a = z << 1 & 65535
            self.code = self.code << 1 | self.bufint >> self.scount & 1
            self.code &= 65535
            if self.scount < 16:
                self.preload()
            self.fence = self.code
            if self.code >= 32768:
                self.fence = 32767
            result = mps
        return result

    def decode_sub(self, ctx, index, z):
        if False:
            for i in range(10):
                print('nop')
        bit = ctx[index] & 1
        d = 24576 + (z + self.a >> 2)
        if z > d:
            z = d
        if z > self.code:
            z = 65536 - z
            self.a += +z
            self.code = self.code + z
            ctx[index] = self.dn[ctx[index]]
            shift = self.ffz()
            self.scount -= shift
            self.a = self.a << shift & 65535
            self.code = (self.code << shift | self.bufint >> self.scount & (1 << shift) - 1) & 65535
            if self.scount < 16:
                self.preload()
            self.fence = self.code
            if self.code >= 32768:
                self.fence = 32767
            return bit ^ 1
        else:
            if self.a >= self.m[ctx[index]]:
                ctx[index] = self.up[ctx[index]]
            self.scount -= 1
            self.a = z << 1 & 65535
            self.code = (self.code << 1 | self.bufint >> self.scount & 1) & 65535
            if self.scount < 16:
                self.preload()
            self.fence = self.code
            if self.code >= 32768:
                self.fence = 32767
            return bit

    def zpcodec_decode(self, ctx, index):
        if False:
            i = 10
            return i + 15
        z = self.a + self.p[ctx[index]]
        if z <= self.fence:
            self.a = z
            res = ctx[index] & 1
        else:
            res = self.decode_sub(ctx, index, z)
        return res

    def read_byte(self):
        if False:
            while True:
                i = 10
        try:
            self.byte = self.instream[self.inptr]
            self.inptr += 1
            return True
        except IndexError:
            return False

    def ffz(self):
        if False:
            return 10
        x = self.a
        if x >= 65280:
            return self.ffzt[x & 255] + 8
        else:
            return self.ffzt[x >> 8 & 255]

def main():
    if False:
        i = 10
        return i + 15
    import sys
    from calibre_extensions import bzzdec as d
    with open(sys.argv[1], 'rb') as f:
        raw = f.read()
    print(d.decompress(raw))
if __name__ == '__main__':
    main()