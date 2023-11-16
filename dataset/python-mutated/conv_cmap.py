import codecs
import pickle as pickle
import sys

class CMapConverter:

    def __init__(self, enc2codec={}):
        if False:
            return 10
        self.enc2codec = enc2codec
        self.code2cid = {}
        self.is_vertical = {}
        self.cid2unichr_h = {}
        self.cid2unichr_v = {}
        return

    def get_encs(self):
        if False:
            i = 10
            return i + 15
        return self.code2cid.keys()

    def get_maps(self, enc):
        if False:
            while True:
                i = 10
        if enc.endswith('-H'):
            (hmapenc, vmapenc) = (enc, None)
        elif enc == 'H':
            (hmapenc, vmapenc) = ('H', 'V')
        else:
            (hmapenc, vmapenc) = (enc + '-H', enc + '-V')
        if hmapenc in self.code2cid:
            hmap = self.code2cid[hmapenc]
        else:
            hmap = {}
            self.code2cid[hmapenc] = hmap
        vmap = None
        if vmapenc:
            self.is_vertical[vmapenc] = True
            if vmapenc in self.code2cid:
                vmap = self.code2cid[vmapenc]
            else:
                vmap = {}
                self.code2cid[vmapenc] = vmap
        return (hmap, vmap)

    def load(self, fp):
        if False:
            while True:
                i = 10
        encs = None
        for line in fp:
            (line, _, _) = line.strip().partition('#')
            if not line:
                continue
            values = line.split('\t')
            if encs is None:
                assert values[0] == 'CID', str(values)
                encs = values
                continue

            def put(dmap, code, cid, force=False):
                if False:
                    return 10
                for b in code[:-1]:
                    if b in dmap:
                        dmap = dmap[b]
                    else:
                        d = {}
                        dmap[b] = d
                        dmap = d
                b = code[-1]
                if force or (b not in dmap or dmap[b] == cid):
                    dmap[b] = cid
                return

            def add(unimap, enc, code):
                if False:
                    while True:
                        i = 10
                try:
                    codec = self.enc2codec[enc]
                    c = code.decode(codec, 'strict')
                    if len(c) == 1:
                        if c not in unimap:
                            unimap[c] = 0
                        unimap[c] += 1
                except KeyError:
                    pass
                except UnicodeError:
                    pass
                return

            def pick(unimap):
                if False:
                    print('Hello World!')
                chars = list(unimap.items())
                chars.sort(key=lambda x: (x[1], -ord(x[0])), reverse=True)
                (c, _) = chars[0]
                return c
            cid = int(values[0])
            unimap_h = {}
            unimap_v = {}
            for (enc, value) in zip(encs, values):
                if enc == 'CID':
                    continue
                if value == '*':
                    continue
                hcodes = []
                vcodes = []
                for code in value.split(','):
                    vertical = code.endswith('v')
                    if vertical:
                        code = code[:-1]
                    try:
                        code = codecs.decode(code, 'hex_codec')
                    except Exception:
                        code = chr(int(code, 16))
                    if vertical:
                        vcodes.append(code)
                        add(unimap_v, enc, code)
                    else:
                        hcodes.append(code)
                        add(unimap_h, enc, code)
                (hmap, vmap) = self.get_maps(enc)
                if vcodes:
                    assert vmap is not None
                    for code in vcodes:
                        put(vmap, code, cid, True)
                    for code in hcodes:
                        put(hmap, code, cid, True)
                else:
                    for code in hcodes:
                        put(hmap, code, cid)
                        put(vmap, code, cid)
            if unimap_h:
                self.cid2unichr_h[cid] = pick(unimap_h)
            if unimap_v or unimap_h:
                self.cid2unichr_v[cid] = pick(unimap_v or unimap_h)
        return

    def dump_cmap(self, fp, enc):
        if False:
            for i in range(10):
                print('nop')
        data = dict(IS_VERTICAL=self.is_vertical.get(enc, False), CODE2CID=self.code2cid.get(enc))
        fp.write(pickle.dumps(data, 2))
        return

    def dump_unicodemap(self, fp):
        if False:
            i = 10
            return i + 15
        data = dict(CID2UNICHR_H=self.cid2unichr_h, CID2UNICHR_V=self.cid2unichr_v)
        fp.write(pickle.dumps(data, 2))
        return

def main(argv):
    if False:
        for i in range(10):
            print('nop')
    import getopt
    import gzip
    import os.path

    def usage():
        if False:
            i = 10
            return i + 15
        print('usage: %s [-c enc=codec] output_dir regname [cid2code.txt ...]' % argv[0])
        return 100
    try:
        (opts, args) = getopt.getopt(argv[1:], 'c:')
    except getopt.GetoptError:
        return usage()
    enc2codec = {}
    for (k, v) in opts:
        if k == '-c':
            (enc, _, codec) = v.partition('=')
            enc2codec[enc] = codec
    if not args:
        return usage()
    outdir = args.pop(0)
    if not args:
        return usage()
    regname = args.pop(0)
    converter = CMapConverter(enc2codec)
    for path in args:
        print('reading: %r...' % path)
        fp = open(path)
        converter.load(fp)
        fp.close()
    for enc in converter.get_encs():
        fname = '%s.pickle.gz' % enc
        path = os.path.join(outdir, fname)
        print('writing: %r...' % path)
        fp = gzip.open(path, 'wb')
        converter.dump_cmap(fp, enc)
        fp.close()
    fname = 'to-unicode-%s.pickle.gz' % regname
    path = os.path.join(outdir, fname)
    print('writing: %r...' % path)
    fp = gzip.open(path, 'wb')
    converter.dump_unicodemap(fp)
    fp.close()
    return
if __name__ == '__main__':
    sys.exit(main(sys.argv))