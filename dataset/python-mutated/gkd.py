from . import ifd, tags, values, __geotiff__, __PY3__
import collections
GeoKeyModel = {33550: collections.namedtuple('ModelPixelScale', 'ScaleX, ScaleY, ScaleZ'), 33922: collections.namedtuple('ModelTiepoint', 'I,J,K,X,Y,Z'), 34264: collections.namedtuple('ModelTransformation', 'a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p')}

def Transform(obj, x=0.0, y=0.0, z1=0.0, z2=1.0):
    if False:
        i = 10
        return i + 15
    return (obj[0] * x + obj[1] * y + obj[2] * z1 + obj[3] * z2, obj[4] * x + obj[5] * y + obj[6] * z1 + obj[7] * z2, obj[8] * x + obj[9] * y + obj[10] * z1 + obj[11] * z2, obj[12] * x + obj[13] * y + obj[14] * z1 + obj[15] * z2)
_TAGS = {1024: ('GTModelTypeGeoKey', [3], 0, None), 1025: ('GTRasterTypeGeoKey', [3], 1, None), 1026: ('GTCitationGeoKey', [2], None, None), 2048: ('GeographicTypeGeoKey', [3], 4326, None), 2049: ('GeogCitationGeoKey', [2], None, None), 2050: ('GeogGeodeticDatumGeoKey', [3], None, None), 2051: ('GeogPrimeMeridianGeoKey', [3], 8901, None), 2052: ('GeogLinearUnitsGeoKey', [3], 9001, None), 2053: ('GeogLinearUnitSizeGeoKey', [12], None, None), 2054: ('GeogAngularUnitsGeoKey', [3], 9101, None), 2055: ('GeogAngularUnitsSizeGeoKey', [12], None, None), 2056: ('GeogEllipsoidGeoKey', [3], None, None), 2057: ('GeogSemiMajorAxisGeoKey', [12], None, None), 2058: ('GeogSemiMinorAxisGeoKey', [12], None, None), 2059: ('GeogInvFlatteningGeoKey', [12], None, None), 2060: ('GeogAzimuthUnitsGeoKey', [3], None, None), 2061: ('GeogPrimeMeridianLongGeoKey', [12], None, None), 3072: ('ProjectedCSTypeGeoKey', [3], None, None), 3073: ('PCSCitationGeoKey', [2], None, None), 3074: ('ProjectionGeoKey', [3], None, None), 3075: ('ProjCoordTransGeoKey', [3], None, None), 3076: ('ProjLinearUnitsGeoKey', [3], None, None), 3077: ('ProjLinearUnitSizeGeoKey', [12], None, None), 3078: ('ProjStdParallel1GeoKey', [12], None, None), 3079: ('ProjStdParallel2GeoKey', [12], None, None), 3080: ('ProjNatOriginLongGeoKey', [12], None, None), 3081: ('ProjNatOriginLatGeoKey', [12], None, None), 3082: ('ProjFalseEastingGeoKey', [12], None, None), 3083: ('ProjFalseNorthingGeoKey', [12], None, None), 3084: ('ProjFalseOriginLongGeoKey', [12], None, None), 3085: ('ProjFalseOriginLatGeoKey', [12], None, None), 3086: ('ProjFalseOriginEastingGeoKey', [12], None, None), 3087: ('ProjFalseOriginNorthingGeoKey', [12], None, None), 3088: ('ProjCenterLongGeoKey', [12], None, None), 3089: ('ProjCenterLatGeoKey', [12], None, None), 3090: ('ProjCenterEastingGeoKey', [12], None, None), 3091: ('ProjFalseOriginNorthingGeoKey', [12], None, None), 3092: ('ProjScaleAtNatOriginGeoKey', [12], None, None), 3093: ('ProjScaleAtCenterGeoKey', [12], None, None), 3094: ('ProjAzimuthAngleGeoKey', [12], None, None), 3095: ('ProjStraightVertPoleLongGeoKey', [12], None, None), 4096: ('VerticalCSTypeGeoKey', [3], None, None), 4097: ('VerticalCitationGeoKey', [2], None, None), 4098: ('VerticalDatumGeoKey', [3], None, None), 4099: ('VerticalUnitsGeoKey', [3], None, None)}
_2TAG = dict(((v[0], t) for (t, v) in _TAGS.items()))
_2KEY = dict(((v, k) for (k, v) in _2TAG.items()))
if __PY3__:
    import functools
    reduce = functools.reduce
    long = int

class GkdTag(ifd.TiffTag):
    strict = True

    def __init__(self, tag=0, value=None, name='GeoTiff Tag'):
        if False:
            for i in range(10):
                print('nop')
        self.name = name
        if tag == 0:
            return
        (self.key, types, default, self.comment) = _TAGS.get(tag, ('Unknown', [0], None, 'Undefined tag'))
        value = default if value == None else value
        self.tag = tag
        restricted = getattr(values, self.key, {})
        if restricted:
            reverse = dict(((v, k) for (k, v) in restricted.items()))
            if value in restricted:
                self.meaning = restricted.get(value)
            elif value in reverse:
                value = reverse[value]
                self.meaning = value
            elif GkdTag.strict:
                raise ValueError('"%s" value must be one of %s, get %s instead' % (self.key, list(restricted.keys()), value))
        (self.type, self.count, self.value) = self._encode(value, types)

    def __setattr__(self, attr, value):
        if False:
            print('Hello World!')
        object.__setattr__(self, attr, value)

    def _encode(self, value, types):
        if False:
            return 10
        if isinstance(value, str):
            value = value.encode()
        elif not hasattr(value, '__len__'):
            value = (value,)
        typ = 0
        if 2 in types:
            typ = 34737
        elif 12 in types:
            typ = 34736
        return (typ, len(value), value)

    def _decode(self):
        if False:
            return 10
        if self.count == 1:
            return self.value[0]
        else:
            return self.value

class Gkd(dict):
    tagname = 'Geotiff Tag'
    version = __geotiff__[0]
    revision = __geotiff__[1:]

    def __init__(self, value={}, **pairs):
        if False:
            for i in range(10):
                print('nop')
        dict.__init__(self)
        self.from_ifd(value, **pairs)

    def __getitem__(self, tag):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(tag, str):
            tag = _2TAG[tag]
        return dict.__getitem__(self, tag)._decode()

    def __setitem__(self, tag, value):
        if False:
            i = 10
            return i + 15
        if isinstance(tag, str):
            tag = _2TAG[tag]
        dict.__setitem__(self, tag, GkdTag(tag, value, name=self.tagname))

    def get(self, tag, error=None):
        if False:
            for i in range(10):
                print('nop')
        if hasattr(self, '_%s' % tag):
            return getattr(self, '_%s' % tag)
        else:
            return dict.get(self, tag, error)

    def to_ifd(self):
        if False:
            return 10
        (_34735, _34736, _34737, nbkey, _ifd) = ((), (), b'', 0, {})
        for (key, tag) in sorted(self.items(), key=lambda a: a[0]):
            if tag.type == 0:
                _34735 += (key, 0, 1) + tag.value
                nbkey += 1
            elif tag.type == 34736:
                _34735 += (key, 34736, 1, len(_34736))
                _34736 += tag.value
                nbkey += 1
            elif tag.type == 34737:
                _34735 += (key, 34737, tag.count + 1, len(_34737))
                _34737 += tag.value + b'|'
                nbkey += 1
        result = ifd.Ifd()
        result.set(33922, 12, reduce(tuple.__add__, [tuple(e) for e in self.get(33922, ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0],))]))
        result.set(33550, 12, tuple(self.get(33550, (1.0, 1.0, 1.0))))
        result.set(34264, 12, tuple(self.get(34264, (1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0))))
        result.set(34735, 3, (self.version,) + self.revision + (nbkey,) + _34735)
        result.set(34736, 12, _34736)
        result.set(34737, 2, _34737)
        return result

    def from_ifd(self, ifd={}, **kw):
        if False:
            print('Hello World!')
        pairs = dict(ifd, **kw)
        for tag in [t for t in [33922, 33550, 34264] if t in pairs]:
            nt = GeoKeyModel[tag]
            if tag == 33922:
                n = len(nt._fields)
                seq = ifd[tag]
                setattr(self, '_%s' % tag, tuple((nt(*seq[i:i + n]) for i in range(0, len(seq), n))))
            else:
                setattr(self, '_%s' % tag, nt(*ifd[tag]))
        if 34736 in pairs:
            _34736 = ifd[34736]
        if 34737 in pairs:
            _34737 = ifd[34737]
        if 34735 in pairs:
            _34735 = ifd[34735]
            self.version = _34735[0]
            self.revision = _34735[1:3]
            for (tag, typ, count, value) in zip(_34735[4::4], _34735[5::4], _34735[6::4], _34735[7::4]):
                if typ == 0:
                    self[tag] = value
                elif typ == 34736:
                    self[tag] = _34736[value]
                elif typ == 34737:
                    self[tag] = _34737[value:value + count - 1]

    def getModelTransformation(self, tie_index=0):
        if False:
            for i in range(10):
                print('nop')
        if hasattr(self, '_34264'):
            matrix = GeoKeyModel[34264](*getattr(self, '_34264'))
        elif hasattr(self, '_33922') and hasattr(self, '_33550'):
            (Sx, Sy, Sz) = getattr(self, '_33550')
            (I, J, K, X, Y, Z) = getattr(self, '_33922')[tie_index]
            matrix = GeoKeyModel[34264](Sx, 0.0, 0.0, X - I * Sx, 0.0, -Sy, 0.0, Y + J * Sy, 0.0, 0.0, Sz, Z - K * Sz, 0.0, 0.0, 0.0, 1.0)
        else:
            matrix = GeoKeyModel[34264](1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        return lambda x, y, z1=0.0, z2=1.0, m=matrix: Transform(m, x, y, z1, z2)

    def tags(self):
        if False:
            return 10
        for v in sorted(dict.values(self), key=lambda e: e.tag):
            yield v