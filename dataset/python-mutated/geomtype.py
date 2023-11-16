from django.contrib.gis.gdal.error import GDALException

class OGRGeomType:
    """Encapsulate OGR Geometry Types."""
    wkb25bit = -2147483648
    _types = {0: 'Unknown', 1: 'Point', 2: 'LineString', 3: 'Polygon', 4: 'MultiPoint', 5: 'MultiLineString', 6: 'MultiPolygon', 7: 'GeometryCollection', 100: 'None', 101: 'LinearRing', 102: 'PointZ', 1 + wkb25bit: 'Point25D', 2 + wkb25bit: 'LineString25D', 3 + wkb25bit: 'Polygon25D', 4 + wkb25bit: 'MultiPoint25D', 5 + wkb25bit: 'MultiLineString25D', 6 + wkb25bit: 'MultiPolygon25D', 7 + wkb25bit: 'GeometryCollection25D'}
    _str_types = {v.lower(): k for (k, v) in _types.items()}

    def __init__(self, type_input):
        if False:
            return 10
        'Figure out the correct OGR Type based upon the input.'
        if isinstance(type_input, OGRGeomType):
            num = type_input.num
        elif isinstance(type_input, str):
            type_input = type_input.lower()
            if type_input == 'geometry':
                type_input = 'unknown'
            num = self._str_types.get(type_input)
            if num is None:
                raise GDALException('Invalid OGR String Type "%s"' % type_input)
        elif isinstance(type_input, int):
            if type_input not in self._types:
                raise GDALException('Invalid OGR Integer Type: %d' % type_input)
            num = type_input
        else:
            raise TypeError('Invalid OGR input type given.')
        self.num = num

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the value of the name property.'
        return self.name

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        '\n        Do an equivalence test on the OGR type with the given\n        other OGRGeomType, the short-hand string, or the integer.\n        '
        if isinstance(other, OGRGeomType):
            return self.num == other.num
        elif isinstance(other, str):
            return self.name.lower() == other.lower()
        elif isinstance(other, int):
            return self.num == other
        else:
            return False

    @property
    def name(self):
        if False:
            print('Hello World!')
        'Return a short-hand string form of the OGR Geometry type.'
        return self._types[self.num]

    @property
    def django(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the Django GeometryField for this OGR Type.'
        s = self.name.replace('25D', '')
        if s in ('LinearRing', 'None'):
            return None
        elif s == 'Unknown':
            s = 'Geometry'
        elif s == 'PointZ':
            s = 'Point'
        return s + 'Field'

    def to_multi(self):
        if False:
            i = 10
            return i + 15
        '\n        Transform Point, LineString, Polygon, and their 25D equivalents\n        to their Multi... counterpart.\n        '
        if self.name.startswith(('Point', 'LineString', 'Polygon')):
            self.num += 3