"""Fake ee module for use with testing."""
import box

class Image:

    def __init__(self, *_, **__):
        if False:
            i = 10
            return i + 15
        pass

    @classmethod
    def constant(self, *_, **__):
        if False:
            i = 10
            return i + 15
        return Image()

    def getMapId(self, *_, **__):
        if False:
            while True:
                i = 10
        return box.Box({'tile_fetcher': {'url_format': 'url-format'}})

    def updateMask(self, *_, **__):
        if False:
            print('Hello World!')
        return self

    def blend(self, *_, **__):
        if False:
            for i in range(10):
                print('nop')
        return self

    def bandNames(self, *_, **__):
        if False:
            while True:
                i = 10
        return List(['B1', 'B2'])

    def reduceRegion(self, *_, **__):
        if False:
            for i in range(10):
                print('nop')
        return Dictionary({'B1': 42, 'B2': 3.14})

class List:

    def __init__(self, items, *_, **__):
        if False:
            i = 10
            return i + 15
        self.items = items

    def getInfo(self, *_, **__):
        if False:
            i = 10
            return i + 15
        return self.items

class Dictionary:

    def __init__(self, data):
        if False:
            print('Hello World!')
        self.data = data

    def getInfo(self):
        if False:
            i = 10
            return i + 15
        return self.data

class ReduceRegionResult:

    def getInfo(self):
        if False:
            return 10
        return

class Geometry:
    geometry = None

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        if len(args):
            self.geometry = args[0]
        if kwargs.get('type'):
            self.geom_type = kwargs.get('type')

    @classmethod
    def Point(self, *_, **__):
        if False:
            i = 10
            return i + 15
        return Geometry(type=String('Point'))

    @classmethod
    def BBox(self, *_, **__):
        if False:
            print('Hello World!')
        return Geometry(type=String('BBox'))

    @classmethod
    def Polygon(self, *_, **__):
        if False:
            for i in range(10):
                print('nop')
        return Geometry(type=String('Polygon'))

    def transform(self, *_, **__):
        if False:
            for i in range(10):
                print('nop')
        return Geometry(type=self.geom_type)

    def bounds(self, *_, **__):
        if False:
            return 10
        return Geometry.Polygon()

    def centroid(self, *_, **__):
        if False:
            i = 10
            return i + 15
        return Geometry.Point()

    def type(self, *_, **__):
        if False:
            for i in range(10):
                print('nop')
        return self.geom_type

    def getInfo(self, *_, **__):
        if False:
            return 10
        if self.type().value == 'Polygon':
            return {'geodesic': False, 'type': 'Polygon', 'coordinates': [[[-178, -76], [179, -76], [179, 80], [-178, 80], [-178, -76]]]}
        if self.type().value == 'Point':
            return {'geodesic': False, 'type': 'Point', 'coordinates': [120, -70]}
        raise ValueError('Unexpected geometry type in test: ', self.type().value)

    def __eq__(self, other: object):
        if False:
            while True:
                i = 10
        return self.geometry == getattr(other, 'geometry')

class String:

    def __init__(self, value):
        if False:
            return 10
        self.value = value

    def compareTo(self, other_str):
        if False:
            while True:
                i = 10
        return self.value == other_str.value

    def getInfo(self, *_, **__):
        if False:
            return 10
        return self.value

class FeatureCollection:
    features = []

    def __init__(self, *args, **_):
        if False:
            while True:
                i = 10
        if len(args):
            self.features = args[0]

    def style(self, *_, **__):
        if False:
            while True:
                i = 10
        return Image()

    def first(self, *_, **__):
        if False:
            return 10
        return Feature()

    def filterBounds(self, *_, **__):
        if False:
            for i in range(10):
                print('nop')
        return FeatureCollection()

    def geometry(self, *_, **__):
        if False:
            i = 10
            return i + 15
        return Geometry.Polygon()

    def __eq__(self, other: object):
        if False:
            while True:
                i = 10
        return self.features == getattr(other, 'features')

class Feature:
    feature = None
    properties = None

    def __init__(self, *args, **_):
        if False:
            for i in range(10):
                print('nop')
        if len(args) > 0:
            self.feature = args[0]
        if len(args) >= 2:
            self.properties = args[1]

    def geometry(self, *_, **__):
        if False:
            i = 10
            return i + 15
        return Geometry(type=String('Polygon'))

    def getInfo(self, *_, **__):
        if False:
            return 10
        return {'type': 'Feature', 'geometry': {'type': 'LineString', 'coordinates': [[-67.1, 46.2], [-67.3, 46.4], [-67.5, 46.6]]}, 'id': '00000000000000000001', 'properties': {'fullname': '', 'linearid': '110469267091', 'mtfcc': 'S1400', 'rttyp': ''}}

    def __eq__(self, other: object):
        if False:
            print('Hello World!')
        featuresEqual = self.feature == getattr(other, 'feature')
        propertiesEqual = self.properties == getattr(other, 'properties')
        return featuresEqual and propertiesEqual

class ImageCollection:

    def __init__(self, *_, **__):
        if False:
            return 10
        pass

    def mosaic(self, *_, **__):
        if False:
            for i in range(10):
                print('nop')
        return Image()

class Reducer:

    @classmethod
    def first(cls, *_, **__):
        if False:
            i = 10
            return i + 15
        return Reducer()

class Algorithms:

    @classmethod
    def If(cls, *_, **__):
        if False:
            i = 10
            return i + 15
        return Algorithms()