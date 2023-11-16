from .base import GEOSBase
from .prototypes import prepared as capi

class PreparedGeometry(GEOSBase):
    """
    A geometry that is prepared for performing certain operations.
    At the moment this includes the contains covers, and intersects
    operations.
    """
    ptr_type = capi.PREPGEOM_PTR
    destructor = capi.prepared_destroy

    def __init__(self, geom):
        if False:
            while True:
                i = 10
        self._base_geom = geom
        from .geometry import GEOSGeometry
        if not isinstance(geom, GEOSGeometry):
            raise TypeError
        self.ptr = capi.geos_prepare(geom.ptr)

    def contains(self, other):
        if False:
            return 10
        return capi.prepared_contains(self.ptr, other.ptr)

    def contains_properly(self, other):
        if False:
            for i in range(10):
                print('nop')
        return capi.prepared_contains_properly(self.ptr, other.ptr)

    def covers(self, other):
        if False:
            return 10
        return capi.prepared_covers(self.ptr, other.ptr)

    def intersects(self, other):
        if False:
            print('Hello World!')
        return capi.prepared_intersects(self.ptr, other.ptr)

    def crosses(self, other):
        if False:
            i = 10
            return i + 15
        return capi.prepared_crosses(self.ptr, other.ptr)

    def disjoint(self, other):
        if False:
            i = 10
            return i + 15
        return capi.prepared_disjoint(self.ptr, other.ptr)

    def overlaps(self, other):
        if False:
            i = 10
            return i + 15
        return capi.prepared_overlaps(self.ptr, other.ptr)

    def touches(self, other):
        if False:
            for i in range(10):
                print('nop')
        return capi.prepared_touches(self.ptr, other.ptr)

    def within(self, other):
        if False:
            for i in range(10):
                print('nop')
        return capi.prepared_within(self.ptr, other.ptr)