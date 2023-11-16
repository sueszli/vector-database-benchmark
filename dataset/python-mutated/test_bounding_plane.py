from panda3d.core import Plane, BoundingPlane, BoundingSphere, BoundingVolume

def test_plane_contains_sphere():
    if False:
        print('Hello World!')
    plane = BoundingPlane((0, 0, 1, 0))
    assert plane.contains(BoundingSphere((0, 0, 2), 1)) == BoundingVolume.IF_no_intersection
    assert plane.contains(BoundingSphere((0, 0, 0), 1)) == BoundingVolume.IF_possible | BoundingVolume.IF_some
    assert plane.contains(BoundingSphere((0, 0, -2), 1)) == BoundingVolume.IF_possible | BoundingVolume.IF_some | BoundingVolume.IF_all

def test_plane_contains_plane():
    if False:
        i = 10
        return i + 15
    a = BoundingPlane((1, 0, 0, 1))
    assert a.contains(a) == BoundingVolume.IF_possible | BoundingVolume.IF_some | BoundingVolume.IF_all
    a = BoundingPlane((1, 0, 0, 1))
    b = BoundingPlane((-1, 0, 0, -1))
    assert a.contains(b) == BoundingVolume.IF_no_intersection
    assert b.contains(a) == BoundingVolume.IF_no_intersection
    a = BoundingPlane(Plane((1, 0, 0), (1, 0, 0)))
    b = BoundingPlane(Plane((1, 0, 0), (2, 0, 0)))
    assert a.contains(b) == BoundingVolume.IF_possible | BoundingVolume.IF_some
    assert b.contains(a) == BoundingVolume.IF_possible | BoundingVolume.IF_some | BoundingVolume.IF_all
    a = BoundingPlane(Plane((1, 0, 0), (1, 0, 0)))
    b = BoundingPlane(Plane((-1, 0, 0), (2, 0, 0)))
    assert a.contains(b) == BoundingVolume.IF_no_intersection
    assert b.contains(a) == BoundingVolume.IF_no_intersection
    a = BoundingPlane(Plane((1, 0, 0), (2, 0, 0)))
    b = BoundingPlane(Plane((-1, 0, 0), (1, 0, 0)))
    assert a.contains(b) == BoundingVolume.IF_possible | BoundingVolume.IF_some
    assert b.contains(a) == BoundingVolume.IF_possible | BoundingVolume.IF_some
    a = BoundingPlane(Plane((1, 0, 0), (2, 0, 0)))
    b = BoundingPlane(Plane((0.8, 0.6, 0), (4, 0, 0)))
    assert a.contains(b) == BoundingVolume.IF_possible | BoundingVolume.IF_some
    assert b.contains(a) == BoundingVolume.IF_possible | BoundingVolume.IF_some
    a = BoundingPlane(Plane((1, 0, 0), (2, 0, 0)))
    b = BoundingPlane(Plane((-0.8, -0.6, 0), (4, 0, 0)))
    assert a.contains(b) == BoundingVolume.IF_possible | BoundingVolume.IF_some
    assert b.contains(a) == BoundingVolume.IF_possible | BoundingVolume.IF_some
    a = BoundingPlane(Plane((1, 0, 0, 0)))
    b = BoundingPlane(Plane((0, 1, 0, 0)))
    c = BoundingPlane(Plane((0, 0, 1, 0)))
    assert a.contains(b) == BoundingVolume.IF_possible | BoundingVolume.IF_some
    assert b.contains(c) == BoundingVolume.IF_possible | BoundingVolume.IF_some
    assert a.contains(c) == BoundingVolume.IF_possible | BoundingVolume.IF_some