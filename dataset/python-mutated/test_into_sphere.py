from collisions import *

def test_sphere_into_sphere():
    if False:
        for i in range(10):
            print('nop')
    sphere1 = CollisionSphere(0, 0, 3, 3)
    sphere2 = CollisionSphere(0, 0, 0, 3)
    entry = make_collision(sphere1, sphere2)[0]
    assert entry is not None
    assert entry.get_from() == sphere1
    assert entry.get_into() == sphere2
    (entry, np_from, np_into) = make_collision(CollisionSphere(0, 0, 10, 7), sphere2)
    assert entry.get_surface_point(np_from) == Point3(0, 0, 3)
    assert entry.get_surface_normal(np_into) == Vec3(0, 0, 1)
    entry = make_collision(CollisionSphere(0, 0, 10, 6), sphere2)[0]
    assert entry is None

def test_box_into_sphere():
    if False:
        print('Hello World!')
    box = CollisionBox((0, 0, 0), 2, 3, 4)
    sphere = CollisionSphere(0, 0, 0, 3)
    entry = make_collision(box, sphere)[0]
    assert entry is not None
    assert entry.get_from() == box
    assert entry.get_into() == sphere
    (entry, np_from, np_into) = make_collision(CollisionBox((0, 0, 10), 6, 6, 7), sphere)
    assert entry.get_surface_point(np_from) == Point3(0, 0, 3)
    assert entry.get_surface_normal(np_into) == Vec3(0, 0, 1)
    entry = make_collision(CollisionBox((0, 0, 10), 6, 6, 6), sphere)[0]
    assert entry is None

def test_capsule_into_sphere():
    if False:
        while True:
            i = 10
    capsule = CollisionCapsule((0, 0, 1.0), (10, 0, 1.0), 1.0)
    sphere = CollisionSphere(5, 0, 1.5, 1.0)
    entry = make_collision(capsule, sphere)[0]
    assert entry is not None
    assert entry.get_from() == capsule
    assert entry.get_into() == sphere
    entry = make_collision(CollisionCapsule((0, 0, 0), (10, 0, 0), 1.0), sphere)[0]
    assert entry is not None
    entry = make_collision(CollisionCapsule((0, 0, 0), (10, 0, 0), 0.25), sphere)[0]
    assert entry is None
    entry = make_collision(CollisionCapsule((5, 0, 0), (5, 0, 0), 1.0), sphere)[0]
    assert entry is not None
    entry = make_collision(CollisionCapsule((5, 0, 0), (5, 0, 0), 0.25), sphere)[0]
    assert entry is None

def test_segment_into_sphere():
    if False:
        return 10
    segment = CollisionSegment((0, 0, 0), (10, 0, 0))
    sphere = CollisionSphere(5, 0, 0.5, 1.0)
    entry = make_collision(segment, sphere)[0]
    assert entry is not None
    assert entry.get_from() == segment
    assert entry.get_into() == sphere
    entry = make_collision(CollisionSegment((0, 0, 0), (3, 0, 0)), sphere)[0]
    assert entry is None

def test_plane_into_sphere():
    if False:
        print('Hello World!')
    plane = CollisionPlane(Plane(Vec3(0, 0, 1), Point3(0, 0, 0)))
    sphere = CollisionSphere(0, 0, 0, 1)
    entry = make_collision(plane, sphere)[0]
    assert entry is None