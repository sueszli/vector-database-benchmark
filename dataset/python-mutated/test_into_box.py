from collisions import *

def test_sphere_into_box():
    if False:
        i = 10
        return i + 15
    sphere = CollisionSphere(0, 0, 4, 3)
    box = CollisionBox((0, 0, 0), 2, 3, 4)
    entry = make_collision(sphere, box)[0]
    assert entry is not None
    assert entry.get_from() == sphere
    assert entry.get_into() == box
    (entry, np_from, np_into) = make_collision(CollisionSphere(0, 0, 10, 6), box)
    assert entry.get_surface_point(np_from) == Point3(0, 0, 4)
    assert entry.get_surface_normal(np_into) == Vec3(0, 0, 1)
    (entry, np_from, np_into) = make_collision(CollisionSphere(0, 0, 0, 1), box)
    assert entry is not None
    (entry, np_from, np_into) = make_collision(CollisionSphere(1.5, 0, 0, 1), box)
    assert entry is not None
    (entry, np_from, np_into) = make_collision(CollisionSphere(2, 0, 0, 1), box)
    assert entry is not None
    (entry, np_from, np_into) = make_collision(CollisionSphere(2.5, 0, 0, 1), box)
    assert entry is not None
    (entry, np_from, np_into) = make_collision(CollisionSphere(3.5, 0, 0, 1), box)
    assert entry is None
    entry = make_collision(CollisionSphere(100, 100, 100, 100), box)[0]
    assert entry is None

def test_plane_into_box():
    if False:
        while True:
            i = 10
    plane = CollisionPlane(Plane(Vec3(0, 0, 1), Point3(0, 0, 0)))
    box = CollisionBox((0, 0, 0), 2, 3, 4)
    entry = make_collision(plane, box)[0]
    assert entry is None

def test_ray_into_box():
    if False:
        print('Hello World!')
    ray = CollisionRay(1, 1, 1, 0, 1, 0)
    box = CollisionBox((0, 0, 0), 3, 3, 5)
    entry = make_collision(ray, box)[0]
    assert entry is not None
    assert entry.get_from() == ray
    assert entry.get_into() == box
    (entry, np_from, np_into) = make_collision(CollisionRay(3, 3, 0, 1, -1, 0), box)
    assert entry.get_surface_point(np_from) == Point3(3, 3, 0)
    entry = make_collision(CollisionRay(0, 0, 100, 1, 0, 0), box)[0]
    assert entry is None

def test_parabola_into_box():
    if False:
        for i in range(10):
            print('nop')
    parabola = CollisionParabola()
    parabola.set_t1(0)
    parabola.set_t2(2)
    box = CollisionBox((0, 0, 0), 3, 3, 3)
    parabola.set_parabola(LParabola((-1, 0, -1), (1, 0, 1), (1, 1, 1)))
    entry = make_collision(parabola, box)[0]
    assert entry is None
    parabola.set_parabola(LParabola((0, 0, 1), (0, 0, 1), (1, 1, 1)))
    assert parabola.get_parabola().calc_point(1) == (1, 1, 3)
    (entry, np_from, into) = make_collision(parabola, box)
    assert entry.get_surface_point(np_from) == (1, 1, 3)
    assert entry.get_from() == parabola
    assert entry.get_into() == box
    parabola.set_parabola(LParabola((0, 0, 0), (0, 0, 1), (-3, 0, -3)))
    (entry, np_from, np_into) = make_collision(parabola, box)
    assert entry.get_surface_point(np_from) == (-3, 0, -3)
    parabola.set_parabola(LParabola((0, 0, 0), (0, 0, 1), (-5, 0, 0)))
    entry = make_collision(parabola, box)[0]
    assert entry is None
    parabola.set_parabola(LParabola((-2, -2, -2), (1, 1, 1), (4, 4, 4)))
    assert parabola.get_parabola().calc_point(1) == (3, 3, 3)
    (entry, np_from, into) = make_collision(parabola, box)
    assert entry.get_surface_point(np_from) == (3, 3, 3)
    parabola.set_parabola(LParabola((1, 1, 1), (1, 1, 1), (3, 3, 3)))
    (entry, np_from, np_into) = make_collision(parabola, box)
    assert entry.get_surface_point(np_from) == (3, 3, 3)
    parabola.set_parabola(LParabola((1, 0, 1), (-1, 0, -1), (-5, -3, -5)))
    assert parabola.get_parabola().calc_point(2) == (-3, -3, -3)
    (entry, np_from, np_into) = make_collision(parabola, box)
    assert entry.get_surface_point(np_from) == (-3, -3, -3)
    parabola.set_parabola(LParabola((-1, -1, -1), (-1, -1, -1), (3, 3, 3)))
    (entry, np_from, np_into) = make_collision(parabola, box)
    assert entry.get_surface_point(np_from) == parabola.get_parabola().calc_point(0)
    parabola.set_t1(1)
    (entry, np_from, np_into) = make_collision(parabola, box)
    assert parabola.get_parabola().calc_point(2) == (-3, -3, -3)
    assert entry.get_surface_point(np_from) == parabola.get_parabola().calc_point(2)
    assert entry.get_surface_normal(np_from) is not None