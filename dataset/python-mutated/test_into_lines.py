from collisions import *

def test_sphere_into_line():
    if False:
        return 10
    entry = make_collision(CollisionSphere(0, 0, 0, 3), CollisionLine(0, 0, 0, 1, 0, 0))[0]
    assert entry is None

def test_sphere_into_ray():
    if False:
        i = 10
        return i + 15
    entry = make_collision(CollisionSphere(0, 0, 0, 3), CollisionRay(0, 0, 0, 3, 3, 3))[0]
    assert entry is None

def test_sphere_into_segment():
    if False:
        print('Hello World!')
    entry = make_collision(CollisionSphere(0, 0, 0, 3), CollisionSegment(0, 0, 0, 3, 3, 3))[0]
    assert entry is None

def test_sphere_into_parabola():
    if False:
        i = 10
        return i + 15
    parabola = LParabola((1, 0, 0), (0, 1, 0), (0, 0, 1))
    entry = make_collision(CollisionSphere(0, 0, 0, 3), CollisionParabola(parabola, 1, 2))[0]
    assert entry is None