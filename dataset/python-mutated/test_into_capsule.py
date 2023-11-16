from collisions import *

def test_box_into_capsule():
    if False:
        return 10
    capsule = CollisionCapsule((1, 0, 0), (-1, 0, 0), 0.5)
    box = CollisionBox((0, 1, 0), 0.5, 0.5, 0.5)
    entry = make_collision(box, capsule)[0]
    assert entry is not None
    box = CollisionBox((0, 1.1, 0), 0.5, 0.5, 0.5)
    entry = make_collision(box, capsule)[0]
    assert entry is None
    box = CollisionBox((0, 0.8, 0), 0.5, 0.5, 0.5)
    entry = make_collision(box, capsule)[0]
    assert entry is not None
    box = CollisionBox((2, 0, 0), 0.5, 0.5, 0.5)
    entry = make_collision(box, capsule)[0]
    assert entry is not None
    box = CollisionBox((2.01, 0, 0), 0.5, 0.5, 0.5)
    entry = make_collision(box, capsule)[0]
    assert entry is None
    box = CollisionBox((-2, 0, 0), 0.5, 0.5, 0.5)
    entry = make_collision(box, capsule)[0]
    assert entry is not None
    box = CollisionBox((-2.01, 0, 0), 0.5, 0.5, 0.5)
    entry = make_collision(box, capsule)[0]
    assert entry is None