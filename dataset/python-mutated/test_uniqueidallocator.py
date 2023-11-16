from panda3d.core import UniqueIdAllocator
IndexEnd = 4294967295

def test_inclusive_allocation():
    if False:
        i = 10
        return i + 15
    allocator = UniqueIdAllocator(0, 0)
    assert allocator.allocate() == 0
    assert allocator.allocate() == IndexEnd

def test_normal_allocation():
    if False:
        while True:
            i = 10
    allocator = UniqueIdAllocator(0, 10)
    for i in range(10 + 1):
        assert allocator.allocate() == i
    assert allocator.allocate() == IndexEnd

def test_min_value_allocation():
    if False:
        return 10
    allocator = UniqueIdAllocator(1, 5)
    for i in range(1, 5 + 1):
        assert allocator.allocate() == i
    assert allocator.allocate() == IndexEnd

def test_regular_is_allocated():
    if False:
        print('Hello World!')
    allocator = UniqueIdAllocator(1, 5)
    for i in range(1, 5 + 1):
        assert not allocator.is_allocated(i)
    for i in range(1, 5 + 1):
        assert allocator.is_allocated(allocator.allocate())

def test_bounded_is_allocated():
    if False:
        return 10
    allocator = UniqueIdAllocator(1, 5)
    for i in range(1, 5 + 1):
        assert allocator.is_allocated(allocator.allocate())
    assert not allocator.is_allocated(0)
    assert not allocator.is_allocated(10)

def test_initial_reserve_id():
    if False:
        return 10
    allocator = UniqueIdAllocator(1, 3)
    assert not allocator.is_allocated(2)
    allocator.initial_reserve_id(2)
    assert allocator.is_allocated(2)
    assert allocator.allocate() == 1
    assert allocator.allocate() == 3
    assert allocator.allocate() == IndexEnd

def test_initial_reserve_id_exhaustion():
    if False:
        while True:
            i = 10
    allocator = UniqueIdAllocator(1, 3)
    for i in range(1, 3 + 1):
        allocator.initial_reserve_id(i)
    assert allocator.allocate() == IndexEnd

def test_free():
    if False:
        print('Hello World!')
    allocator = UniqueIdAllocator(0, 0)
    assert allocator.allocate() == 0
    assert allocator.is_allocated(0)
    assert allocator.free(0)
    assert not allocator.is_allocated(0)

def test_free_reallocation():
    if False:
        while True:
            i = 10
    allocator = UniqueIdAllocator(1, 5)
    for i in range(1, 5 + 1):
        assert allocator.allocate() == i
        assert allocator.is_allocated(i)
    for i in range(1, 5 + 1):
        assert allocator.free(i)
    for i in range(1, 5 + 1):
        assert not allocator.is_allocated(i)
        assert allocator.allocate() == i
    assert allocator.allocate() == IndexEnd

def test_free_unallocated():
    if False:
        print('Hello World!')
    allocator = UniqueIdAllocator(0, 2)
    assert allocator.allocate() == 0
    assert allocator.free(0)
    for i in range(0, 2 + 1):
        assert not allocator.free(i)

def test_free_bounds():
    if False:
        for i in range(10):
            print('nop')
    allocator = UniqueIdAllocator(1, 3)
    assert not allocator.free(0)
    assert not allocator.free(4)

def test_free_reallocation_mid():
    if False:
        while True:
            i = 10
    allocator = UniqueIdAllocator(1, 5)
    for i in range(1, 5 + 1):
        assert allocator.allocate() == i
        assert allocator.is_allocated(i)
    assert allocator.free(2)
    assert allocator.free(3)
    assert allocator.allocate() == 2
    assert allocator.allocate() == 3
    assert allocator.allocate() == IndexEnd

def test_free_initial_reserve_id():
    if False:
        for i in range(10):
            print('nop')
    allocator = UniqueIdAllocator(1, 3)
    allocator.initial_reserve_id(1)
    assert allocator.free(1)
    assert allocator.allocate() == 2
    assert allocator.allocate() == 3
    assert allocator.allocate() == 1
    assert allocator.allocate() == IndexEnd

def test_fraction_used():
    if False:
        while True:
            i = 10
    allocator = UniqueIdAllocator(1, 4)
    assert allocator.fraction_used() == 0
    for fraction in (0.25, 0.5, 0.75, 1):
        allocator.allocate()
        assert allocator.fraction_used() == fraction
    assert allocator.allocate() == IndexEnd