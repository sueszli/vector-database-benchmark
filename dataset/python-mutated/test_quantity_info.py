"""Test the propagation of info on Quantity during operations."""
import copy
import numpy as np
from astropy import units as u

def assert_info_equal(a, b, ignore=set()):
    if False:
        for i in range(10):
            print('nop')
    a_info = a.info
    b_info = b.info
    for attr in (a_info.attr_names | b_info.attr_names) - ignore:
        if attr == 'unit':
            assert a_info.unit.is_equivalent(b_info.unit)
        else:
            assert getattr(a_info, attr, None) == getattr(b_info, attr, None)

def assert_no_info(a):
    if False:
        i = 10
        return i + 15
    assert 'info' not in a.__dict__

class TestQuantityInfo:

    @classmethod
    def setup_class(self):
        if False:
            for i in range(10):
                print('nop')
        self.q = u.Quantity(np.arange(1.0, 5.0), 'm/s')
        self.q.info.name = 'v'
        self.q.info.description = 'air speed of a african swallow'

    def test_copy(self):
        if False:
            for i in range(10):
                print('nop')
        q_copy1 = self.q.copy()
        assert_info_equal(q_copy1, self.q)
        q_copy2 = copy.copy(self.q)
        assert_info_equal(q_copy2, self.q)
        q_copy3 = copy.deepcopy(self.q)
        assert_info_equal(q_copy3, self.q)

    def test_slice(self):
        if False:
            return 10
        q_slice = self.q[1:3]
        assert_info_equal(q_slice, self.q)
        q_take = self.q.take([0, 1])
        assert_info_equal(q_take, self.q)

    def test_item(self):
        if False:
            i = 10
            return i + 15
        q1 = self.q[1]
        assert_no_info(q1)
        q_item = self.q.item(1)
        assert_no_info(q_item)

    def test_iter(self):
        if False:
            print('Hello World!')
        for q in self.q:
            assert_no_info(q)
        for q in iter(self.q):
            assert_no_info(q)

    def test_change_to_equivalent_unit(self):
        if False:
            for i in range(10):
                print('nop')
        q1 = self.q.to(u.km / u.hr)
        assert_info_equal(q1, self.q)
        q2 = self.q.si
        assert_info_equal(q2, self.q)
        q3 = self.q.cgs
        assert_info_equal(q3, self.q)
        q4 = self.q.decompose()
        assert_info_equal(q4, self.q)

    def test_reshape(self):
        if False:
            i = 10
            return i + 15
        q = self.q.reshape(-1, 1, 2)
        assert_info_equal(q, self.q)
        q2 = q.squeeze()
        assert_info_equal(q2, self.q)

    def test_insert(self):
        if False:
            i = 10
            return i + 15
        q = self.q.copy()
        q.insert(1, 1 * u.cm / u.hr)
        assert_info_equal(q, self.q)

    def test_unary_op(self):
        if False:
            print('Hello World!')
        q = -self.q
        assert_no_info(q)

    def test_binary_op(self):
        if False:
            for i in range(10):
                print('nop')
        q = self.q + self.q
        assert_no_info(q)

    def test_unit_change(self):
        if False:
            return 10
        q = self.q * u.s
        assert_no_info(q)
        q2 = u.s / self.q
        assert_no_info(q)

    def test_inplace_unit_change(self):
        if False:
            while True:
                i = 10
        q = self.q.copy()
        q *= u.s
        assert_info_equal(q, self.q, ignore={'unit'})

class TestStructuredQuantity:

    @classmethod
    def setup_class(self):
        if False:
            while True:
                i = 10
        value = np.array([(1.0, 2.0), (3.0, 4.0)], dtype=[('p', 'f8'), ('v', 'f8')])
        self.q = u.Quantity(value, 'm, m/s')
        self.q.info.name = 'pv'
        self.q.info.description = 'Location and speed'

    def test_keying(self):
        if False:
            for i in range(10):
                print('nop')
        q_p = self.q['p']
        assert_no_info(q_p)

    def test_slicing(self):
        if False:
            print('Hello World!')
        q = self.q[:1]
        assert_info_equal(q, self.q)

    def test_item(self):
        if False:
            return 10
        q = self.q[1]
        assert_no_info(q)

class TestQuantitySubclass:
    """Regression test for gh-14514: _new_view should __array_finalize__.

    But info should be propagated only for slicing, etc.
    """

    @classmethod
    def setup_class(self):
        if False:
            return 10

        class MyQuantity(u.Quantity):

            def __array_finalize__(self, obj):
                if False:
                    for i in range(10):
                        print('nop')
                super().__array_finalize__(obj)
                if hasattr(obj, 'swallow'):
                    self.swallow = obj.swallow
        self.my_q = MyQuantity([10.0, 20.0], u.m / u.s)
        self.my_q.swallow = 'African'
        self.my_q_w_info = self.my_q.copy()
        self.my_q_w_info.info.name = 'swallow'

    def test_setup(self):
        if False:
            i = 10
            return i + 15
        assert_no_info(self.my_q)
        assert self.my_q_w_info.swallow == self.my_q.swallow
        assert self.my_q_w_info.info.name == 'swallow'

    def test_slice(self):
        if False:
            while True:
                i = 10
        slc1 = self.my_q[:1]
        assert slc1.swallow == self.my_q.swallow
        assert_no_info(slc1)
        slc2 = self.my_q_w_info[1:]
        assert slc2.swallow == self.my_q.swallow
        assert_info_equal(slc2, self.my_q_w_info)

    def test_op(self):
        if False:
            while True:
                i = 10
        square1 = self.my_q ** 2
        assert square1.swallow == self.my_q.swallow
        assert_no_info(square1)
        square2 = self.my_q_w_info ** 2
        assert square2.swallow == self.my_q.swallow
        assert_no_info(square2)