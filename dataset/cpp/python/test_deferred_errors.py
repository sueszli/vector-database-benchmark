# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import pytest

from hypothesis import find, given, strategies as st
from hypothesis.errors import InvalidArgument
from hypothesis.strategies._internal.core import defines_strategy


def test_does_not_error_on_initial_calculation():
    st.floats(max_value=float("nan"))
    st.sampled_from([])
    st.lists(st.integers(), min_size=5, max_size=2)
    st.floats(min_value=2.0, max_value=1.0)


def test_errors_each_time():
    s = st.integers(max_value=1, min_value=3)
    with pytest.raises(InvalidArgument):
        s.example()
    with pytest.raises(InvalidArgument):
        s.example()


def test_errors_on_test_invocation():
    @given(st.integers(max_value=1, min_value=3))
    def test(x):
        pass

    with pytest.raises(InvalidArgument):
        test()


def test_errors_on_find():
    s = st.lists(st.integers(), min_size=5, max_size=2)
    with pytest.raises(InvalidArgument):
        find(s, lambda x: True)


def test_errors_on_example():
    s = st.floats(min_value=2.0, max_value=1.0)
    with pytest.raises(InvalidArgument):
        s.example()


def test_does_not_recalculate_the_strategy():
    calls = [0]

    @defines_strategy()
    def foo():
        calls[0] += 1
        return st.just(1)

    f = foo()
    assert calls == [0]
    f.example()
    assert calls == [1]
    f.example()
    assert calls == [1]
