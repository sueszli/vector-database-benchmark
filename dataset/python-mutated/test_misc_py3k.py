import operator
from typing import cast
from sqlalchemy import Column
from sqlalchemy.testing import eq_
from sqlalchemy.testing import fixtures

class TestGenerics(fixtures.TestBase):

    def test_traversible_is_generic(self):
        if False:
            while True:
                i = 10
        'test #6759'
        col = Column[int]
        eq_(cast(object, col).__reduce__(), (operator.getitem, (Column, int)))