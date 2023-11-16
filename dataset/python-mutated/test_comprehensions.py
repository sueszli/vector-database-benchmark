"""
Created on Nov 6, 2011

@author: sean
"""
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import unittest
from ...decompiler.tests import Base

class ListComprehension(Base):

    def test_comp1(self):
        if False:
            print('Hello World!')
        stmnt = '[a for b in c]'
        self.statement(stmnt)

    def test_comp2(self):
        if False:
            return 10
        stmnt = '[a() +1 for b in c]'
        self.statement(stmnt)

    def test_comp3(self):
        if False:
            return 10
        stmnt = 'y = [a() +1 for b in c]'
        self.statement(stmnt)

    def test_comp_ifs(self):
        if False:
            while True:
                i = 10
        stmnt = 'y = [a() +1 for b in c if asdf]'
        self.statement(stmnt)

    def test_comp_ifs1(self):
        if False:
            i = 10
            return i + 15
        stmnt = 'y = [a() +1 for b in c if asdf if asd]'
        self.statement(stmnt)

    def test_comp_ifs2(self):
        if False:
            i = 10
            return i + 15
        stmnt = 'y = [a() +1 for b in c if asdf if not asd]'
        self.statement(stmnt)

    @unittest.expectedFailure
    def test_multi_comp1(self):
        if False:
            for i in range(10):
                print('nop')
        stmnt = '[a for b in c for d in e]'
        self.statement(stmnt)

    @unittest.expectedFailure
    def test_multi_comp2(self):
        if False:
            i = 10
            return i + 15
        stmnt = '[a() +1 for b in c for d in e]'
        self.statement(stmnt)

    @unittest.expectedFailure
    def test_multi_comp3(self):
        if False:
            for i in range(10):
                print('nop')
        stmnt = 'y = [a() +1 for b in c for d in e]'
        self.statement(stmnt)

    @unittest.expectedFailure
    def test_multi_comp_ifs(self):
        if False:
            for i in range(10):
                print('nop')
        stmnt = 'y = [a() +1 for b in c if asdf for d in e]'
        self.statement(stmnt)

    @unittest.expectedFailure
    def test_multi_comp_ifs1(self):
        if False:
            return 10
        stmnt = 'y = [a() +1 for b in c if asdf if asd for d in e if this]'
        self.statement(stmnt)

    @unittest.expectedFailure
    def test_multi_comp_ifs2(self):
        if False:
            for i in range(10):
                print('nop')
        stmnt = 'y = [a() +1 for b in c for d in e if adsf]'
        self.statement(stmnt)

class SetComprehension(Base):

    def test_comp1(self):
        if False:
            while True:
                i = 10
        stmnt = '{a for b in c}'
        self.statement(stmnt)

    def test_comp2(self):
        if False:
            return 10
        stmnt = '{a() +1 for b in c}'
        self.statement(stmnt)

    def test_comp3(self):
        if False:
            while True:
                i = 10
        stmnt = 'y = {a() +1 for b in c}'
        self.statement(stmnt)

    def test_comp_ifs(self):
        if False:
            while True:
                i = 10
        stmnt = 'y = {a() +1 for b in c if asdf}'
        self.statement(stmnt)

    def test_comp_ifs1(self):
        if False:
            return 10
        stmnt = 'y = {a() +1 for b in c if asdf if asd}'
        self.statement(stmnt)

    def test_comp_ifs2(self):
        if False:
            i = 10
            return i + 15
        stmnt = 'y = {a() +1 for b in c if asdf if not asd}'
        self.statement(stmnt)

    @unittest.expectedFailure
    def test_multi_comp1(self):
        if False:
            print('Hello World!')
        stmnt = '{a for b in c for d in e}'
        self.statement(stmnt)

    @unittest.expectedFailure
    def test_multi_comp2(self):
        if False:
            print('Hello World!')
        stmnt = '{a() +1 for b in c for d in e}'
        self.statement(stmnt)

    @unittest.expectedFailure
    def test_multi_comp3(self):
        if False:
            for i in range(10):
                print('nop')
        stmnt = 'y = {a() +1 for b in c for d in e}'
        self.statement(stmnt)

    @unittest.expectedFailure
    def test_multi_comp_ifs(self):
        if False:
            i = 10
            return i + 15
        stmnt = 'y = {a() +1 for b in c if asdf for d in e}'
        self.statement(stmnt)

    @unittest.expectedFailure
    def test_multi_comp_ifs1(self):
        if False:
            return 10
        stmnt = 'y = {a() +1 for b in c if asdf if asd for d in e if this}'
        self.statement(stmnt)

    @unittest.expectedFailure
    def test_multi_comp_ifs2(self):
        if False:
            i = 10
            return i + 15
        stmnt = 'y = {a() +1 for b in c for d in e if adsf}'
        self.statement(stmnt)

class DictComprehension(Base):

    def test_comp1(self):
        if False:
            i = 10
            return i + 15
        stmnt = '{a:q for b in c}'
        self.statement(stmnt)

    def test_comp2(self):
        if False:
            while True:
                i = 10
        stmnt = '{a() +1:q for b in c}'
        self.statement(stmnt)

    def test_comp3(self):
        if False:
            i = 10
            return i + 15
        stmnt = 'y = {a() +1:q for b in c}'
        self.statement(stmnt)

    def test_comp_ifs(self):
        if False:
            for i in range(10):
                print('nop')
        stmnt = 'y = {a() +1:q for b in c if asdf}'
        self.statement(stmnt)

    def test_comp_ifs1(self):
        if False:
            return 10
        stmnt = 'y = {a() +1:q for b in c if asdf if asd}'
        self.statement(stmnt)

    def test_comp_ifs2(self):
        if False:
            i = 10
            return i + 15
        stmnt = 'y = {a() +1:q for b in c if asdf if not asd}'
        self.statement(stmnt)

    @unittest.expectedFailure
    def test_multi_comp1(self):
        if False:
            for i in range(10):
                print('nop')
        stmnt = '{a:q for b in c for d in e}'
        self.statement(stmnt)

    @unittest.expectedFailure
    def test_multi_comp2(self):
        if False:
            print('Hello World!')
        stmnt = '{a():q +1 for b in c for d in e}'
        self.statement(stmnt)

    @unittest.expectedFailure
    def test_multi_comp3(self):
        if False:
            for i in range(10):
                print('nop')
        stmnt = 'y = {a() +1:q for b in c for d in e}'
        self.statement(stmnt)

    @unittest.expectedFailure
    def test_multi_comp_ifs(self):
        if False:
            for i in range(10):
                print('nop')
        stmnt = 'y = {a() +1:q for b in c if asdf for d in e}'
        self.statement(stmnt)

    @unittest.expectedFailure
    def test_multi_comp_ifs1(self):
        if False:
            while True:
                i = 10
        stmnt = 'y = {a() +1:q for b in c if asdf if asd for d in e if this}'
        self.statement(stmnt)

    @unittest.expectedFailure
    def test_multi_comp_ifs2(self):
        if False:
            for i in range(10):
                print('nop')
        stmnt = 'y = {a() +1:q for b in c for d in e if adsf}'
        self.statement(stmnt)
if __name__ == '__main__':
    unittest.main()