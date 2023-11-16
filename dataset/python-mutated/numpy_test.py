"""
Copyright 2013 Steven Diamond

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import abc
import numpy

class Meta:

    def __subclasscheck__(self, subclass):
        if False:
            print('Hello World!')
        print('hello')

    def __array_finalize__(self, obj):
        if False:
            print('Hello World!')
        return 1

class Test(numpy.ndarray):

    def __init__(self, shape) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    def __coerce__(self, other):
        if False:
            print('Hello World!')
        print(other)
        return (self, self)

    def __radd__(self, other):
        if False:
            i = 10
            return i + 15
        print(other)

    def __getattribute__(self, name):
        if False:
            for i in range(10):
                print('nop')
        import pdb
        pdb.set_trace()
        if name in self.__dict__:
            return self.__dict__[name]
        else:
            raise AttributeError("'Test' object has no attribute 'affa'")
if __name__ == '__main__':
    print(issubclass(Test, Meta))
    print(issubclass(Meta, numpy.ndarray))
    print(issubclass(Test, numpy.ndarray))
    print(issubclass(numpy.ndarray, Test))
    a = numpy.arange(2)
    t = Test(1)
    a + t
    import pdb
    pdb.set_trace()