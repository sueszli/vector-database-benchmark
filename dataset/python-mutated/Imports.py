from pybench import Test
import os
import package.submodule

class SecondImport(Test):
    version = 2.0
    operations = 5 * 5
    rounds = 40000

    def test(self):
        if False:
            return 10
        for i in xrange(self.rounds):
            import os
            import os
            import os
            import os
            import os
            import os
            import os
            import os
            import os
            import os
            import os
            import os
            import os
            import os
            import os
            import os
            import os
            import os
            import os
            import os
            import os
            import os
            import os
            import os
            import os

    def calibrate(self):
        if False:
            while True:
                i = 10
        for i in xrange(self.rounds):
            pass

class SecondPackageImport(Test):
    version = 2.0
    operations = 5 * 5
    rounds = 40000

    def test(self):
        if False:
            for i in range(10):
                print('nop')
        for i in xrange(self.rounds):
            import package
            import package
            import package
            import package
            import package
            import package
            import package
            import package
            import package
            import package
            import package
            import package
            import package
            import package
            import package
            import package
            import package
            import package
            import package
            import package
            import package
            import package
            import package
            import package
            import package

    def calibrate(self):
        if False:
            i = 10
            return i + 15
        for i in xrange(self.rounds):
            pass

class SecondSubmoduleImport(Test):
    version = 2.0
    operations = 5 * 5
    rounds = 40000

    def test(self):
        if False:
            return 10
        for i in xrange(self.rounds):
            import package.submodule
            import package.submodule
            import package.submodule
            import package.submodule
            import package.submodule
            import package.submodule
            import package.submodule
            import package.submodule
            import package.submodule
            import package.submodule
            import package.submodule
            import package.submodule
            import package.submodule
            import package.submodule
            import package.submodule
            import package.submodule
            import package.submodule
            import package.submodule
            import package.submodule
            import package.submodule
            import package.submodule
            import package.submodule
            import package.submodule
            import package.submodule
            import package.submodule

    def calibrate(self):
        if False:
            return 10
        for i in xrange(self.rounds):
            pass