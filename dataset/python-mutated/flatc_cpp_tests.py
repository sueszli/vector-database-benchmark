from flatc_test import *

class CppTests:

    def Flatten(self):
        if False:
            print('Hello World!')
        flatc(['--cpp', 'foo.fbs'])
        assert_file_and_contents('foo_generated.h', '#include "bar_generated.h"')

    def FlattenAbsolutePath(self):
        if False:
            return 10
        flatc(['--cpp', make_absolute('foo.fbs')])
        assert_file_and_contents('foo_generated.h', '#include "bar_generated.h"')

    def FlattenSubDirectory(self):
        if False:
            i = 10
            return i + 15
        flatc(['--cpp', 'bar/bar.fbs'])
        assert_file_and_contents('bar_generated.h', '#include "baz_generated.h"')

    def FlattenOutPath(self):
        if False:
            i = 10
            return i + 15
        flatc(['--cpp', '-o', '.tmp', 'foo.fbs'])
        assert_file_and_contents('.tmp/foo_generated.h', '#include "bar_generated.h"')

    def FlattenOutPathSuperDirectory(self):
        if False:
            i = 10
            return i + 15
        flatc(['--cpp', '-o', '../.tmp', 'foo.fbs'])
        assert_file_and_contents('../.tmp/foo_generated.h', '#include "bar_generated.h"')

    def FlattenOutPathSubDirectory(self):
        if False:
            return 10
        flatc(['--cpp', '-o', '.tmp', 'bar/bar.fbs'])
        assert_file_and_contents('.tmp/bar_generated.h', '#include "baz_generated.h"')

    def KeepPrefix(self):
        if False:
            for i in range(10):
                print('nop')
        flatc(['--cpp', '--keep-prefix', 'foo.fbs'])
        assert_file_and_contents('foo_generated.h', '#include "bar/bar_generated.h"')

    def KeepPrefixAbsolutePath(self):
        if False:
            return 10
        flatc(['--cpp', '--keep-prefix', make_absolute('foo.fbs')])
        assert_file_and_contents('foo_generated.h', '#include "bar/bar_generated.h"')

    def KeepPrefixSubDirectory(self):
        if False:
            return 10
        flatc(['--cpp', '--keep-prefix', 'bar/bar.fbs'])
        assert_file_and_contents('bar_generated.h', '#include "baz/baz_generated.h"')

    def KeepPrefixOutPath(self):
        if False:
            while True:
                i = 10
        flatc(['--cpp', '--keep-prefix', '-o', '.tmp', 'foo.fbs'])
        assert_file_and_contents('.tmp/foo_generated.h', '#include "bar/bar_generated.h"')

    def KeepPrefixOutPathSubDirectory(self):
        if False:
            for i in range(10):
                print('nop')
        flatc(['--cpp', '--keep-prefix', '-o', '.tmp', 'bar/bar.fbs'])
        assert_file_and_contents('.tmp/bar_generated.h', '#include "baz/baz_generated.h"')

    def IncludePrefix(self):
        if False:
            return 10
        flatc(['--cpp', '--include-prefix', 'test', 'foo.fbs'])
        assert_file_and_contents('foo_generated.h', '#include "test/bar_generated.h"')

    def IncludePrefixAbolutePath(self):
        if False:
            i = 10
            return i + 15
        flatc(['--cpp', '--include-prefix', 'test', make_absolute('foo.fbs')])
        assert_file_and_contents('foo_generated.h', '#include "test/bar_generated.h"')

    def IncludePrefixSubDirectory(self):
        if False:
            for i in range(10):
                print('nop')
        flatc(['--cpp', '--include-prefix', 'test', 'bar/bar.fbs'])
        assert_file_and_contents('bar_generated.h', '#include "test/baz_generated.h"')

    def IncludePrefixOutPath(self):
        if False:
            print('Hello World!')
        flatc(['--cpp', '--include-prefix', 'test', '-o', '.tmp', 'foo.fbs'])
        assert_file_and_contents('.tmp/foo_generated.h', '#include "test/bar_generated.h"')

    def IncludePrefixOutPathSubDirectory(self):
        if False:
            i = 10
            return i + 15
        flatc(['--cpp', '--include-prefix', 'test', '-o', '.tmp', 'bar/bar.fbs'])
        assert_file_and_contents('.tmp/bar_generated.h', '#include "test/baz_generated.h"')

    def KeepPrefixIncludePrefix(self):
        if False:
            print('Hello World!')
        flatc(['--cpp', '--keep-prefix', '--include-prefix', 'test', 'foo.fbs'])
        assert_file_and_contents('foo_generated.h', '#include "test/bar/bar_generated.h"')

    def KeepPrefixIncludePrefixAbsolutePath(self):
        if False:
            while True:
                i = 10
        flatc(['--cpp', '--keep-prefix', '--include-prefix', 'test', make_absolute('foo.fbs')])
        assert_file_and_contents('foo_generated.h', '#include "test/bar/bar_generated.h"')

    def KeepPrefixIncludePrefixSubDirectory(self):
        if False:
            i = 10
            return i + 15
        flatc(['--cpp', '--keep-prefix', '--include-prefix', 'test', 'bar/bar.fbs'])
        assert_file_and_contents('bar_generated.h', '#include "test/baz/baz_generated.h"')

    def KeepPrefixIncludePrefixOutPathSubDirectory(self):
        if False:
            print('Hello World!')
        flatc(['--cpp', '--keep-prefix', '--include-prefix', 'test', '-o', '.tmp', 'bar/bar.fbs'])
        assert_file_and_contents('.tmp/bar_generated.h', '#include "test/baz/baz_generated.h"')

    def KeepPrefixIncludePrefixOutPathSuperDirectory(self):
        if False:
            i = 10
            return i + 15
        flatc(['--cpp', '--keep-prefix', '--include-prefix', 'test', '-o', '../.tmp', 'bar/bar.fbs'])
        assert_file_and_contents('../.tmp/bar_generated.h', '#include "test/baz/baz_generated.h"')

    def KeepPrefixIncludePrefixoutPathAbsoluePaths_SuperDirectoryReference(self):
        if False:
            for i in range(10):
                print('nop')
        flatc(['--cpp', '--keep-prefix', '--include-prefix', 'generated', '-I', str(script_path.absolute()), '-o', str(Path(script_path, '.tmp').absolute()), str(Path(script_path, 'bar/bar_with_foo.fbs').absolute())])
        assert_file_and_contents('.tmp/bar_with_foo_generated.h', ['#include "generated/baz/baz_generated.h"', '#include "generated/foo_generated.h"'])

    def KeepPrefixIncludePrefixoutPath_SuperDirectoryReference(self):
        if False:
            print('Hello World!')
        flatc(['--cpp', '--keep-prefix', '--include-prefix', 'generated', '-I', './', '-o', '.tmp', 'bar/bar_with_foo.fbs'])
        assert_file_and_contents('.tmp/bar_with_foo_generated.h', ['#include "generated/baz/baz_generated.h"', '#include "generated/foo_generated.h"'], unlink=False)