from flatc_test import *

class TsTests:

    def Base(self):
        if False:
            while True:
                i = 10
        flatc(['--ts', 'foo.fbs'])
        assert_file_and_contents('foo_generated.ts', ["export { Bar } from './bar.js';", "export { Foo } from './foo.js';"])
        assert_file_and_contents('foo.ts', 'export class Foo {')
        assert_file_doesnt_exists('bar.ts')

    def BaseMultipleFiles(self):
        if False:
            print('Hello World!')
        flatc(['--ts', 'foo.fbs', 'bar/bar.fbs'])
        assert_file_and_contents('foo_generated.ts', ["export { Bar } from './bar.js';", "export { Foo } from './foo.js';"])
        assert_file_and_contents('foo.ts', 'export class Foo {')
        assert_file_and_contents('bar.ts', 'export class Bar {')

    def BaseWithNamespace(self):
        if False:
            for i in range(10):
                print('nop')
        flatc(['--ts', 'foo_with_ns.fbs'])
        assert_file_and_contents('foo_with_ns_generated.ts', ["export { Bar } from './bar/bar.js';", "export { Foo } from './something/foo.js';"])
        assert_file_and_contents('something/foo.ts', ['export class Foo {', "import { Bar } from '../bar/bar.js';"])
        assert_file_doesnt_exists('bar.ts')

    def GenAll(self):
        if False:
            while True:
                i = 10
        flatc(['--ts', '--gen-all', 'foo.fbs'])
        assert_file_and_contents('foo_generated.ts', ["export { Bar } from './bar.js'", "export { Baz } from './baz.js'", "export { Foo } from './foo.js'"])
        assert_file_and_contents('foo.ts', ["import { Bar } from './bar.js';", 'export class Foo {'])
        assert_file_and_contents('bar.ts', ["import { Baz } from './baz.js';", 'export class Bar {'])
        assert_file_and_contents('baz.ts', ['export enum Baz {'])

    def FlatFiles(self):
        if False:
            print('Hello World!')
        flatc(['--ts', '--ts-flat-files', 'foo.fbs'])
        assert_file_and_contents('foo_generated.ts', ["import {Bar as Bar} from './bar_generated.js';", 'export class Foo {'])
        assert_file_doesnt_exists('foo.ts')

    def FlatFilesWithNamespace(self):
        if False:
            i = 10
            return i + 15
        flatc(['--ts', '--ts-flat-files', 'foo_with_ns.fbs'])
        assert_file_and_contents('foo_with_ns_generated.ts', ["import {Bar as Bar} from './bar_with_ns_generated.js';", 'export class Foo {'])
        assert_file_doesnt_exists('foo.ts')

    def FlatFilesMultipleFiles(self):
        if False:
            i = 10
            return i + 15
        flatc(['--ts', '--ts-flat-files', 'foo.fbs', 'bar/bar.fbs'])
        assert_file_and_contents('foo_generated.ts', ["import {Bar as Bar} from './bar_generated.js';", 'export class Foo {'])
        assert_file_and_contents('bar_generated.ts', ["import {Baz as Baz} from './baz_generated.js';", 'export class Bar {'])
        assert_file_doesnt_exists('foo.ts')
        assert_file_doesnt_exists('bar.ts')

    def FlatFilesGenAll(self):
        if False:
            for i in range(10):
                print('nop')
        flatc(['--ts', '--ts-flat-files', '--gen-all', 'foo.fbs'])
        assert_file_and_contents('foo_generated.ts', ['export class Foo {', 'export class Bar {', 'export enum Baz {'], doesnt_contain=['import {Bar as Bar}', 'import {Baz as Baz}'])
        assert_file_doesnt_exists('foo.ts')
        assert_file_doesnt_exists('bar.ts')
        assert_file_doesnt_exists('baz.ts')

    def ZFlatFilesGenAllWithNamespacing(self):
        if False:
            for i in range(10):
                print('nop')
        flatc(['--ts', '--ts-flat-files', '--gen-all', 'foo_with_ns.fbs'])
        assert_file_and_contents('foo_with_ns_generated.ts', ['export class bar_Bar {', 'export class bar_Foo {', 'export enum Baz {', 'export enum baz_Baz {', 'export class something_Foo {'], doesnt_contain=['import {Bar as Bar}', 'import {Baz as Baz}'])
        assert_file_doesnt_exists('foo.ts')
        assert_file_doesnt_exists('bar.ts')
        assert_file_doesnt_exists('baz.ts')