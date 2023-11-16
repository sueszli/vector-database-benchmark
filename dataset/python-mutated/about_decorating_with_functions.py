from runner.koan import *

class AboutDecoratingWithFunctions(Koan):

    def addcowbell(fn):
        if False:
            for i in range(10):
                print('nop')
        fn.wow_factor = 'COWBELL BABY!'
        return fn

    @addcowbell
    def mediocre_song(self):
        if False:
            print('Hello World!')
        return 'o/~ We all live in a broken submarine o/~'

    def test_decorators_can_modify_a_function(self):
        if False:
            i = 10
            return i + 15
        self.assertRegex(self.mediocre_song(), __)
        self.assertEqual(__, self.mediocre_song.wow_factor)

    def xmltag(fn):
        if False:
            return 10

        def func(*args):
            if False:
                print('Hello World!')
            return '<' + fn(*args) + '/>'
        return func

    @xmltag
    def render_tag(self, name):
        if False:
            i = 10
            return i + 15
        return name

    def test_decorators_can_change_a_function_output(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(__, self.render_tag('llama'))