import os
from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.profiler import option_builder
from tensorflow.python.profiler import profile_context
from tensorflow.python.profiler.internal import model_analyzer_testlib as lib
builder = option_builder.ProfileOptionBuilder

class ProfilerContextTest(test.TestCase):

    @test_util.run_deprecated_v1
    def testBasics(self):
        if False:
            return 10
        ops.reset_default_graph()
        outfile = os.path.join(test.get_temp_dir(), 'dump')
        opts = builder(builder.time_and_memory()).with_file_output(outfile).build()
        x = lib.BuildFullModel()
        profile_str = None
        profile_step100 = os.path.join(test.get_temp_dir(), 'profile_100')
        with profile_context.ProfileContext(test.get_temp_dir()) as pctx:
            pctx.add_auto_profiling('op', options=opts, profile_steps=[15, 50, 100])
            with session.Session() as sess:
                self.evaluate(variables.global_variables_initializer())
                total_steps = 101
                for i in range(total_steps):
                    self.evaluate(x)
                    if i == 14 or i == 49:
                        self.assertTrue(gfile.Exists(outfile))
                        gfile.Remove(outfile)
                    if i == 99:
                        self.assertTrue(gfile.Exists(profile_step100))
                        with gfile.Open(outfile, 'r') as f:
                            profile_str = f.read()
                        gfile.Remove(outfile)
            self.assertEqual(set([15, 50, 100]), set(pctx.get_profiles('op').keys()))
        with lib.ProfilerFromFile(os.path.join(test.get_temp_dir(), 'profile_100')) as profiler:
            profiler.profile_operations(options=opts)
            with gfile.Open(outfile, 'r') as f:
                self.assertEqual(profile_str, f.read())

    @test_util.run_deprecated_v1
    def testAutoTracingInDeubMode(self):
        if False:
            while True:
                i = 10
        ops.reset_default_graph()
        x = lib.BuildFullModel()
        with profile_context.ProfileContext(test.get_temp_dir(), debug=True):
            with session.Session() as sess:
                self.evaluate(variables.global_variables_initializer())
                for _ in range(10):
                    self.evaluate(x)
                    for f in gfile.ListDirectory(test.get_temp_dir()):
                        self.assertFalse('run_meta' in f)
                self.evaluate(x)
                self.assertTrue(gfile.Exists(os.path.join(test.get_temp_dir(), 'run_meta_11')))
                gfile.Remove(os.path.join(test.get_temp_dir(), 'run_meta_11'))
                self.evaluate(x)
                for f in gfile.ListDirectory(test.get_temp_dir()):
                    self.assertFalse('run_meta' in f)

    @test_util.run_deprecated_v1
    def testDisabled(self):
        if False:
            while True:
                i = 10
        ops.reset_default_graph()
        x = lib.BuildFullModel()
        with profile_context.ProfileContext(test.get_temp_dir(), enabled=False) as pctx:
            with session.Session() as sess:
                self.evaluate(variables.global_variables_initializer())
                for _ in range(10):
                    self.evaluate(x)
            self.assertTrue(pctx.profiler is None)
            self.assertTrue(getattr(session.BaseSession, 'profile_context', None) is None)
        with profile_context.ProfileContext(test.get_temp_dir()) as pctx:
            with session.Session() as sess:
                self.evaluate(variables.global_variables_initializer())
                for _ in range(10):
                    self.evaluate(x)
            self.assertFalse(pctx.profiler is None)
            self.assertFalse(getattr(session.BaseSession, 'profile_context', None) is None)
if __name__ == '__main__':
    test.main()