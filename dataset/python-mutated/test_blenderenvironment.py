from apps.blender.blenderenvironment import BlenderEnvironment
from golem.environments.minperformancemultiplier import MinPerformanceMultiplier
from golem.model import Performance
from golem.testutils import DatabaseFixture, PEP8MixIn
from golem.tools.ci import ci_skip

@ci_skip
class BlenderEnvTest(DatabaseFixture, PEP8MixIn):
    PEP8_FILES = ['apps/blender/blenderenvironment.py']

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.env = BlenderEnvironment()

    def test_blender(self):
        if False:
            while True:
                i = 10
        'Basic environment test.'
        self.assertTrue(self.env.check_support())

    def test_get_performance(self):
        if False:
            return 10
        'Changing estimated performance in ClientConfigDescriptor.'
        assert self.env.get_benchmark_result().performance == 0.0
        fake_performance = 2345.2
        p = Performance(environment_id=BlenderEnvironment.get_id(), value=fake_performance)
        p.save()
        self.assertEqual(self.env.get_benchmark_result().performance, fake_performance)

    def test_get_min_accepted_performance_default(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(MinPerformanceMultiplier.get(), 0.0)
        self.assertEqual(self.env.get_min_accepted_performance(), 0.0)

    def test_get_min_accepted_performance(self):
        if False:
            while True:
                i = 10
        p = Performance(environment_id=BlenderEnvironment.get_id(), min_accepted_step=100)
        p.save()
        MinPerformanceMultiplier.set(3.141)
        self.assertEqual(MinPerformanceMultiplier.get(), 3.141)
        self.assertEqual(self.env.get_min_accepted_performance(), 314.1)