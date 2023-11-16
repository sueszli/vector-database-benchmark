from caffe2.python import control, core, test_util, workspace
import logging
logger = logging.getLogger(__name__)

class TestControl(test_util.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.N_ = 10
        self.init_net_ = core.Net('init-net')
        cnt = self.init_net_.CreateCounter([], init_count=0)
        const_n = self.init_net_.ConstantFill([], shape=[], value=self.N_, dtype=core.DataType.INT64)
        const_0 = self.init_net_.ConstantFill([], shape=[], value=0, dtype=core.DataType.INT64)
        self.cnt_net_ = core.Net('cnt-net')
        self.cnt_net_.CountUp([cnt])
        curr_cnt = self.cnt_net_.RetrieveCount([cnt])
        self.init_net_.ConstantFill([], [curr_cnt], shape=[], value=0, dtype=core.DataType.INT64)
        self.cnt_net_.AddExternalOutput(curr_cnt)
        self.cnt_2_net_ = core.Net('cnt-2-net')
        self.cnt_2_net_.CountUp([cnt])
        self.cnt_2_net_.CountUp([cnt])
        curr_cnt_2 = self.cnt_2_net_.RetrieveCount([cnt])
        self.init_net_.ConstantFill([], [curr_cnt_2], shape=[], value=0, dtype=core.DataType.INT64)
        self.cnt_2_net_.AddExternalOutput(curr_cnt_2)
        self.cond_net_ = core.Net('cond-net')
        cond_blob = self.cond_net_.LT([curr_cnt, const_n])
        self.cond_net_.AddExternalOutput(cond_blob)
        self.not_cond_net_ = core.Net('not-cond-net')
        cond_blob = self.not_cond_net_.GE([curr_cnt, const_n])
        self.not_cond_net_.AddExternalOutput(cond_blob)
        self.true_cond_net_ = core.Net('true-cond-net')
        true_blob = self.true_cond_net_.LT([const_0, const_n])
        self.true_cond_net_.AddExternalOutput(true_blob)
        self.false_cond_net_ = core.Net('false-cond-net')
        false_blob = self.false_cond_net_.GT([const_0, const_n])
        self.false_cond_net_.AddExternalOutput(false_blob)
        self.idle_net_ = core.Net('idle-net')
        self.idle_net_.ConstantFill([], shape=[], value=0, dtype=core.DataType.INT64)

    def CheckNetOutput(self, nets_and_expects):
        if False:
            i = 10
            return i + 15
        '\n        Check the net output is expected\n        nets_and_expects is a list of tuples (net, expect)\n        '
        for (net, expect) in nets_and_expects:
            output = workspace.FetchBlob(net.Proto().external_output[-1])
            self.assertEqual(output, expect)

    def CheckNetAllOutput(self, net, expects):
        if False:
            i = 10
            return i + 15
        '\n        Check the net output is expected\n        expects is a list of bools.\n        '
        self.assertEqual(len(net.Proto().external_output), len(expects))
        for i in range(len(expects)):
            output = workspace.FetchBlob(net.Proto().external_output[i])
            self.assertEqual(output, expects[i])

    def BuildAndRunPlan(self, step):
        if False:
            while True:
                i = 10
        plan = core.Plan('test')
        plan.AddStep(control.Do('init', self.init_net_))
        plan.AddStep(step)
        self.assertEqual(workspace.RunPlan(plan), True)

    def ForLoopTest(self, nets_or_steps):
        if False:
            i = 10
            return i + 15
        step = control.For('myFor', nets_or_steps, self.N_)
        self.BuildAndRunPlan(step)
        self.CheckNetOutput([(self.cnt_net_, self.N_)])

    def testForLoopWithNets(self):
        if False:
            i = 10
            return i + 15
        self.ForLoopTest(self.cnt_net_)
        self.ForLoopTest([self.cnt_net_, self.idle_net_])

    def testForLoopWithStep(self):
        if False:
            return 10
        step = control.Do('count', self.cnt_net_)
        self.ForLoopTest(step)
        self.ForLoopTest([step, self.idle_net_])

    def WhileLoopTest(self, nets_or_steps):
        if False:
            i = 10
            return i + 15
        step = control.While('myWhile', self.cond_net_, nets_or_steps)
        self.BuildAndRunPlan(step)
        self.CheckNetOutput([(self.cnt_net_, self.N_)])

    def testWhileLoopWithNet(self):
        if False:
            i = 10
            return i + 15
        self.WhileLoopTest(self.cnt_net_)
        self.WhileLoopTest([self.cnt_net_, self.idle_net_])

    def testWhileLoopWithStep(self):
        if False:
            for i in range(10):
                print('nop')
        step = control.Do('count', self.cnt_net_)
        self.WhileLoopTest(step)
        self.WhileLoopTest([step, self.idle_net_])

    def UntilLoopTest(self, nets_or_steps):
        if False:
            while True:
                i = 10
        step = control.Until('myUntil', self.not_cond_net_, nets_or_steps)
        self.BuildAndRunPlan(step)
        self.CheckNetOutput([(self.cnt_net_, self.N_)])

    def testUntilLoopWithNet(self):
        if False:
            print('Hello World!')
        self.UntilLoopTest(self.cnt_net_)
        self.UntilLoopTest([self.cnt_net_, self.idle_net_])

    def testUntilLoopWithStep(self):
        if False:
            for i in range(10):
                print('nop')
        step = control.Do('count', self.cnt_net_)
        self.UntilLoopTest(step)
        self.UntilLoopTest([step, self.idle_net_])

    def DoWhileLoopTest(self, nets_or_steps):
        if False:
            i = 10
            return i + 15
        step = control.DoWhile('myDoWhile', self.cond_net_, nets_or_steps)
        self.BuildAndRunPlan(step)
        self.CheckNetOutput([(self.cnt_net_, self.N_)])

    def testDoWhileLoopWithNet(self):
        if False:
            return 10
        self.DoWhileLoopTest(self.cnt_net_)
        self.DoWhileLoopTest([self.idle_net_, self.cnt_net_])

    def testDoWhileLoopWithStep(self):
        if False:
            while True:
                i = 10
        step = control.Do('count', self.cnt_net_)
        self.DoWhileLoopTest(step)
        self.DoWhileLoopTest([self.idle_net_, step])

    def DoUntilLoopTest(self, nets_or_steps):
        if False:
            return 10
        step = control.DoUntil('myDoUntil', self.not_cond_net_, nets_or_steps)
        self.BuildAndRunPlan(step)
        self.CheckNetOutput([(self.cnt_net_, self.N_)])

    def testDoUntilLoopWithNet(self):
        if False:
            print('Hello World!')
        self.DoUntilLoopTest(self.cnt_net_)
        self.DoUntilLoopTest([self.cnt_net_, self.idle_net_])

    def testDoUntilLoopWithStep(self):
        if False:
            print('Hello World!')
        step = control.Do('count', self.cnt_net_)
        self.DoUntilLoopTest(step)
        self.DoUntilLoopTest([self.idle_net_, step])

    def IfCondTest(self, cond_net, expect, cond_on_blob):
        if False:
            for i in range(10):
                print('nop')
        if cond_on_blob:
            step = control.Do('if-all', control.Do('count', cond_net), control.If('myIf', cond_net.Proto().external_output[-1], self.cnt_net_))
        else:
            step = control.If('myIf', cond_net, self.cnt_net_)
        self.BuildAndRunPlan(step)
        self.CheckNetOutput([(self.cnt_net_, expect)])

    def testIfCondTrueOnNet(self):
        if False:
            while True:
                i = 10
        self.IfCondTest(self.true_cond_net_, 1, False)

    def testIfCondTrueOnBlob(self):
        if False:
            i = 10
            return i + 15
        self.IfCondTest(self.true_cond_net_, 1, True)

    def testIfCondFalseOnNet(self):
        if False:
            while True:
                i = 10
        self.IfCondTest(self.false_cond_net_, 0, False)

    def testIfCondFalseOnBlob(self):
        if False:
            print('Hello World!')
        self.IfCondTest(self.false_cond_net_, 0, True)

    def IfElseCondTest(self, cond_net, cond_value, expect, cond_on_blob):
        if False:
            return 10
        if cond_value:
            run_net = self.cnt_net_
        else:
            run_net = self.cnt_2_net_
        if cond_on_blob:
            step = control.Do('if-else-all', control.Do('count', cond_net), control.If('myIfElse', cond_net.Proto().external_output[-1], self.cnt_net_, self.cnt_2_net_))
        else:
            step = control.If('myIfElse', cond_net, self.cnt_net_, self.cnt_2_net_)
        self.BuildAndRunPlan(step)
        self.CheckNetOutput([(run_net, expect)])

    def testIfElseCondTrueOnNet(self):
        if False:
            i = 10
            return i + 15
        self.IfElseCondTest(self.true_cond_net_, True, 1, False)

    def testIfElseCondTrueOnBlob(self):
        if False:
            print('Hello World!')
        self.IfElseCondTest(self.true_cond_net_, True, 1, True)

    def testIfElseCondFalseOnNet(self):
        if False:
            while True:
                i = 10
        self.IfElseCondTest(self.false_cond_net_, False, 2, False)

    def testIfElseCondFalseOnBlob(self):
        if False:
            for i in range(10):
                print('nop')
        self.IfElseCondTest(self.false_cond_net_, False, 2, True)

    def IfNotCondTest(self, cond_net, expect, cond_on_blob):
        if False:
            i = 10
            return i + 15
        if cond_on_blob:
            step = control.Do('if-not', control.Do('count', cond_net), control.IfNot('myIfNot', cond_net.Proto().external_output[-1], self.cnt_net_))
        else:
            step = control.IfNot('myIfNot', cond_net, self.cnt_net_)
        self.BuildAndRunPlan(step)
        self.CheckNetOutput([(self.cnt_net_, expect)])

    def testIfNotCondTrueOnNet(self):
        if False:
            print('Hello World!')
        self.IfNotCondTest(self.true_cond_net_, 0, False)

    def testIfNotCondTrueOnBlob(self):
        if False:
            for i in range(10):
                print('nop')
        self.IfNotCondTest(self.true_cond_net_, 0, True)

    def testIfNotCondFalseOnNet(self):
        if False:
            for i in range(10):
                print('nop')
        self.IfNotCondTest(self.false_cond_net_, 1, False)

    def testIfNotCondFalseOnBlob(self):
        if False:
            print('Hello World!')
        self.IfNotCondTest(self.false_cond_net_, 1, True)

    def IfNotElseCondTest(self, cond_net, cond_value, expect, cond_on_blob):
        if False:
            print('Hello World!')
        if cond_value:
            run_net = self.cnt_2_net_
        else:
            run_net = self.cnt_net_
        if cond_on_blob:
            step = control.Do('if-not-else', control.Do('count', cond_net), control.IfNot('myIfNotElse', cond_net.Proto().external_output[-1], self.cnt_net_, self.cnt_2_net_))
        else:
            step = control.IfNot('myIfNotElse', cond_net, self.cnt_net_, self.cnt_2_net_)
        self.BuildAndRunPlan(step)
        self.CheckNetOutput([(run_net, expect)])

    def testIfNotElseCondTrueOnNet(self):
        if False:
            for i in range(10):
                print('nop')
        self.IfNotElseCondTest(self.true_cond_net_, True, 2, False)

    def testIfNotElseCondTrueOnBlob(self):
        if False:
            return 10
        self.IfNotElseCondTest(self.true_cond_net_, True, 2, True)

    def testIfNotElseCondFalseOnNet(self):
        if False:
            print('Hello World!')
        self.IfNotElseCondTest(self.false_cond_net_, False, 1, False)

    def testIfNotElseCondFalseOnBlob(self):
        if False:
            return 10
        self.IfNotElseCondTest(self.false_cond_net_, False, 1, True)

    def testSwitch(self):
        if False:
            i = 10
            return i + 15
        step = control.Switch('mySwitch', (self.false_cond_net_, self.cnt_net_), (self.true_cond_net_, self.cnt_2_net_))
        self.BuildAndRunPlan(step)
        self.CheckNetOutput([(self.cnt_net_, 0), (self.cnt_2_net_, 2)])

    def testSwitchNot(self):
        if False:
            while True:
                i = 10
        step = control.SwitchNot('mySwitchNot', (self.false_cond_net_, self.cnt_net_), (self.true_cond_net_, self.cnt_2_net_))
        self.BuildAndRunPlan(step)
        self.CheckNetOutput([(self.cnt_net_, 1), (self.cnt_2_net_, 0)])

    def testBoolNet(self):
        if False:
            print('Hello World!')
        bool_net = control.BoolNet(('a', True))
        step = control.Do('bool', bool_net)
        self.BuildAndRunPlan(step)
        self.CheckNetAllOutput(bool_net, [True])
        bool_net = control.BoolNet(('a', True), ('b', False))
        step = control.Do('bool', bool_net)
        self.BuildAndRunPlan(step)
        self.CheckNetAllOutput(bool_net, [True, False])
        bool_net = control.BoolNet([('a', True), ('b', False)])
        step = control.Do('bool', bool_net)
        self.BuildAndRunPlan(step)
        self.CheckNetAllOutput(bool_net, [True, False])

    def testCombineConditions(self):
        if False:
            print('Hello World!')
        combine_net = control.CombineConditions('test', [self.true_cond_net_, self.false_cond_net_], 'Or')
        step = control.Do('combine', self.true_cond_net_, self.false_cond_net_, combine_net)
        self.BuildAndRunPlan(step)
        self.CheckNetOutput([(combine_net, True)])
        combine_net = control.CombineConditions('test', [self.true_cond_net_, self.false_cond_net_], 'And')
        step = control.Do('combine', self.true_cond_net_, self.false_cond_net_, combine_net)
        self.BuildAndRunPlan(step)
        self.CheckNetOutput([(combine_net, False)])

    def testMergeConditionNets(self):
        if False:
            while True:
                i = 10
        merge_net = control.MergeConditionNets('test', [self.true_cond_net_, self.false_cond_net_], 'Or')
        step = control.Do('merge', merge_net)
        self.BuildAndRunPlan(step)
        self.CheckNetOutput([(merge_net, True)])
        merge_net = control.MergeConditionNets('test', [self.true_cond_net_, self.false_cond_net_], 'And')
        step = control.Do('merge', merge_net)
        self.BuildAndRunPlan(step)
        self.CheckNetOutput([(merge_net, False)])