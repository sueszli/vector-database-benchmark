import paddle
paddle.enable_static()
import unittest
from paddle import base
from paddle.distributed import CountFilterEntry, ProbabilityEntry, ShowClickEntry

class EntryAttrChecks(unittest.TestCase):

    def base(self):
        if False:
            return 10
        with self.assertRaises(NotImplementedError):
            from paddle.distributed.entry_attr import EntryAttr
            base = EntryAttr()
            base._to_attr()

    def probability_entry(self):
        if False:
            while True:
                i = 10
        prob = ProbabilityEntry(0.5)
        ss = prob._to_attr()
        self.assertEqual('probability_entry:0.5', ss)
        with self.assertRaises(ValueError):
            prob1 = ProbabilityEntry('none')
        with self.assertRaises(ValueError):
            prob2 = ProbabilityEntry(-1)

    def countfilter_entry(self):
        if False:
            print('Hello World!')
        counter = CountFilterEntry(20)
        ss = counter._to_attr()
        self.assertEqual('count_filter_entry:20', ss)
        with self.assertRaises(ValueError):
            counter1 = CountFilterEntry('none')
        with self.assertRaises(ValueError):
            counter2 = CountFilterEntry(-1)

    def showclick_entry(self):
        if False:
            for i in range(10):
                print('nop')
        showclick = ShowClickEntry('show', 'click')
        ss = showclick._to_attr()
        self.assertEqual('show_click_entry:show:click', ss)

    def spaese_layer(self):
        if False:
            for i in range(10):
                print('nop')
        prog = base.Program()
        scope = base.core.Scope()
        with base.scope_guard(scope):
            with base.program_guard(prog):
                input = paddle.static.data(name='dnn_data', shape=[-1, 1], dtype='int64', lod_level=1)
                prob = ProbabilityEntry(0.5)
                emb = paddle.static.nn.sparse_embedding(input=input, size=[100, 10], is_test=False, entry=prob, param_attr=base.ParamAttr(name='deep_embedding'))
                pool = paddle.static.nn.sequence_lod.sequence_pool(input=emb, pool_type='sum')
                predict = paddle.static.nn.fc(x=pool, size=2, activation='softmax')
        block = prog.global_block()
        for op in block.ops:
            if op.type == 'lookup_table':
                entry = op.attr('entry')
                is_test = op.attr('is_test')
                is_sparse = op.attr('is_sparse')
                is_distributed = op.attr('is_distributed')
                self.assertEqual(entry, 'probability_entry:0.5')
                self.assertTrue(is_distributed)
                self.assertTrue(is_sparse)
                self.assertFalse(is_test)

class TestEntryAttrs(EntryAttrChecks):

    def test_base(self):
        if False:
            print('Hello World!')
        self.base()

    def test_prob(self):
        if False:
            print('Hello World!')
        self.probability_entry()

    def test_counter(self):
        if False:
            return 10
        self.countfilter_entry()

    def test_showclick(self):
        if False:
            for i in range(10):
                print('nop')
        self.showclick_entry()

    def test_spaese_embedding_layer(self):
        if False:
            i = 10
            return i + 15
        self.spaese_layer()
if __name__ == '__main__':
    unittest.main()