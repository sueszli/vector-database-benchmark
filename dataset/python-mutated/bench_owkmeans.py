import numpy as np
from Orange.data import Domain, Table, ContinuousVariable
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.unsupervised.owkmeans import OWKMeans
from .base import benchmark

def table(rows, cols):
    if False:
        return 10
    return Table.from_numpy(Domain([ContinuousVariable(str(i)) for i in range(cols)]), np.random.RandomState(0).rand(rows, cols))

class BenchOWKmeans(WidgetTest):

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        super().setUpClass()
        cls.d_100_100 = table(100, 100)
        cls.d_sampled_silhouette = table(10000, 1)
        cls.d_10_500 = table(10, 500)

    def setUp(self):
        if False:
            while True:
                i = 10
        self.widget = None

    def widget_from_to(self):
        if False:
            print('Hello World!')
        self.widget = self.create_widget(OWKMeans, stored_settings={'auto_commit': False})
        self.widget.controls.k_from.setValue(2)
        self.widget.controls.k_to.setValue(6)

    @benchmark(number=3, warmup=1, repeat=3)
    def bench_from_to_100_100(self):
        if False:
            for i in range(10):
                print('nop')
        self.widget_from_to()
        self.send_signal(self.widget.Inputs.data, self.d_100_100)
        self.commit_and_wait(wait=100 * 1000)

    @benchmark(number=3, warmup=1, repeat=3)
    def bench_from_to_100_100_no_normalize(self):
        if False:
            i = 10
            return i + 15
        self.widget_from_to()
        self.widget.normalize = False
        self.send_signal(self.widget.Inputs.data, self.d_100_100)
        self.commit_and_wait(wait=100 * 1000)

    @benchmark(number=3, warmup=1, repeat=3)
    def bench_from_to_sampled_silhouette(self):
        if False:
            return 10
        self.widget_from_to()
        self.send_signal(self.widget.Inputs.data, self.d_sampled_silhouette)
        self.commit_and_wait(wait=100 * 1000)

    @benchmark(number=3, warmup=1, repeat=3)
    def bench_wide(self):
        if False:
            i = 10
            return i + 15
        self.widget = self.create_widget(OWKMeans, stored_settings={'auto_commit': False})
        self.send_signal(self.widget.Inputs.data, self.d_10_500)
        self.commit_and_wait(wait=100 * 1000)