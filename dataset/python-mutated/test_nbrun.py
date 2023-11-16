import os
from rest_framework.authtoken.models import Token
from rest_framework.test import APITestCase
from execnb.nbio import dict2nb
from apps.nb.nbrun import NbRun
from apps.nb.utils import one_cell_notebook, test_notebook

class NbRunTestCase(APITestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        pass

    def test_iframe_in_output(self):
        if False:
            for i in range(10):
                print('nop')
        fname = 'test-iframe-output.html'
        with open(fname, 'w') as fout:
            fout.write('test')
        nbrun = NbRun()
        code = f'from IPython.display import IFrame\nIFrame(src="{fname}", width=100, height=100)'
        nb = one_cell_notebook(code)
        nb = dict2nb(nb)
        nbrun.run_cell(nb.cells[0])
        self.assertTrue(fname not in nbrun.export_html(nb, full_header=False))
        os.remove(fname)

    def test_stop_on_error(self):
        if False:
            while True:
                i = 10
        nb = test_notebook(code=['2+2', 'print(a)', 'print(1)'])
        nb = dict2nb(nb)
        nbrun = NbRun(stop_on_error=False)
        nbrun.run_notebook(nb)
        self.assertTrue(len(nb.cells[2].outputs) > 0)
        nbrun = NbRun(stop_on_error=True)
        nbrun.run_notebook(nb)
        self.assertTrue(len(nb.cells[2].outputs) == 0)