import inspect
import os
import numpy as np
from gnuradio import gr, gr_unittest, filter
from file_taps_loader import file_taps_loader

class qa_file_taps_loader(gr_unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            print('Hello World!')
        self.tb = None

    def test_float_taps(self):
        if False:
            print('Hello World!')
        '\n        Load some float taps.\n        '
        path = 'test_taps_float'
        c_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + '/' + path
        test_file_content = 'restype,fir\nfs,320000.\npbend,50000.0\ngain,2.0\nsbstart,100000.0\nfilttype,lpf\natten,40.0\nwintype,0\nntaps,11\ntaps,-0.010157553479075432,-0.01920645497739315,0.01674678549170494,0.20396660268306732,0.49091556668281555,0.6354700922966003,0.49091556668281555,0.20396660268306732,0.01674678549170494,-0.01920645497739315,-0.010157553479075432\n'
        test_file = open(c_path, 'w')
        test_file.write(test_file_content)
        test_file.close()
        verbose = False
        ftl = filter.file_taps_loader(c_path, verbose)
        os.remove(c_path)
        expected_taps = tuple(np.array((-0.01015755, -0.01920645, 0.01674679, 0.2039666, 0.49091557, 0.63547009, 0.49091557, 0.2039666, 0.01674679, -0.01920645, -0.01015755), dtype=float))
        self.assertEqual(type(ftl.get_taps()), type(expected_taps))
        self.assertEqual(type(ftl.get_taps()[0]), type(expected_taps[0]))
        self.assertFloatTuplesAlmostEqual(expected_taps, ftl.get_taps(), 6)
        filter.fft_filter_ccc(1, ftl.get_taps())

    def test_complex_taps(self):
        if False:
            print('Hello World!')
        '\n        Load some complex taps.\n        '
        path = 'test_taps_complex'
        c_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + '/' + path
        test_file_content = 'restype,fir\nfs,320000.0\npbend,100000.0\ngain,2.0\nfilttype,cbpf\natten,10.0\nntaps,15\ntb,10000.0\nwintype,0\npbstart,50000.0\ntaps,(0.00151832250413-0.0018500800943j),(-0.00244879024103-0.00163623178378j),(0.0109249493107-0.0204391144216j),(0.0673716515303+0.0279062893242j),(-0.0446246191859+0.147107481956j),(-0.243210762739-0.0483776889741j),(0.031814865768-0.323020994663j),(0.354127079248+8.44304750558e-08j),(0.0318147130311+0.323020994663j),(-0.243210792542+0.0483775734901j),(-0.0446245521307-0.147107496858j),(0.0673716664314-0.0279062557966j),(0.0109249399975+0.0204391200095j),(-0.00244879163802+0.00163622945547j),(0.0015183207579+0.0018500816077j)\n'
        test_file = open(c_path, 'w')
        test_file.write(test_file_content)
        test_file.close()
        verbose = False
        ftl = filter.file_taps_loader(c_path, verbose)
        os.remove(c_path)
        expected_taps = tuple(np.array((0.0015183225041255355 - 0.0018500800943002105j, -0.002448790241032839 - 0.0016362317837774754j, 0.010924949310719967 - 0.020439114421606064j, 0.06737165153026581 + 0.02790628932416439j, -0.04462461918592453 + 0.14710748195648193j, -0.24321076273918152 - 0.048377688974142075j, 0.03181486576795578 - 0.3230209946632385j, 0.35412707924842834 + 8.44304750557967e-08j, 0.03181471303105354 + 0.3230209946632385j, -0.2432107925415039 + 0.04837757349014282j, -0.04462455213069916 - 0.14710749685764313j, 0.067371666431427 - 0.027906255796551704j, 0.01092493999749422 + 0.02043912000954151j, -0.0024487916380167007 + 0.0016362294554710388j, 0.001518320757895708 + 0.0018500816076993942j), dtype=complex))
        self.assertEqual(type(ftl.get_taps()), type(expected_taps))
        self.assertEqual(type(ftl.get_taps()[0]), type(expected_taps[0]))
        self.assertComplexTuplesAlmostEqual(expected_taps, ftl.get_taps(), 6)
        filter.fft_filter_ccc(1, ftl.get_taps())

    def test_raises_file_missing(self):
        if False:
            while True:
                i = 10
        '\n        Load some file, that does not exist.\n        '
        path = 'blablabla__wrong'
        verbose = False
        with self.assertRaises(RuntimeError) as re:
            _ = filter.file_taps_loader(path, verbose)
        self.assertEqual(str(re.exception), 'file_taps_loader: Can not open ' + '"' + path + '"' + '.')
if __name__ == '__main__':
    gr_unittest.run(qa_file_taps_loader)