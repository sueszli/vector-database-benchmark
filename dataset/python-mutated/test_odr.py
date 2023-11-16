import tempfile
import shutil
import os
import numpy as np
from numpy import pi
from numpy.testing import assert_array_almost_equal, assert_equal, assert_warns, assert_allclose
import pytest
from pytest import raises as assert_raises
from scipy.odr import Data, Model, ODR, RealData, OdrStop, OdrWarning, multilinear, exponential, unilinear, quadratic, polynomial

class TestODR:

    def test_bad_data(self):
        if False:
            i = 10
            return i + 15
        assert_raises(ValueError, Data, 2, 1)
        assert_raises(ValueError, RealData, 2, 1)

    def empty_data_func(self, B, x):
        if False:
            while True:
                i = 10
        return B[0] * x + B[1]

    def test_empty_data(self):
        if False:
            i = 10
            return i + 15
        beta0 = [0.02, 0.0]
        linear = Model(self.empty_data_func)
        empty_dat = Data([], [])
        assert_warns(OdrWarning, ODR, empty_dat, linear, beta0=beta0)
        empty_dat = RealData([], [])
        assert_warns(OdrWarning, ODR, empty_dat, linear, beta0=beta0)

    def explicit_fcn(self, B, x):
        if False:
            return 10
        ret = B[0] + B[1] * np.power(np.exp(B[2] * x) - 1.0, 2)
        return ret

    def explicit_fjd(self, B, x):
        if False:
            for i in range(10):
                print('nop')
        eBx = np.exp(B[2] * x)
        ret = B[1] * 2.0 * (eBx - 1.0) * B[2] * eBx
        return ret

    def explicit_fjb(self, B, x):
        if False:
            print('Hello World!')
        eBx = np.exp(B[2] * x)
        res = np.vstack([np.ones(x.shape[-1]), np.power(eBx - 1.0, 2), B[1] * 2.0 * (eBx - 1.0) * eBx * x])
        return res

    def test_explicit(self):
        if False:
            return 10
        explicit_mod = Model(self.explicit_fcn, fjacb=self.explicit_fjb, fjacd=self.explicit_fjd, meta=dict(name='Sample Explicit Model', ref='ODRPACK UG, pg. 39'))
        explicit_dat = Data([0.0, 0.0, 5.0, 7.0, 7.5, 10.0, 16.0, 26.0, 30.0, 34.0, 34.5, 100.0], [1265.0, 1263.6, 1258.0, 1254.0, 1253.0, 1249.8, 1237.0, 1218.0, 1220.6, 1213.8, 1215.5, 1212.0])
        explicit_odr = ODR(explicit_dat, explicit_mod, beta0=[1500.0, -50.0, -0.1], ifixx=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0])
        explicit_odr.set_job(deriv=2)
        explicit_odr.set_iprint(init=0, iter=0, final=0)
        out = explicit_odr.run()
        assert_array_almost_equal(out.beta, np.array([1264.6548050648876, -54.018409956678255, -0.08784971216525372]))
        assert_array_almost_equal(out.sd_beta, np.array([1.0349270280543437, 1.583997785262061, 0.0063321988657267]))
        assert_array_almost_equal(out.cov_beta, np.array([[0.4494959237900304, -0.3742197689036474, -0.0008097821746846891], [-0.3742197689036474, 1.0529686462751804, -0.0019453521827942002], [-0.0008097821746846891, -0.0019453521827942002, 1.6827336938454476e-05]]))

    def implicit_fcn(self, B, x):
        if False:
            return 10
        return B[2] * np.power(x[0] - B[0], 2) + 2.0 * B[3] * (x[0] - B[0]) * (x[1] - B[1]) + B[4] * np.power(x[1] - B[1], 2) - 1.0

    def test_implicit(self):
        if False:
            for i in range(10):
                print('nop')
        implicit_mod = Model(self.implicit_fcn, implicit=1, meta=dict(name='Sample Implicit Model', ref='ODRPACK UG, pg. 49'))
        implicit_dat = Data([[0.5, 1.2, 1.6, 1.86, 2.12, 2.36, 2.44, 2.36, 2.06, 1.74, 1.34, 0.9, -0.28, -0.78, -1.36, -1.9, -2.5, -2.88, -3.18, -3.44], [-0.12, -0.6, -1.0, -1.4, -2.54, -3.36, -4.0, -4.75, -5.25, -5.64, -5.97, -6.32, -6.44, -6.44, -6.41, -6.25, -5.88, -5.5, -5.24, -4.86]], 1)
        implicit_odr = ODR(implicit_dat, implicit_mod, beta0=[-1.0, -3.0, 0.09, 0.02, 0.08])
        out = implicit_odr.run()
        assert_array_almost_equal(out.beta, np.array([-0.9993809167281279, -2.9310484652026476, 0.0875730502693354, 0.0162299708984738, 0.0797537982976416]))
        assert_array_almost_equal(out.sd_beta, np.array([0.1113840353364371, 0.1097673310686467, 0.0041060738314314, 0.0027500347539902, 0.0034962501532468]))
        assert_allclose(out.cov_beta, np.array([[2.1089274602333052, -1.943768641197904, 0.07026355086834445, -0.04717526737347486, 0.052515575927380355], [-1.943768641197904, 2.0481509222414456, -0.06160051585305731, 0.04626882780623293, -0.05882230750139147], [0.07026355086834445, -0.06160051585305731, 0.002865954256157931, -0.001462866226001449, 0.0014528860663055824], [-0.04717526737347486, 0.04626882780623293, -0.001462866226001449, 0.0012855592885514335, -0.0012692942951415293], [0.052515575927380355, -0.05882230750139147, 0.0014528860663055824, -0.0012692942951415293, 0.0020778813389755596]]), rtol=1e-06, atol=2e-06)

    def multi_fcn(self, B, x):
        if False:
            print('Hello World!')
        if (x < 0.0).any():
            raise OdrStop
        theta = pi * B[3] / 2.0
        ctheta = np.cos(theta)
        stheta = np.sin(theta)
        omega = np.power(2.0 * pi * x * np.exp(-B[2]), B[3])
        phi = np.arctan2(omega * stheta, 1.0 + omega * ctheta)
        r = (B[0] - B[1]) * np.power(np.sqrt(np.power(1.0 + omega * ctheta, 2) + np.power(omega * stheta, 2)), -B[4])
        ret = np.vstack([B[1] + r * np.cos(B[4] * phi), r * np.sin(B[4] * phi)])
        return ret

    def test_multi(self):
        if False:
            i = 10
            return i + 15
        multi_mod = Model(self.multi_fcn, meta=dict(name='Sample Multi-Response Model', ref='ODRPACK UG, pg. 56'))
        multi_x = np.array([30.0, 50.0, 70.0, 100.0, 150.0, 200.0, 300.0, 500.0, 700.0, 1000.0, 1500.0, 2000.0, 3000.0, 5000.0, 7000.0, 10000.0, 15000.0, 20000.0, 30000.0, 50000.0, 70000.0, 100000.0, 150000.0])
        multi_y = np.array([[4.22, 4.167, 4.132, 4.038, 4.019, 3.956, 3.884, 3.784, 3.713, 3.633, 3.54, 3.433, 3.358, 3.258, 3.193, 3.128, 3.059, 2.984, 2.934, 2.876, 2.838, 2.798, 2.759], [0.136, 0.167, 0.188, 0.212, 0.236, 0.257, 0.276, 0.297, 0.309, 0.311, 0.314, 0.311, 0.305, 0.289, 0.277, 0.255, 0.24, 0.218, 0.202, 0.182, 0.168, 0.153, 0.139]])
        n = len(multi_x)
        multi_we = np.zeros((2, 2, n), dtype=float)
        multi_ifixx = np.ones(n, dtype=int)
        multi_delta = np.zeros(n, dtype=float)
        multi_we[0, 0, :] = 559.6
        multi_we[1, 0, :] = multi_we[0, 1, :] = -1634.0
        multi_we[1, 1, :] = 8397.0
        for i in range(n):
            if multi_x[i] < 100.0:
                multi_ifixx[i] = 0
            elif multi_x[i] <= 150.0:
                pass
            elif multi_x[i] <= 1000.0:
                multi_delta[i] = 25.0
            elif multi_x[i] <= 10000.0:
                multi_delta[i] = 560.0
            elif multi_x[i] <= 100000.0:
                multi_delta[i] = 9500.0
            else:
                multi_delta[i] = 144000.0
            if multi_x[i] == 100.0 or multi_x[i] == 150.0:
                multi_we[:, :, i] = 0.0
        multi_dat = Data(multi_x, multi_y, wd=0.0001 / np.power(multi_x, 2), we=multi_we)
        multi_odr = ODR(multi_dat, multi_mod, beta0=[4.0, 2.0, 7.0, 0.4, 0.5], delta0=multi_delta, ifixx=multi_ifixx)
        multi_odr.set_job(deriv=1, del_init=1)
        out = multi_odr.run()
        assert_array_almost_equal(out.beta, np.array([4.379988030593896, 2.4333057577497703, 8.002884589950398, 0.5101147161764654, 0.5173902330489161]))
        assert_array_almost_equal(out.sd_beta, np.array([0.0130625231081944, 0.0130499785273277, 0.1167085962217757, 0.0132642749596149, 0.0288529201353984]))
        assert_array_almost_equal(out.cov_beta, np.array([[0.0064918418231375, 0.0036159705923791, 0.0438637051470406, -0.0058700836512467, 0.011281212888768], [0.0036159705923791, 0.0064793789429006, 0.0517610978353126, -0.0051181304940204, 0.0130726943624117], [0.0438637051470406, 0.0517610978353126, 0.5182263323095322, -0.0563083340093696, 0.1269490939468611], [-0.0058700836512467, -0.0051181304940204, -0.0563083340093696, 0.0066939246261263, -0.0140184391377962], [0.011281212888768, 0.0130726943624117, 0.1269490939468611, -0.0140184391377962, 0.0316733013820852]]))

    def pearson_fcn(self, B, x):
        if False:
            return 10
        return B[0] + B[1] * x

    def test_pearson(self):
        if False:
            print('Hello World!')
        p_x = np.array([0.0, 0.9, 1.8, 2.6, 3.3, 4.4, 5.2, 6.1, 6.5, 7.4])
        p_y = np.array([5.9, 5.4, 4.4, 4.6, 3.5, 3.7, 2.8, 2.8, 2.4, 1.5])
        p_sx = np.array([0.03, 0.03, 0.04, 0.035, 0.07, 0.11, 0.13, 0.22, 0.74, 1.0])
        p_sy = np.array([1.0, 0.74, 0.5, 0.35, 0.22, 0.22, 0.12, 0.12, 0.1, 0.04])
        p_dat = RealData(p_x, p_y, sx=p_sx, sy=p_sy)
        pr_dat = RealData(p_y, p_x, sx=p_sy, sy=p_sx)
        p_mod = Model(self.pearson_fcn, meta=dict(name='Uni-linear Fit'))
        p_odr = ODR(p_dat, p_mod, beta0=[1.0, 1.0])
        pr_odr = ODR(pr_dat, p_mod, beta0=[1.0, 1.0])
        out = p_odr.run()
        assert_array_almost_equal(out.beta, np.array([5.476740029923167, -0.4796082367610305]))
        assert_array_almost_equal(out.sd_beta, np.array([0.3590121690702467, 0.0706291186037444]))
        assert_array_almost_equal(out.cov_beta, np.array([[0.0854275622946333, -0.0161807025443155], [-0.0161807025443155, 0.003306337993922]]))
        rout = pr_odr.run()
        assert_array_almost_equal(rout.beta, np.array([11.419202241078123, -2.0850374506165474]))
        assert_array_almost_equal(rout.sd_beta, np.array([0.9820231665657161, 0.3070515616198911]))
        assert_array_almost_equal(rout.cov_beta, np.array([[0.6391799462548782, -0.1955657291119177], [-0.1955657291119177, 0.0624888159223392]]))

    def lorentz(self, beta, x):
        if False:
            i = 10
            return i + 15
        return beta[0] * beta[1] * beta[2] / np.sqrt(np.power(x * x - beta[2] * beta[2], 2.0) + np.power(beta[1] * x, 2.0))

    def test_lorentz(self):
        if False:
            for i in range(10):
                print('nop')
        l_sy = np.array([0.29] * 18)
        l_sx = np.array([0.000972971, 0.000948268, 0.000707632, 0.000706679, 0.000706074, 0.000703918, 0.000698955, 0.000456856, 0.000455207, 0.000662717, 0.000654619, 0.000652694, 8.59202e-07, 0.00106589, 0.00106378, 0.00125483, 0.00140818, 0.00241839])
        l_dat = RealData([3.9094, 3.85945, 3.84976, 3.84716, 3.84551, 3.83964, 3.82608, 3.78847, 3.78163, 3.72558, 3.70274, 3.6973, 3.67373, 3.65982, 3.6562, 3.62498, 3.55525, 3.41886], [652, 910.5, 984, 1000, 1007.5, 1053, 1160.5, 1409.5, 1430, 1122, 957.5, 920, 777.5, 709.5, 698, 578.5, 418.5, 275.5], sx=l_sx, sy=l_sy)
        l_mod = Model(self.lorentz, meta=dict(name='Lorentz Peak'))
        l_odr = ODR(l_dat, l_mod, beta0=(1000.0, 0.1, 3.8))
        out = l_odr.run()
        assert_array_almost_equal(out.beta, np.array([1430.6780846149925, 0.1339050903453831, 3.779819360010901]))
        assert_array_almost_equal(out.sd_beta, np.array([0.7362118681133096, 0.0003506889994147165, 0.0002445120928140899]))
        assert_array_almost_equal(out.cov_beta, np.array([[0.24714409064597873, -6.906726191111084e-05, -3.123695327042499e-05], [-6.906726191111084e-05, 5.607753151733301e-08, 3.61332618327226e-08], [-3.123695327042499e-05, 3.61332618327226e-08, 2.726122002517173e-08]]))

    def test_ticket_1253(self):
        if False:
            while True:
                i = 10

        def linear(c, x):
            if False:
                i = 10
                return i + 15
            return c[0] * x + c[1]
        c = [2.0, 3.0]
        x = np.linspace(0, 10)
        y = linear(c, x)
        model = Model(linear)
        data = Data(x, y, wd=1.0, we=1.0)
        job = ODR(data, model, beta0=[1.0, 1.0])
        result = job.run()
        assert_equal(result.info, 2)

    def test_ifixx(self):
        if False:
            i = 10
            return i + 15
        x1 = [-2.01, -0.99, -0.001, 1.02, 1.98]
        x2 = [3.98, 1.01, 0.001, 0.998, 4.01]
        fix = np.vstack((np.zeros_like(x1, dtype=int), np.ones_like(x2, dtype=int)))
        data = Data(np.vstack((x1, x2)), y=1, fix=fix)
        model = Model(lambda beta, x: x[1, :] - beta[0] * x[0, :] ** 2.0, implicit=True)
        odr1 = ODR(data, model, beta0=np.array([1.0]))
        sol1 = odr1.run()
        odr2 = ODR(data, model, beta0=np.array([1.0]), ifixx=fix)
        sol2 = odr2.run()
        assert_equal(sol1.beta, sol2.beta)

    def test_ticket_11800(self):
        if False:
            i = 10
            return i + 15
        beta_true = np.array([1.0, 2.3, 1.1, -1.0, 1.3, 0.5])
        nr_measurements = 10
        std_dev_x = 0.01
        x_error = np.array([[0.00063445, 0.00515731, 0.00162719, 0.01022866, -0.01624845, 0.00482652, 0.00275988, -0.00714734, -0.00929201, -0.00687301], [-0.00831623, -0.00821211, -0.00203459, 0.00938266, -0.00701829, 0.0032169, 0.00259194, -0.00581017, -0.0030283, 0.01014164]])
        std_dev_y = 0.05
        y_error = np.array([[0.05275304, 0.04519563, -0.07524086, 0.03575642, 0.04745194, 0.03806645, 0.07061601, -0.00753604, -0.02592543, -0.02394929], [0.03632366, 0.06642266, 0.08373122, 0.03988822, -0.0092536, -0.03750469, -0.03198903, 0.01642066, 0.01293648, -0.05627085]])
        beta_solution = np.array([2.6292023575666588, -126.60848499629961, 129.70357277540307, -1.8856098540118547, 78.38341607712749, -76.41240768380871])

        def func(beta, x):
            if False:
                i = 10
                return i + 15
            y0 = beta[0] + beta[1] * x[0, :] + beta[2] * x[1, :]
            y1 = beta[3] + beta[4] * x[0, :] + beta[5] * x[1, :]
            return np.vstack((y0, y1))

        def df_dbeta_odr(beta, x):
            if False:
                i = 10
                return i + 15
            nr_meas = np.shape(x)[1]
            zeros = np.zeros(nr_meas)
            ones = np.ones(nr_meas)
            dy0 = np.array([ones, x[0, :], x[1, :], zeros, zeros, zeros])
            dy1 = np.array([zeros, zeros, zeros, ones, x[0, :], x[1, :]])
            return np.stack((dy0, dy1))

        def df_dx_odr(beta, x):
            if False:
                return 10
            nr_meas = np.shape(x)[1]
            ones = np.ones(nr_meas)
            dy0 = np.array([beta[1] * ones, beta[2] * ones])
            dy1 = np.array([beta[4] * ones, beta[5] * ones])
            return np.stack((dy0, dy1))
        x0_true = np.linspace(1, 10, nr_measurements)
        x1_true = np.linspace(1, 10, nr_measurements)
        x_true = np.array([x0_true, x1_true])
        y_true = func(beta_true, x_true)
        x_meas = x_true + x_error
        y_meas = y_true + y_error
        model_f = Model(func, fjacb=df_dbeta_odr, fjacd=df_dx_odr)
        data = RealData(x_meas, y_meas, sx=std_dev_x, sy=std_dev_y)
        odr_obj = ODR(data, model_f, beta0=0.9 * beta_true, maxit=100)
        odr_obj.set_job(deriv=3)
        odr_out = odr_obj.run()
        assert_equal(odr_out.info, 1)
        assert_array_almost_equal(odr_out.beta, beta_solution)

    def test_multilinear_model(self):
        if False:
            return 10
        x = np.linspace(0.0, 5.0)
        y = 10.0 + 5.0 * x
        data = Data(x, y)
        odr_obj = ODR(data, multilinear)
        output = odr_obj.run()
        assert_array_almost_equal(output.beta, [10.0, 5.0])

    def test_exponential_model(self):
        if False:
            return 10
        x = np.linspace(0.0, 5.0)
        y = -10.0 + np.exp(0.5 * x)
        data = Data(x, y)
        odr_obj = ODR(data, exponential)
        output = odr_obj.run()
        assert_array_almost_equal(output.beta, [-10.0, 0.5])

    def test_polynomial_model(self):
        if False:
            while True:
                i = 10
        x = np.linspace(0.0, 5.0)
        y = 1.0 + 2.0 * x + 3.0 * x ** 2 + 4.0 * x ** 3
        poly_model = polynomial(3)
        data = Data(x, y)
        odr_obj = ODR(data, poly_model)
        output = odr_obj.run()
        assert_array_almost_equal(output.beta, [1.0, 2.0, 3.0, 4.0])

    def test_unilinear_model(self):
        if False:
            while True:
                i = 10
        x = np.linspace(0.0, 5.0)
        y = 1.0 * x + 2.0
        data = Data(x, y)
        odr_obj = ODR(data, unilinear)
        output = odr_obj.run()
        assert_array_almost_equal(output.beta, [1.0, 2.0])

    def test_quadratic_model(self):
        if False:
            for i in range(10):
                print('nop')
        x = np.linspace(0.0, 5.0)
        y = 1.0 * x ** 2 + 2.0 * x + 3.0
        data = Data(x, y)
        odr_obj = ODR(data, quadratic)
        output = odr_obj.run()
        assert_array_almost_equal(output.beta, [1.0, 2.0, 3.0])

    def test_work_ind(self):
        if False:
            while True:
                i = 10

        def func(par, x):
            if False:
                print('Hello World!')
            (b0, b1) = par
            return b0 + b1 * x
        n_data = 4
        x = np.arange(n_data)
        y = np.where(x % 2, x + 0.1, x - 0.1)
        x_err = np.full(n_data, 0.1)
        y_err = np.full(n_data, 0.1)
        linear_model = Model(func)
        real_data = RealData(x, y, sx=x_err, sy=y_err)
        odr_obj = ODR(real_data, linear_model, beta0=[0.4, 0.4])
        odr_obj.set_job(fit_type=0)
        out = odr_obj.run()
        sd_ind = out.work_ind['sd']
        assert_array_almost_equal(out.sd_beta, out.work[sd_ind:sd_ind + len(out.sd_beta)])

    @pytest.mark.skipif(True, reason='Fortran I/O prone to crashing so better not to run this test, see gh-13127')
    def test_output_file_overwrite(self):
        if False:
            while True:
                i = 10
        '\n        Verify fix for gh-1892\n        '

        def func(b, x):
            if False:
                while True:
                    i = 10
            return b[0] + b[1] * x
        p = Model(func)
        data = Data(np.arange(10), 12 * np.arange(10))
        tmp_dir = tempfile.mkdtemp()
        error_file_path = os.path.join(tmp_dir, 'error.dat')
        report_file_path = os.path.join(tmp_dir, 'report.dat')
        try:
            ODR(data, p, beta0=[0.1, 13], errfile=error_file_path, rptfile=report_file_path).run()
            ODR(data, p, beta0=[0.1, 13], errfile=error_file_path, rptfile=report_file_path, overwrite=True).run()
        finally:
            shutil.rmtree(tmp_dir)

    def test_odr_model_default_meta(self):
        if False:
            return 10

        def func(b, x):
            if False:
                while True:
                    i = 10
            return b[0] + b[1] * x
        p = Model(func)
        p.set_meta(name='Sample Model Meta', ref='ODRPACK')
        assert_equal(p.meta, {'name': 'Sample Model Meta', 'ref': 'ODRPACK'})

    def test_work_array_del_init(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Verify fix for gh-18739 where del_init=1 fails.\n        '

        def func(b, x):
            if False:
                i = 10
                return i + 15
            return b[0] + b[1] * x
        n_data = 4
        x = np.arange(n_data)
        y = np.where(x % 2, x + 0.1, x - 0.1)
        x_err = np.full(n_data, 0.1)
        y_err = np.full(n_data, 0.1)
        linear_model = Model(func)
        rd0 = RealData(x, y, sx=x_err, sy=y_err)
        rd1 = RealData(x, y, sx=x_err, sy=0.1)
        rd2 = RealData(x, y, sx=x_err, sy=[0.1])
        rd3 = RealData(x, y, sx=x_err, sy=np.full((1, n_data), 0.1))
        rd4 = RealData(x, y, sx=x_err, covy=[[0.01]])
        rd5 = RealData(x, y, sx=x_err, covy=np.full((1, 1, n_data), 0.01))
        for rd in [rd0, rd1, rd2, rd3, rd4, rd5]:
            odr_obj = ODR(rd, linear_model, beta0=[0.4, 0.4], delta0=np.full(n_data, -0.1))
            odr_obj.set_job(fit_type=0, del_init=1)
            odr_obj.run()