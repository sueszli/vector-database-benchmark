import numpy as np
from numpy.testing import assert_, assert_allclose, assert_raises
import statsmodels.datasets.macrodata.data as macro
from statsmodels.tsa.vector_ar.tests.JMulTi_results.parse_jmulti_vecm_output import sublists
from statsmodels.tsa.vector_ar.var_model import VAR
from .JMulTi_results.parse_jmulti_var_output import dt_s_tup_to_string, load_results_jmulti
atol = 0.001
rtol = 0.01
datasets = []
data = {}
results_ref = {}
results_sm = {}
debug_mode = False
dont_test_se_t_p = False
deterministic_terms_list = ['nc', 'c', 'ct']
seasonal_list = [0, 4]
dt_s_list = [(det, s) for det in deterministic_terms_list for s in seasonal_list]
all_tests = ['coefs', 'det', 'Sigma_u', 'log_like', 'fc', 'causality', 'impulse-response', 'lag order', 'test normality', 'whiteness', 'exceptions']
to_test = all_tests

def load_data(dataset, data_dict):
    if False:
        i = 10
        return i + 15
    dtset = dataset.load_pandas()
    variables = dataset.variable_names
    loaded = dtset.data[variables].astype(float).values
    data_dict[dataset] = loaded.reshape((-1, len(variables)))

def reorder_jmultis_det_terms(jmulti_output, constant, seasons):
    if False:
        print('Hello World!')
    "\n    In case of seasonal terms and a trend term we have to reorder them to make\n    the outputs from JMulTi and statsmodels comparable.\n    JMulTi's ordering is: [constant], [seasonal terms], [trend term] while\n    in statsmodels it is: [constant], [trend term], [seasonal terms]\n\n    Parameters\n    ----------\n    jmulti_output : ndarray (neqs x number_of_deterministic_terms)\n\n    constant : bool\n        Indicates whether there is a constant term or not in jmulti_output.\n    seasons : int\n        Number of seasons in the model. That means there are seasons-1\n        columns for seasonal terms in jmulti_output\n\n    Returns\n    -------\n    reordered : ndarray (neqs x number_of_deterministic_terms)\n        jmulti_output reordered such that the order of deterministic terms\n        matches that of statsmodels.\n    "
    if seasons == 0:
        return jmulti_output
    constant = int(constant)
    const_column = jmulti_output[:, :constant]
    season_columns = jmulti_output[:, constant:constant + seasons - 1].copy()
    trend_columns = jmulti_output[:, constant + seasons - 1:].copy()
    return np.hstack((const_column, trend_columns, season_columns))

def generate_exog_from_season(seasons, endog_len):
    if False:
        for i in range(10):
            print('nop')
    '\n    Translate seasons to exog matrix.\n\n    Parameters\n    ----------\n    seasons : int\n        Number of seasons.\n    endog_len : int\n        Number of observations.\n\n    Returns\n    -------\n    exog : ndarray or None\n        If seasonal deterministic terms exist, the corresponding exog-matrix is\n        returned.\n        Otherwise, None is returned.\n    '
    exog_stack = []
    if seasons > 0:
        season_exog = np.zeros((seasons - 1, endog_len))
        for i in range(seasons - 1):
            season_exog[i, i::seasons] = 1
        season_exog = season_exog.T
        exog_stack.append(season_exog)
    if exog_stack != []:
        exog = np.column_stack(exog_stack)
    else:
        exog = None
    return exog

def load_results_statsmodels(dataset):
    if False:
        print('Hello World!')
    results_per_deterministic_terms = dict.fromkeys(dt_s_list)
    for dt_s_tup in dt_s_list:
        endog = data[dataset]
        exog = generate_exog_from_season(dt_s_tup[1], len(endog))
        model = VAR(endog, exog)
        trend = dt_s_tup[0] if dt_s_tup[0] != 'nc' else 'n'
        results_per_deterministic_terms[dt_s_tup] = model.fit(maxlags=4, trend=trend, method='ols')
    return results_per_deterministic_terms

def build_err_msg(ds, dt_s, parameter_str):
    if False:
        i = 10
        return i + 15
    dt = dt_s_tup_to_string(dt_s)
    seasons = dt_s[1]
    err_msg = 'Error in ' + parameter_str + ' for:\n'
    err_msg += '- Dataset: ' + ds.__str__() + '\n'
    err_msg += '- Deterministic terms: '
    err_msg += dt_s[0] if dt != 'nc' else 'no det. terms'
    if seasons > 0:
        err_msg += ', seasons: ' + str(seasons)
    return err_msg

def setup():
    if False:
        i = 10
        return i + 15
    datasets.append(macro)
    for ds in datasets:
        load_data(ds, data)
        results_ref[ds] = load_results_jmulti(ds, dt_s_list)
        results_sm[ds] = load_results_statsmodels(ds)
setup()

def test_ols_coefs():
    if False:
        for i in range(10):
            print('nop')
    if debug_mode:
        if 'coefs' not in to_test:
            return
        print('\n\nESTIMATED PARAMETER MATRICES FOR LAGGED ENDOG', end='')
    for ds in datasets:
        for dt_s in dt_s_list:
            if debug_mode:
                print('\n' + dt_s_tup_to_string(dt_s) + ': ', end='')
            err_msg = build_err_msg(ds, dt_s, 'PARAMETER MATRICES ENDOG')
            obtained = np.hstack(results_sm[ds][dt_s].coefs)
            desired = results_ref[ds][dt_s]['est']['Lagged endogenous term']
            assert_allclose(obtained, desired, rtol, atol, False, err_msg)
            if debug_mode and dont_test_se_t_p:
                continue
            obt = results_sm[ds][dt_s].stderr_endog_lagged
            des = results_ref[ds][dt_s]['se']['Lagged endogenous term'].T
            assert_allclose(obt, des, rtol, atol, False, 'STANDARD ERRORS\n' + err_msg)
            obt = results_sm[ds][dt_s].tvalues_endog_lagged
            des = results_ref[ds][dt_s]['t']['Lagged endogenous term'].T
            assert_allclose(obt, des, rtol, atol, False, 't-VALUES\n' + err_msg)
            obt = results_sm[ds][dt_s].pvalues_endog_lagged
            des = results_ref[ds][dt_s]['p']['Lagged endogenous term'].T
            assert_allclose(obt, des, rtol, atol, False, 'p-VALUES\n' + err_msg)

def test_ols_det_terms():
    if False:
        print('Hello World!')
    if debug_mode:
        if 'det' not in to_test:
            return
        print('\n\nESTIMATED PARAMETERS FOR DETERMINISTIC TERMS', end='')
    for ds in datasets:
        for dt_s in dt_s_list:
            if debug_mode:
                print('\n' + dt_s_tup_to_string(dt_s) + ': ', end='')
            err_msg = build_err_msg(ds, dt_s, 'PARAMETER MATRICES EXOG')
            det_key_ref = 'Deterministic term'
            if det_key_ref not in results_ref[ds][dt_s]['est'].keys():
                assert_(results_sm[ds][dt_s].coefs_exog.size == 0 and results_sm[ds][dt_s].stderr_dt.size == 0 and (results_sm[ds][dt_s].tvalues_dt.size == 0) and (results_sm[ds][dt_s].pvalues_dt.size == 0), err_msg)
                continue
            obtained = results_sm[ds][dt_s].coefs_exog
            desired = results_ref[ds][dt_s]['est'][det_key_ref]
            desired = reorder_jmultis_det_terms(desired, dt_s[0].startswith('c'), dt_s[1])
            assert_allclose(obtained, desired, rtol, atol, False, err_msg)
            if debug_mode and dont_test_se_t_p:
                continue
            obt = results_sm[ds][dt_s].stderr_dt
            des = results_ref[ds][dt_s]['se'][det_key_ref]
            des = reorder_jmultis_det_terms(des, dt_s[0].startswith('c'), dt_s[1]).T
            assert_allclose(obt, des, rtol, atol, False, 'STANDARD ERRORS\n' + err_msg)
            obt = results_sm[ds][dt_s].tvalues_dt
            des = results_ref[ds][dt_s]['t'][det_key_ref]
            des = reorder_jmultis_det_terms(des, dt_s[0].startswith('c'), dt_s[1]).T
            assert_allclose(obt, des, rtol, atol, False, 't-VALUES\n' + err_msg)
            obt = results_sm[ds][dt_s].pvalues_dt
            des = results_ref[ds][dt_s]['p'][det_key_ref]
            des = reorder_jmultis_det_terms(des, dt_s[0].startswith('c'), dt_s[1]).T
            assert_allclose(obt, des, rtol, atol, False, 'p-VALUES\n' + err_msg)

def test_ols_sigma():
    if False:
        i = 10
        return i + 15
    if debug_mode:
        if 'Sigma_u' not in to_test:
            return
        print('\n\nSIGMA_U', end='')
    for ds in datasets:
        for dt in dt_s_list:
            if debug_mode:
                print('\n' + dt_s_tup_to_string(dt) + ': ', end='')
            err_msg = build_err_msg(ds, dt, 'Sigma_u')
            obtained = results_sm[ds][dt].sigma_u
            desired = results_ref[ds][dt]['est']['Sigma_u']
            assert_allclose(obtained, desired, rtol, atol, False, err_msg)

def test_log_like():
    if False:
        i = 10
        return i + 15
    if debug_mode:
        if 'log_like' not in to_test:
            return
        else:
            print('\n\nLOG LIKELIHOOD', end='')
    for ds in datasets:
        for dt in dt_s_list:
            if debug_mode:
                print('\n' + dt_s_tup_to_string(dt) + ': ', end='')
            err_msg = build_err_msg(ds, dt, 'Log Likelihood')
            obtained = results_sm[ds][dt].llf
            desired = results_ref[ds][dt]['log_like']
            assert_allclose(obtained, desired, rtol, atol, False, err_msg)

def test_fc():
    if False:
        for i in range(10):
            print('nop')
    if debug_mode:
        if 'fc' not in to_test:
            return
        else:
            print('\n\nFORECAST', end='')
    for ds in datasets:
        for dt in dt_s_list:
            if debug_mode:
                print('\n' + dt_s_tup_to_string(dt) + ': ', end='')
            steps = 5
            last_observations = results_sm[ds][dt].endog[-results_sm[ds][dt].k_ar:]
            seasons = dt[1]
            if seasons == 0:
                exog_future = None
            else:
                exog_future = np.zeros((steps, seasons - 1))
                exog_future[1:seasons] = np.identity(seasons - 1)
            err_msg = build_err_msg(ds, dt, 'FORECAST')
            obtained = results_sm[ds][dt].forecast(y=last_observations, steps=steps, exog_future=exog_future)
            desired = results_ref[ds][dt]['fc']['fc']
            assert_allclose(obtained, desired, rtol, atol, False, err_msg)
            err_msg = build_err_msg(ds, dt, 'FORECAST WITH INTERVALS')
            obtained = results_sm[ds][dt].forecast_interval(y=last_observations, steps=steps, alpha=0.05, exog_future=exog_future)
            obt = obtained[0]
            obt_l = obtained[1]
            obt_u = obtained[2]
            des = results_ref[ds][dt]['fc']['fc']
            des_l = results_ref[ds][dt]['fc']['lower']
            des_u = results_ref[ds][dt]['fc']['upper']
            assert_allclose(obt, des, rtol, atol, False, err_msg)
            assert_allclose(obt_l, des_l, rtol, atol, False, err_msg)
            assert_allclose(obt_u, des_u, rtol, atol, False, err_msg)

def test_causality():
    if False:
        for i in range(10):
            print('nop')
    if debug_mode:
        if 'causality' not in to_test:
            return
        else:
            print('\n\nCAUSALITY', end='')
    for ds in datasets:
        for dt in dt_s_list:
            if debug_mode:
                print('\n' + dt_s_tup_to_string(dt) + ': ', end='')
            err_msg_g_p = build_err_msg(ds, dt, 'GRANGER CAUS. - p-VALUE')
            err_msg_g_t = build_err_msg(ds, dt, 'GRANGER CAUS. - TEST STAT.')
            err_msg_i_p = build_err_msg(ds, dt, 'INSTANT. CAUS. - p-VALUE')
            err_msg_i_t = build_err_msg(ds, dt, 'INSTANT. CAUS. - TEST STAT.')
            v_ind = range(len(ds.variable_names))
            for causing_ind in sublists(v_ind, 1, len(v_ind) - 1):
                causing_names = ['y' + str(i + 1) for i in causing_ind]
                causing_key = tuple((ds.variable_names[i] for i in causing_ind))
                caused_ind = [i for i in v_ind if i not in causing_ind]
                caused_names = ['y' + str(i + 1) for i in caused_ind]
                caused_key = tuple((ds.variable_names[i] for i in caused_ind))
                granger_sm_ind = results_sm[ds][dt].test_causality(caused_ind, causing_ind)
                granger_sm_str = results_sm[ds][dt].test_causality(caused_names, causing_names)
                g_t_obt = granger_sm_ind.test_statistic
                g_t_des = results_ref[ds][dt]['granger_caus']['test_stat'][causing_key, caused_key]
                assert_allclose(g_t_obt, g_t_des, rtol, atol, False, err_msg_g_t)
                g_t_obt_str = granger_sm_str.test_statistic
                assert_allclose(g_t_obt_str, g_t_obt, 1e-07, 0, False, err_msg_g_t + ' - sequences of integers and '.upper() + 'strings as arguments do not yield the same result!'.upper())
                if len(causing_ind) == 1 or len(caused_ind) == 1:
                    ci = causing_ind[0] if len(causing_ind) == 1 else causing_ind
                    ce = caused_ind[0] if len(caused_ind) == 1 else caused_ind
                    granger_sm_single_ind = results_sm[ds][dt].test_causality(ce, ci)
                    g_t_obt_single = granger_sm_single_ind.test_statistic
                    assert_allclose(g_t_obt_single, g_t_obt, 1e-07, 0, False, err_msg_g_t + ' - list of int and int as '.upper() + 'argument do not yield the same result!'.upper())
                g_p_obt = granger_sm_ind.pvalue
                g_p_des = results_ref[ds][dt]['granger_caus']['p'][causing_key, caused_key]
                assert_allclose(g_p_obt, g_p_des, rtol, atol, False, err_msg_g_p)
                g_p_obt_str = granger_sm_str.pvalue
                assert_allclose(g_p_obt_str, g_p_obt, 1e-07, 0, False, err_msg_g_t + ' - sequences of integers and '.upper() + 'strings as arguments do not yield the same result!'.upper())
                if len(causing_ind) == 1:
                    g_p_obt_single = granger_sm_single_ind.pvalue
                    assert_allclose(g_p_obt_single, g_p_obt, 1e-07, 0, False, err_msg_g_t + ' - list of int and int as '.upper() + 'argument do not yield the same result!'.upper())
                inst_sm_ind = results_sm[ds][dt].test_inst_causality(causing_ind)
                inst_sm_str = results_sm[ds][dt].test_inst_causality(causing_names)
                t_obt = inst_sm_ind.test_statistic
                t_des = results_ref[ds][dt]['inst_caus']['test_stat'][causing_key, caused_key]
                assert_allclose(t_obt, t_des, rtol, atol, False, err_msg_i_t)
                t_obt_str = inst_sm_str.test_statistic
                assert_allclose(t_obt_str, t_obt, 1e-07, 0, False, err_msg_i_t + ' - sequences of integers and '.upper() + 'strings as arguments do not yield the same result!'.upper())
                if len(causing_ind) == 1:
                    inst_sm_single_ind = results_sm[ds][dt].test_inst_causality(causing_ind[0])
                    t_obt_single = inst_sm_single_ind.test_statistic
                    assert_allclose(t_obt_single, t_obt, 1e-07, 0, False, err_msg_i_t + ' - list of int and int as '.upper() + 'argument do not yield the same result!'.upper())
                p_obt = results_sm[ds][dt].test_inst_causality(causing_ind).pvalue
                p_des = results_ref[ds][dt]['inst_caus']['p'][causing_key, caused_key]
                assert_allclose(p_obt, p_des, rtol, atol, False, err_msg_i_p)
                p_obt_str = inst_sm_str.pvalue
                assert_allclose(p_obt_str, p_obt, 1e-07, 0, False, err_msg_i_p + ' - sequences of integers and '.upper() + 'strings as arguments do not yield the same result!'.upper())
                if len(causing_ind) == 1:
                    inst_sm_single_ind = results_sm[ds][dt].test_inst_causality(causing_ind[0])
                    p_obt_single = inst_sm_single_ind.pvalue
                    assert_allclose(p_obt_single, p_obt, 1e-07, 0, False, err_msg_i_p + ' - list of int and int as '.upper() + 'argument do not yield the same result!'.upper())

def test_impulse_response():
    if False:
        i = 10
        return i + 15
    if debug_mode:
        if 'impulse-response' not in to_test:
            return
        else:
            print('\n\nIMPULSE-RESPONSE', end='')
    for ds in datasets:
        for dt in dt_s_list:
            if debug_mode:
                print('\n' + dt_s_tup_to_string(dt) + ': ', end='')
            err_msg = build_err_msg(ds, dt, 'IMULSE-RESPONSE')
            periods = 20
            obtained_all = results_sm[ds][dt].irf(periods=periods).irfs
            obtained_all = obtained_all.reshape(periods + 1, -1)
            desired_all = results_ref[ds][dt]['ir']
            assert_allclose(obtained_all, desired_all, rtol, atol, False, err_msg)

def test_lag_order_selection():
    if False:
        for i in range(10):
            print('nop')
    if debug_mode:
        if 'lag order' not in to_test:
            return
        else:
            print('\n\nLAG ORDER SELECTION', end='')
    for ds in datasets:
        for dt in dt_s_list:
            if debug_mode:
                print('\n' + dt_s_tup_to_string(dt) + ': ', end='')
            endog_tot = data[ds]
            exog = generate_exog_from_season(dt[1], len(endog_tot))
            model = VAR(endog_tot, exog)
            trend = 'n' if dt[0] == 'nc' else dt[0]
            obtained_all = model.select_order(10, trend=trend)
            for ic in ['aic', 'fpe', 'hqic', 'bic']:
                err_msg = build_err_msg(ds, dt, 'LAG ORDER SELECTION - ' + ic.upper())
                obtained = getattr(obtained_all, ic)
                desired = results_ref[ds][dt]['lagorder'][ic]
                assert_allclose(obtained, desired, rtol, atol, False, err_msg)

def test_normality():
    if False:
        return 10
    if debug_mode:
        if 'test normality' not in to_test:
            return
        else:
            print('\n\nTEST NON-NORMALITY', end='')
    for ds in datasets:
        for dt in dt_s_list:
            if debug_mode:
                print('\n' + dt_s_tup_to_string(dt) + ': ', end='')
            obtained = results_sm[ds][dt].test_normality(signif=0.05)
            err_msg = build_err_msg(ds, dt, 'TEST NON-NORMALITY - STATISTIC')
            obt_statistic = obtained.test_statistic
            des_statistic = results_ref[ds][dt]['test_norm']['joint_test_statistic']
            assert_allclose(obt_statistic, des_statistic, rtol, atol, False, err_msg)
            err_msg = build_err_msg(ds, dt, 'TEST NON-NORMALITY - P-VALUE')
            obt_pvalue = obtained.pvalue
            des_pvalue = results_ref[ds][dt]['test_norm']['joint_pvalue']
            assert_allclose(obt_pvalue, des_pvalue, rtol, atol, False, err_msg)
            obtained.summary()
            str(obtained)

def test_whiteness():
    if False:
        while True:
            i = 10
    if debug_mode:
        if 'whiteness' not in to_test:
            return
        else:
            print('\n\nTEST WHITENESS OF RESIDUALS', end='')
    for ds in datasets:
        for dt in dt_s_list:
            if debug_mode:
                print('\n' + dt_s_tup_to_string(dt) + ': ', end='')
            lags = results_ref[ds][dt]['whiteness']['tested order']
            obtained = results_sm[ds][dt].test_whiteness(nlags=lags)
            err_msg = build_err_msg(ds, dt, 'WHITENESS OF RESIDUALS - TEST STATISTIC')
            desired = results_ref[ds][dt]['whiteness']['test statistic']
            assert_allclose(obtained.test_statistic, desired, rtol, atol, False, err_msg)
            err_msg = build_err_msg(ds, dt, 'WHITENESS OF RESIDUALS - P-VALUE')
            desired = results_ref[ds][dt]['whiteness']['p-value']
            assert_allclose(obtained.pvalue, desired, rtol, atol, False, err_msg)
            obtained = results_sm[ds][dt].test_whiteness(nlags=lags, adjusted=True)
            err_msg = build_err_msg(ds, dt, 'WHITENESS OF RESIDUALS - TEST STATISTIC (ADJUSTED TEST)')
            desired = results_ref[ds][dt]['whiteness']['test statistic adj.']
            assert_allclose(obtained.test_statistic, desired, rtol, atol, False, err_msg)
            err_msg = build_err_msg(ds, dt, 'WHITENESS OF RESIDUALS - P-VALUE (ADJUSTED TEST)')
            desired = results_ref[ds][dt]['whiteness']['p-value adjusted']
            assert_allclose(obtained.pvalue, desired, rtol, atol, False, err_msg)

def test_exceptions():
    if False:
        while True:
            i = 10
    if debug_mode:
        if 'exceptions' not in to_test:
            return
        else:
            print('\n\nEXCEPTIONS\n', end='')
    for ds in datasets:
        for dt in dt_s_list:
            if debug_mode:
                print('\n' + dt_s_tup_to_string(dt) + ': ', end='')
            assert_raises(ValueError, results_sm[ds][dt].test_inst_causality, 0, 0)
            assert_raises(TypeError, results_sm[ds][dt].test_inst_causality, [0.5])