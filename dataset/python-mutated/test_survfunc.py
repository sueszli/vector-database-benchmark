import os
import numpy as np
from statsmodels.duration.survfunc import SurvfuncRight, survdiff, plot_survfunc, CumIncidenceRight
from numpy.testing import assert_allclose
import pandas as pd
import pytest
pdf_output = False
try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

def close_or_save(pdf, fig):
    if False:
        return 10
    if pdf_output:
        pdf.savefig(fig)
'\nlibrary(survival)\nti1 = c(3, 1, 2, 3, 2, 1, 5, 3)\nst1 = c(0, 1, 1, 1, 0, 0, 1, 0)\nti2 = c(1, 1, 2, 3, 7, 1, 5, 3, 9)\nst2 = c(0, 1, 0, 0, 1, 0, 1, 0, 1)\n\nti = c(ti1, ti2)\nst = c(st1, st2)\nix = c(rep(1, length(ti1)), rep(2, length(ti2)))\nsd = survdiff(Surv(ti, st) ~ ix)\n'
ti1 = np.r_[3, 1, 2, 3, 2, 1, 5, 3]
st1 = np.r_[0, 1, 1, 1, 0, 0, 1, 0]
times1 = np.r_[1, 2, 3, 5]
surv_prob1 = np.r_[0.875, 0.7291667, 0.546875, 0.0]
surv_prob_se1 = np.r_[0.1169268, 0.1649762, 0.20058, np.nan]
n_risk1 = np.r_[8, 6, 4, 1]
n_events1 = np.r_[1.0, 1.0, 1.0, 1.0]
ti2 = np.r_[1, 1, 2, 3, 7, 1, 5, 3, 9]
st2 = np.r_[0, 1, 0, 0, 1, 0, 1, 0, 1]
times2 = np.r_[1, 5, 7, 9]
surv_prob2 = np.r_[0.8888889, 0.5925926, 0.2962963, 0.0]
surv_prob_se2 = np.r_[0.1047566, 0.2518034, 0.244432, np.nan]
n_risk2 = np.r_[9, 3, 2, 1]
n_events2 = np.r_[1.0, 1.0, 1.0, 1.0]
cur_dir = os.path.dirname(os.path.abspath(__file__))
fp = os.path.join(cur_dir, 'results', 'bmt.csv')
bmt = pd.read_csv(fp)

def test_survfunc1():
    if False:
        while True:
            i = 10
    sr = SurvfuncRight(ti1, st1)
    assert_allclose(sr.surv_prob, surv_prob1, atol=1e-05, rtol=1e-05)
    assert_allclose(sr.surv_prob_se, surv_prob_se1, atol=1e-05, rtol=1e-05)
    assert_allclose(sr.surv_times, times1)
    assert_allclose(sr.n_risk, n_risk1)
    assert_allclose(sr.n_events, n_events1)

def test_survfunc2():
    if False:
        i = 10
        return i + 15
    sr = SurvfuncRight(ti2, st2)
    assert_allclose(sr.surv_prob, surv_prob2, atol=1e-05, rtol=1e-05)
    assert_allclose(sr.surv_prob_se, surv_prob_se2, atol=1e-05, rtol=1e-05)
    assert_allclose(sr.surv_times, times2)
    assert_allclose(sr.n_risk, n_risk2)
    assert_allclose(sr.n_events, n_events2)

def test_survdiff_basic():
    if False:
        i = 10
        return i + 15
    ti = np.concatenate((ti1, ti2))
    st = np.concatenate((st1, st2))
    groups = np.ones(len(ti))
    groups[0:len(ti1)] = 0
    (z, p) = survdiff(ti, st, groups)
    assert_allclose(z, 2.14673, atol=0.0001, rtol=0.0001)
    assert_allclose(p, 0.14287, atol=0.0001, rtol=0.0001)

def test_simultaneous_cb():
    if False:
        i = 10
        return i + 15
    df = bmt.loc[bmt['Group'] == 'ALL', :]
    sf = SurvfuncRight(df['T'], df['Status'])
    (lcb1, ucb1) = sf.simultaneous_cb(transform='log')
    (lcb2, ucb2) = sf.simultaneous_cb(transform='arcsin')
    ti = sf.surv_times.tolist()
    ix = [ti.index(x) for x in (110, 122, 129, 172)]
    assert_allclose(lcb1[ix], np.r_[0.43590582, 0.42115592, 0.4035897, 0.38785927])
    assert_allclose(ucb1[ix], np.r_[0.93491636, 0.89776803, 0.87922239, 0.85894181])
    assert_allclose(lcb2[ix], np.r_[0.52115708, 0.48079378, 0.45595321, 0.43341115])
    assert_allclose(ucb2[ix], np.r_[0.96465636, 0.92745068, 0.90885428, 0.88796708])

def test_bmt():
    if False:
        print('Hello World!')
    cb = {'linear': [107, 276], 'cloglog': [86, 230], 'log': [107, 332], 'asinsqrt': [104, 276], 'logit': [104, 230]}
    dfa = bmt[bmt.Group == 'ALL']
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    fp = os.path.join(cur_dir, 'results', 'bmt_results.csv')
    rslt = pd.read_csv(fp)
    sf = SurvfuncRight(dfa['T'].values, dfa.Status.values)
    assert_allclose(sf.surv_times, rslt.t)
    assert_allclose(sf.surv_prob, rslt.s, atol=0.0001, rtol=0.0001)
    assert_allclose(sf.surv_prob_se, rslt.se, atol=0.0001, rtol=0.0001)
    for method in ('linear', 'cloglog', 'log', 'logit', 'asinsqrt'):
        (lcb, ucb) = sf.quantile_ci(0.25, method=method)
        assert_allclose(cb[method], np.r_[lcb, ucb])

def test_survdiff():
    if False:
        i = 10
        return i + 15
    full_df = bmt.copy()
    df = bmt[bmt.Group != 'ALL'].copy()
    (stat, p) = survdiff(df['T'], df.Status, df.Group)
    assert_allclose(stat, 13.44556, atol=0.0001, rtol=0.0001)
    (stat, p) = survdiff(df['T'], df.Status, df.Group, weight_type='gb')
    assert_allclose(stat, 15.38787, atol=0.0001, rtol=0.0001)
    (stat, p) = survdiff(df['T'], df.Status, df.Group, weight_type='tw')
    assert_allclose(stat, 14.98382, atol=0.0001, rtol=0.0001)
    (stat, p) = survdiff(df['T'], df.Status, df.Group, weight_type='fh', fh_p=0.5)
    assert_allclose(stat, 14.46866, atol=0.0001, rtol=0.0001)
    (stat, p) = survdiff(df['T'], df.Status, df.Group, weight_type='fh', fh_p=1)
    assert_allclose(stat, 14.845, atol=0.0001, rtol=0.0001)
    (stat, p) = survdiff(full_df['T'], full_df.Status, full_df.Group, weight_type='fh', fh_p=1)
    assert_allclose(stat, 15.67247, atol=0.0001, rtol=0.0001)
    strata = np.arange(df.shape[0]) % 5
    df['strata'] = strata
    (stat, p) = survdiff(df['T'], df.Status, df.Group, strata=df.strata)
    assert_allclose(stat, 11.97799, atol=0.0001, rtol=0.0001)
    (stat, p) = survdiff(df['T'], df.Status, df.Group, strata=df.strata, weight_type='fh', fh_p=0.5)
    assert_allclose(stat, 12.6257, atol=0.0001, rtol=0.0001)
    (stat, p) = survdiff(df['T'], df.Status, df.Group, strata=df.strata, weight_type='fh', fh_p=1)
    assert_allclose(stat, 12.73565, atol=0.0001, rtol=0.0001)
    full_strata = np.arange(full_df.shape[0]) % 5
    full_df['strata'] = full_strata
    (stat, p) = survdiff(full_df['T'], full_df.Status, full_df.Group, strata=full_df.strata, weight_type='fh', fh_p=0.5)
    assert_allclose(stat, 13.56793, atol=0.0001, rtol=0.0001)
    df['strata'] = np.arange(df.shape[0]) % 8
    (stat, p) = survdiff(df['T'], df.Status, df.Group, strata=df.strata)
    assert_allclose(stat, 12.12631, atol=0.0001, rtol=0.0001)
    (stat, p) = survdiff(df['T'], df.Status, df.Group, strata=df.strata, weight_type='fh', fh_p=0.5)
    assert_allclose(stat, 12.9633, atol=0.0001, rtol=0.0001)
    (stat, p) = survdiff(df['T'], df.Status, df.Group, strata=df.strata, weight_type='fh', fh_p=1)
    assert_allclose(stat, 13.35259, atol=0.0001, rtol=0.0001)

@pytest.mark.matplotlib
def test_plot_km(close_figures):
    if False:
        while True:
            i = 10
    if pdf_output:
        from matplotlib.backends.backend_pdf import PdfPages
        pdf = PdfPages('test_survfunc.pdf')
    else:
        pdf = None
    sr1 = SurvfuncRight(ti1, st1)
    sr2 = SurvfuncRight(ti2, st2)
    fig = plot_survfunc(sr1)
    close_or_save(pdf, fig)
    fig = plot_survfunc(sr2)
    close_or_save(pdf, fig)
    fig = plot_survfunc([sr1, sr2])
    close_or_save(pdf, fig)
    gb = bmt.groupby('Group')
    sv = []
    for g in gb:
        s0 = SurvfuncRight(g[1]['T'], g[1]['Status'], title=g[0])
        sv.append(s0)
    fig = plot_survfunc(sv)
    ax = fig.get_axes()[0]
    ax.set_position([0.1, 0.1, 0.64, 0.8])
    (ha, lb) = ax.get_legend_handles_labels()
    fig.legend([ha[k] for k in (0, 2, 4)], [lb[k] for k in (0, 2, 4)], loc='center right')
    close_or_save(pdf, fig)
    ii = bmt.Group == 'ALL'
    sf = SurvfuncRight(bmt.loc[ii, 'T'], bmt.loc[ii, 'Status'])
    fig = sf.plot()
    ax = fig.get_axes()[0]
    ax.set_position([0.1, 0.1, 0.64, 0.8])
    (ha, lb) = ax.get_legend_handles_labels()
    (lcb, ucb) = sf.simultaneous_cb(transform='log')
    plt.fill_between(sf.surv_times, lcb, ucb, color='lightgrey')
    (lcb, ucb) = sf.simultaneous_cb(transform='arcsin')
    plt.plot(sf.surv_times, lcb, color='darkgrey')
    plt.plot(sf.surv_times, ucb, color='darkgrey')
    plt.plot(sf.surv_times, sf.surv_prob - 2 * sf.surv_prob_se, color='red')
    plt.plot(sf.surv_times, sf.surv_prob + 2 * sf.surv_prob_se, color='red')
    plt.xlim(100, 600)
    close_or_save(pdf, fig)
    if pdf_output:
        pdf.close()

def test_weights1():
    if False:
        for i in range(10):
            print('nop')
    tm = np.r_[1, 3, 5, 6, 7, 8, 8, 9, 3, 4, 1, 3, 2]
    st = np.r_[1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0]
    wt = np.r_[1, 2, 3, 2, 3, 1, 2, 1, 1, 2, 2, 3, 1]
    sf = SurvfuncRight(tm, st, freq_weights=wt)
    assert_allclose(sf.surv_times, np.r_[1, 3, 6, 7, 9])
    assert_allclose(sf.surv_prob, np.r_[0.875, 0.65625, 0.51041667, 0.29166667, 0.0])
    assert_allclose(sf.surv_prob_se, np.r_[0.07216878, 0.13307266, 0.20591185, 0.3219071, 1.05053519])

def test_weights2():
    if False:
        while True:
            i = 10
    tm = np.r_[1, 3, 5, 6, 7, 2, 4, 6, 8, 10]
    st = np.r_[1, 1, 0, 1, 1, 1, 1, 0, 1, 1]
    wt = np.r_[1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    tm0 = np.r_[1, 3, 5, 6, 7, 2, 4, 6, 8, 10, 2, 4, 6, 8, 10]
    st0 = np.r_[1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1]
    sf0 = SurvfuncRight(tm, st, freq_weights=wt)
    sf1 = SurvfuncRight(tm0, st0)
    assert_allclose(sf0.surv_times, sf1.surv_times)
    assert_allclose(sf0.surv_prob, sf1.surv_prob)
    assert_allclose(sf0.surv_prob_se, np.r_[0.06666667, 0.1210311, 0.14694547, 0.19524829, 0.23183377, 0.30618115, 0.46770386, 0.84778942])

def test_incidence():
    if False:
        while True:
            i = 10
    ftime = np.r_[1, 1, 2, 4, 4, 4, 6, 6, 7, 8, 9, 9, 9, 1, 2, 2, 4, 4]
    fstat = np.r_[1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ci = CumIncidenceRight(ftime, fstat)
    cinc = [np.array([0.11111111, 0.17037037, 0.17037037, 0.17037037, 0.17037037, 0.17037037, 0.17037037]), np.array([0.0, 0.0, 0.20740741, 0.20740741, 0.20740741, 0.20740741, 0.20740741]), np.array([0.0, 0.0, 0.0, 0.17777778, 0.26666667, 0.26666667, 0.26666667])]
    assert_allclose(cinc[0], ci.cinc[0])
    assert_allclose(cinc[1], ci.cinc[1])
    assert_allclose(cinc[2], ci.cinc[2])
    cinc_se = [np.array([0.07407407, 0.08976251, 0.08976251, 0.08976251, 0.08976251, 0.08976251, 0.08976251]), np.array([0.0, 0.0, 0.10610391, 0.10610391, 0.10610391, 0.10610391, 0.10610391]), np.array([0.0, 0.0, 0.0, 0.11196147, 0.12787781, 0.12787781, 0.12787781])]
    assert_allclose(cinc_se[0], ci.cinc_se[0])
    assert_allclose(cinc_se[1], ci.cinc_se[1])
    assert_allclose(cinc_se[2], ci.cinc_se[2])
    weights = np.ones(len(ftime))
    ciw = CumIncidenceRight(ftime, fstat, freq_weights=weights)
    assert_allclose(ci.cinc[0], ciw.cinc[0])
    assert_allclose(ci.cinc[1], ciw.cinc[1])
    assert_allclose(ci.cinc[2], ciw.cinc[2])

def test_survfunc_entry_1():
    if False:
        while True:
            i = 10
    times = np.r_[1, 3, 3, 5, 5, 7, 7, 8, 8, 9, 10, 10]
    status = np.r_[1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1]
    entry = np.r_[0, 1, 1, 2, 2, 2, 3, 4, 4, 4, 4, 0]
    sf = SurvfuncRight(times, status, entry=entry)
    assert_allclose(sf.n_risk, np.r_[2, 6, 9, 7, 5, 3, 2])
    assert_allclose(sf.surv_times, np.r_[1, 3, 5, 7, 8, 9, 10])
    assert_allclose(sf.surv_prob, np.r_[0.5, 0.4167, 0.3241, 0.2778, 0.2222, 0.1481, 0.0741], atol=0.0001)
    assert_allclose(sf.surv_prob_se, np.r_[0.3536, 0.3043, 0.2436, 0.2132, 0.1776, 0.133, 0.0846], atol=0.0001)

def test_survfunc_entry_2():
    if False:
        while True:
            i = 10
    times = np.r_[1, 3, 3, 5, 5, 7, 7, 8, 8, 9, 10, 10]
    status = np.r_[1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1]
    entry = np.r_[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    sf = SurvfuncRight(times, status, entry=entry)
    sf0 = SurvfuncRight(times, status)
    assert_allclose(sf.n_risk, sf0.n_risk)
    assert_allclose(sf.surv_times, sf0.surv_times)
    assert_allclose(sf.surv_prob, sf0.surv_prob)
    assert_allclose(sf.surv_prob_se, sf0.surv_prob_se)

def test_survfunc_entry_3():
    if False:
        while True:
            i = 10
    times = np.r_[1, 2, 5, 6, 6, 6, 6, 6, 9]
    status = np.r_[0, 0, 1, 1, 1, 0, 1, 1, 0]
    entry = np.r_[0, 1, 1, 2, 2, 2, 3, 4, 4]
    sf = SurvfuncRight(times, status, entry=entry)
    assert_allclose(sf.n_risk, np.r_[7, 6])
    assert_allclose(sf.surv_times, np.r_[5, 6])
    assert_allclose(sf.surv_prob, np.r_[0.857143, 0.285714], atol=1e-05)
    assert_allclose(sf.surv_prob_se, np.r_[0.13226, 0.170747], atol=1e-05)

def test_survdiff_entry_1():
    if False:
        while True:
            i = 10
    ti = np.r_[1, 3, 4, 2, 5, 4, 6, 7, 5, 9]
    st = np.r_[1, 1, 0, 1, 1, 0, 1, 1, 0, 0]
    gr = np.r_[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    entry = np.r_[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    (z1, p1) = survdiff(ti, st, gr, entry=entry)
    (z2, p2) = survdiff(ti, st, gr)
    assert_allclose(z1, z2)
    assert_allclose(p1, p2)

def test_survdiff_entry_2():
    if False:
        i = 10
        return i + 15
    ti = np.r_[5, 3, 4, 2, 5, 4, 6, 7, 5, 9]
    st = np.r_[1, 1, 0, 1, 1, 0, 1, 1, 0, 0]
    gr = np.r_[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    entry = np.r_[1, 2, 2, 1, 3, 3, 5, 4, 2, 5]
    (z, p) = survdiff(ti, st, gr)
    assert_allclose(z, 6.694424)
    assert_allclose(p, 0.00967149)
    (z, p) = survdiff(ti, st, gr, entry=entry)
    assert_allclose(z, 3.0)
    assert_allclose(p, 0.083264516)

def test_survdiff_entry_3():
    if False:
        for i in range(10):
            print('nop')
    ti = np.r_[2, 1, 5, 8, 7, 8, 8, 9, 4, 9]
    st = np.r_[1, 1, 1, 1, 1, 0, 1, 0, 0, 0]
    gr = np.r_[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    entry = np.r_[1, 1, 2, 2, 3, 3, 2, 1, 2, 0]
    (z, p) = survdiff(ti, st, gr)
    assert_allclose(z, 6.9543024)
    assert_allclose(p, 0.008361789)
    (z, p) = survdiff(ti, st, gr, entry=entry)
    assert_allclose(z, 6.75082959)
    assert_allclose(p, 0.00937041)

def test_incidence2():
    if False:
        for i in range(10):
            print('nop')
    np.random.seed(2423)
    n = 200
    time = -np.log(np.random.uniform(size=n))
    status = np.random.randint(0, 3, size=n)
    ii = np.argsort(time)
    time = time[ii]
    status = status[ii]
    ci = CumIncidenceRight(time, status)
    statusa = 1 * (status >= 1)
    sf = SurvfuncRight(time, statusa)
    x = 1 - sf.surv_prob
    y = (ci.cinc[0] + ci.cinc[1])[np.flatnonzero(statusa)]
    assert_allclose(x, y)

def test_kernel_survfunc1():
    if False:
        print('Hello World!')
    n = 100
    np.random.seed(3434)
    x = np.random.normal(size=(n, 3))
    time = np.random.uniform(size=n)
    status = np.random.randint(0, 2, size=n)
    result = SurvfuncRight(time, status, exog=x)
    timex = np.r_[0.30721103, 0.0515439, 0.69246897, 0.16446079, 0.31308528]
    sprob = np.r_[0.98948277, 0.98162275, 0.97129237, 0.96044668, 0.95030368]
    assert_allclose(result.time[0:5], timex)
    assert_allclose(result.surv_prob[0:5], sprob)

def test_kernel_survfunc2():
    if False:
        return 10
    n = 100
    np.random.seed(3434)
    x = np.random.normal(size=(n, 3))
    time = np.random.uniform(0, 10, size=n)
    status = np.random.randint(0, 2, size=n)
    resultkm = SurvfuncRight(time, status)
    result = SurvfuncRight(time, status, exog=x, bw_factor=10000)
    assert_allclose(resultkm.surv_times, result.surv_times)
    assert_allclose(resultkm.surv_prob, result.surv_prob, rtol=1e-06, atol=1e-06)

@pytest.mark.smoke
def test_kernel_survfunc3():
    if False:
        i = 10
        return i + 15
    n = 100
    np.random.seed(3434)
    x = np.random.normal(size=(n, 3))
    time = np.random.randint(0, 10, size=n)
    status = np.random.randint(0, 2, size=n)
    SurvfuncRight(time, status, exog=x, bw_factor=10000)
    SurvfuncRight(time, status, exog=x, bw_factor=np.r_[10000, 10000])

def test_kernel_cumincidence1():
    if False:
        while True:
            i = 10
    n = 100
    np.random.seed(3434)
    x = np.random.normal(size=(n, 3))
    time = np.random.uniform(0, 10, size=n)
    status = np.random.randint(0, 3, size=n)
    result1 = CumIncidenceRight(time, status)
    for dimred in (False, True):
        result2 = CumIncidenceRight(time, status, exog=x, bw_factor=10000, dimred=dimred)
        assert_allclose(result1.times, result2.times)
        for k in (0, 1):
            assert_allclose(result1.cinc[k], result2.cinc[k], rtol=1e-05)

@pytest.mark.smoke
def test_kernel_cumincidence2():
    if False:
        while True:
            i = 10
    n = 100
    np.random.seed(3434)
    x = np.random.normal(size=(n, 3))
    time = np.random.randint(0, 10, size=n)
    status = np.random.randint(0, 3, size=n)
    CumIncidenceRight(time, status, exog=x, bw_factor=10000)