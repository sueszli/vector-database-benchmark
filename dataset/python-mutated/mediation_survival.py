import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.mediation import Mediation
np.random.seed(3424)
n = 1000
exp = np.random.normal(size=n)

def gen_mediator():
    if False:
        i = 10
        return i + 15
    mn = np.exp(exp)
    mtime0 = -mn * np.log(np.random.uniform(size=n))
    ctime = -2 * mn * np.log(np.random.uniform(size=n))
    mstatus = (ctime >= mtime0).astype(int)
    mtime = np.where(mtime0 <= ctime, mtime0, ctime)
    return (mtime0, mtime, mstatus)

def gen_outcome(otype, mtime0):
    if False:
        for i in range(10):
            print('nop')
    if otype == 'full':
        lp = 0.5 * mtime0
    elif otype == 'no':
        lp = exp
    else:
        lp = exp + mtime0
    mn = np.exp(-lp)
    ytime0 = -mn * np.log(np.random.uniform(size=n))
    ctime = -2 * mn * np.log(np.random.uniform(size=n))
    ystatus = (ctime >= ytime0).astype(int)
    ytime = np.where(ytime0 <= ctime, ytime0, ctime)
    return (ytime, ystatus)

def build_df(ytime, ystatus, mtime0, mtime, mstatus):
    if False:
        print('Hello World!')
    df = pd.DataFrame({'ytime': ytime, 'ystatus': ystatus, 'mtime': mtime, 'mstatus': mstatus, 'exp': exp})
    return df

def run(otype):
    if False:
        return 10
    (mtime0, mtime, mstatus) = gen_mediator()
    (ytime, ystatus) = gen_outcome(otype, mtime0)
    df = build_df(ytime, ystatus, mtime0, mtime, mstatus)
    outcome_model = sm.PHReg.from_formula('ytime ~ exp + mtime', status='ystatus', data=df)
    mediator_model = sm.PHReg.from_formula('mtime ~ exp', status='mstatus', data=df)
    med = Mediation(outcome_model, mediator_model, 'exp', 'mtime', outcome_predict_kwargs={'pred_only': True})
    med_result = med.fit(n_rep=20)
    print(med_result.summary())
run('full')
run('partial')
run('no')