import statsmodels.api as sm
import statsmodels.formula.api as smf
star98 = sm.datasets.star98.load_pandas().data
formula = 'SUCCESS ~ LOWINC + PERASIAN + PERBLACK + PERHISP + PCTCHRT +            PCTYRRND + PERMINTE*AVYRSEXP*AVSALK + PERSPENK*PTRATIO*PCTAF'
dta = star98[['NABOVE', 'NBELOW', 'LOWINC', 'PERASIAN', 'PERBLACK', 'PERHISP', 'PCTCHRT', 'PCTYRRND', 'PERMINTE', 'AVYRSEXP', 'AVSALK', 'PERSPENK', 'PTRATIO', 'PCTAF']].copy()
endog = dta['NABOVE'] / (dta['NABOVE'] + dta.pop('NBELOW'))
del dta['NABOVE']
dta['SUCCESS'] = endog
mod1 = smf.glm(formula=formula, data=dta, family=sm.families.Binomial()).fit()
print(mod1.summary())

def double_it(x):
    if False:
        i = 10
        return i + 15
    return 2 * x
formula = 'SUCCESS ~ double_it(LOWINC) + PERASIAN + PERBLACK + PERHISP + PCTCHRT +            PCTYRRND + PERMINTE*AVYRSEXP*AVSALK + PERSPENK*PTRATIO*PCTAF'
mod2 = smf.glm(formula=formula, data=dta, family=sm.families.Binomial()).fit()
print(mod2.summary())
print(mod1.params[1])
print(mod2.params[1] * 2)