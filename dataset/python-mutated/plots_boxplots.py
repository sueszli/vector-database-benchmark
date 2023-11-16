import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
data = sm.datasets.anes96.load_pandas()
party_ID = np.arange(7)
labels = ['Strong Democrat', 'Weak Democrat', 'Independent-Democrat', 'Independent-Independent', 'Independent-Republican', 'Weak Republican', 'Strong Republican']
plt.rcParams['figure.subplot.bottom'] = 0.23
plt.rcParams['figure.figsize'] = (10.0, 8.0)
age = [data.exog['age'][data.endog == id] for id in party_ID]
fig = plt.figure()
ax = fig.add_subplot(111)
plot_opts = {'cutoff_val': 5, 'cutoff_type': 'abs', 'label_fontsize': 'small', 'label_rotation': 30}
sm.graphics.beanplot(age, ax=ax, labels=labels, plot_opts=plot_opts)
ax.set_xlabel('Party identification of respondent.')
ax.set_ylabel('Age')

def beanplot(data, plot_opts={}, jitter=False):
    if False:
        return 10
    'helper function to try out different plot options'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_opts_ = {'cutoff_val': 5, 'cutoff_type': 'abs', 'label_fontsize': 'small', 'label_rotation': 30}
    plot_opts_.update(plot_opts)
    sm.graphics.beanplot(data, ax=ax, labels=labels, jitter=jitter, plot_opts=plot_opts_)
    ax.set_xlabel('Party identification of respondent.')
    ax.set_ylabel('Age')
fig = beanplot(age, jitter=True)
fig = beanplot(age, plot_opts={'violin_width': 0.5, 'violin_fc': '#66c2a5'})
fig = beanplot(age, plot_opts={'violin_fc': '#66c2a5'})
fig = beanplot(age, plot_opts={'bean_size': 0.2, 'violin_width': 0.75, 'violin_fc': '#66c2a5'})
fig = beanplot(age, jitter=True, plot_opts={'violin_fc': '#66c2a5'})
fig = beanplot(age, jitter=True, plot_opts={'violin_width': 0.5, 'violin_fc': '#66c2a5'})
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
plt.rcParams['figure.subplot.bottom'] = 0.23
data = sm.datasets.anes96.load_pandas()
party_ID = np.arange(7)
labels = ['Strong Democrat', 'Weak Democrat', 'Independent-Democrat', 'Independent-Independent', 'Independent-Republican', 'Weak Republican', 'Strong Republican']
age = [data.exog['age'][data.endog == id] for id in party_ID]
fig = plt.figure()
ax = fig.add_subplot(111)
sm.graphics.violinplot(age, ax=ax, labels=labels, plot_opts={'cutoff_val': 5, 'cutoff_type': 'abs', 'label_fontsize': 'small', 'label_rotation': 30})
ax.set_xlabel('Party identification of respondent.')
ax.set_ylabel('Age')
ax.set_title("US national election '96 - Age & Party Identification")
fig2 = plt.figure()
ax = fig2.add_subplot(111)
sm.graphics.beanplot(age, ax=ax, labels=labels, plot_opts={'cutoff_val': 5, 'cutoff_type': 'abs', 'label_fontsize': 'small', 'label_rotation': 30})
ax.set_xlabel('Party identification of respondent.')
ax.set_ylabel('Age')
ax.set_title("US national election '96 - Age & Party Identification")
fig3 = plt.figure()
ax = fig3.add_subplot(111)
plot_opts = {'cutoff_val': 5, 'cutoff_type': 'abs', 'label_fontsize': 'small', 'label_rotation': 30, 'violin_fc': (0.8, 0.8, 0.8), 'jitter_marker': '.', 'jitter_marker_size': 3, 'bean_color': '#FF6F00', 'bean_mean_color': '#009D91'}
sm.graphics.beanplot(age, ax=ax, labels=labels, jitter=True, plot_opts=plot_opts)
ax.set_xlabel('Party identification of respondent.')
ax.set_ylabel('Age')
ax.set_title("US national election '96 - Age & Party Identification")
ix = data.exog['income'] < 16
age = data.exog['age'][ix]
endog = data.endog[ix]
age_lower_income = [age[endog == id] for id in party_ID]
ix = data.exog['income'] >= 20
age = data.exog['age'][ix]
endog = data.endog[ix]
age_higher_income = [age[endog == id] for id in party_ID]
fig = plt.figure()
ax = fig.add_subplot(111)
plot_opts['violin_fc'] = (0.5, 0.5, 0.5)
plot_opts['bean_show_mean'] = False
plot_opts['bean_show_median'] = False
plot_opts['bean_legend_text'] = 'Income < \\$30k'
plot_opts['cutoff_val'] = 10
sm.graphics.beanplot(age_lower_income, ax=ax, labels=labels, side='left', jitter=True, plot_opts=plot_opts)
plot_opts['violin_fc'] = (0.7, 0.7, 0.7)
plot_opts['bean_color'] = '#009D91'
plot_opts['bean_legend_text'] = 'Income > \\$50k'
sm.graphics.beanplot(age_higher_income, ax=ax, labels=labels, side='right', jitter=True, plot_opts=plot_opts)
ax.set_xlabel('Party identification of respondent.')
ax.set_ylabel('Age')
ax.set_title("US national election '96 - Age & Party Identification")