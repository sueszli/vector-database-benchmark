import matplotlib
import numpy as np
matplotlib.use('module://matplotlib_pyodide.html5_canvas_backend')
import datetime
from matplotlib import pyplot as plt
from matplotlib.dates import YEARLY, DateFormatter, RRuleLocator, drange, rrulewrapper

def canvas_text_rotated():
    if False:
        return 10
    np.random.seed(42)
    rule = rrulewrapper(YEARLY, byeaster=1, interval=5)
    loc = RRuleLocator(rule)
    formatter = DateFormatter('%m/%d/%y')
    date1 = datetime.date(1952, 1, 1)
    date2 = datetime.date(2004, 4, 12)
    delta = datetime.timedelta(days=100)
    dates = drange(date1, date2, delta)
    s = np.random.rand(len(dates))
    (fig, ax) = plt.subplots()
    plt.plot_date(dates, s)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(formatter)
    labels = ax.get_xticklabels()
    plt.setp(labels, rotation=30, fontsize=10)
    plt.show()
    plt.close('all')
    plt.clf()