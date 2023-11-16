"""
=====================================
Custom tick formatter for time series
=====================================

.. redirect-from:: /gallery/text_labels_and_annotations/date_index_formatter
.. redirect-from:: /gallery/ticks/date_index_formatter2

When plotting daily data, e.g., financial time series, one often wants
to leave out days on which there is no data, for instance weekends, so that
the data are plotted at regular intervals without extra spaces for the days
with no data.
The example shows how to use an 'index formatter' to achieve the desired plot.
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cbook as cbook
from matplotlib.dates import DateFormatter, DayLocator
import matplotlib.lines as ml
from matplotlib.ticker import Formatter
r = cbook.get_sample_data('goog.npz')['price_data']
r = r[:9]
(fig, (ax1, ax2)) = plt.subplots(nrows=2, figsize=(6, 6), layout='constrained')
fig.get_layout_engine().set(hspace=0.15)
ax1.plot(r['date'], r['adj_close'], 'o-')
gaps = np.flatnonzero(np.diff(r['date']) > np.timedelta64(1, 'D'))
for gap in r[['date', 'adj_close']][np.stack((gaps, gaps + 1)).T]:
    ax1.plot(gap['date'], gap['adj_close'], 'w--', lw=2)
ax1.legend(handles=[ml.Line2D([], [], ls='--', label='Gaps in daily data')])
ax1.set_title('Plot y at x Coordinates')
ax1.xaxis.set_major_locator(DayLocator())
ax1.xaxis.set_major_formatter(DateFormatter('%a'))

def format_date(x, _):
    if False:
        print('Hello World!')
    try:
        return r['date'][round(x)].item().strftime('%a')
    except IndexError:
        pass
ax2.plot(r['adj_close'], 'o-')
ax2.set_title('Plot y at Index Coordinates Using Custom Formatter')
ax2.xaxis.set_major_formatter(format_date)

class MyFormatter(Formatter):

    def __init__(self, dates, fmt='%a'):
        if False:
            while True:
                i = 10
        self.dates = dates
        self.fmt = fmt

    def __call__(self, x, pos=0):
        if False:
            return 10
        'Return the label for time x at position pos.'
        try:
            return self.dates[round(x)].item().strftime(self.fmt)
        except IndexError:
            pass
ax2.xaxis.set_major_formatter(MyFormatter(r['date'], '%a'))
plt.show()