import os
from unittest import TestCase
import pandas as pd
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
import fbprophet.plot as plot
DATA = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data.csv'), parse_dates=['ds'])

class TestFbprophet(TestCase):

    def test_shim(self):
        if False:
            return 10
        m = Prophet()
        m.fit(DATA)
        future = m.make_future_dataframe(10, include_history=False)
        fcst = m.predict(future)
        df_cv = cross_validation(model=m, horizon='4 days', period='10 days', initial='115 days')
        fig = plot.plot_forecast_component(m=m, fcst=fcst, name='weekly')