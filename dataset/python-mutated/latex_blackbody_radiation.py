""" A plot of spectral radiance curves for an ideal radiating blackbody at
various temperatures. This example demonstrates the use of mathtext on axes
and in ``Div`` objects.

.. bokeh-example-metadata::
    :apis: bokeh.plotting.figure.line, bokeh.models.Div
    :refs: :ref:`ug_styling_mathtext`
    :keywords: mathtext, latex

"""
import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import Div
from bokeh.palettes import Spectral
from bokeh.plotting import figure, show
p = figure(width=700, height=500, toolbar_location=None, title='Black body spectral radiance as a function of frequency')

def spectral_radiance(nu, T):
    if False:
        i = 10
        return i + 15
    h = 6.626e-34
    k = 1.3806e-23
    c = 299790000.0
    return 2 * h * nu ** 3 / c ** 2 / (np.exp(h * nu / (k * T)) - 1.0)
Ts = np.arange(2000, 6001, 500)
palette = Spectral[len(Ts)]
nu = np.linspace(0.1, 1000000000000000.0, 500)
for (i, T) in enumerate(Ts):
    B_nu = spectral_radiance(nu, T)
    p.line(nu / 1000000000000000.0, B_nu / 1e-09, line_width=2, legend_label=f'T = {T} K', line_color=palette[i])
p.legend.items = list(reversed(p.legend.items))
Ts = np.linspace(1900, 6101, 50)
peak_freqs = Ts * 58790000000.0
peak_radiance = spectral_radiance(peak_freqs, Ts)
p.line(peak_freqs / 1000000000000000.0, peak_radiance / 1e-09, line_color='silver', line_dash='dashed', line_width=2, legend_label='Peak radiance')
curdoc().theme = 'dark_minimal'
p.y_range.start = 0
p.xaxis.axis_label = '$$\\nu \\:(10^{15}\\ \\text{Hz})$$'
p.yaxis.axis_label = '$$B_\\nu(\\nu, T) \\quad\\left(10^{-9}\\ \\text{W} / (\\text{m}^2 \\cdot \\text{sr} \\cdot \\text{Hz})\\right)$$'
div = Div(text='\nA plot of the spectral radiance, defined as a function of the frequency $$\\nu$$, is given by the formula\n<p \\>\n$$\n\\qquad B_\\nu(\\nu, T) = \\frac{2h\\nu^3}{c^2} \\frac{1}{\\exp(h\\nu/kT)-1}\\ .\n$$\n')
show(column(p, div))