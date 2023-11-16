"""
==================
Embedding in wx #3
==================

Copyright (C) 2003-2004 Andrew Straw, Jeremy O'Donoghue and others

License: This work is licensed under the PSF. A copy should be included
with this source code, and is also available at
https://docs.python.org/3/license.html

This is yet another example of using matplotlib with wx.  Hopefully
this is pretty full-featured:

- both matplotlib toolbar and WX buttons manipulate plot
- full wxApp framework, including widget interaction
- XRC (XML wxWidgets resource) file to create GUI (made with XRCed)

This was derived from embedding_in_wx and dynamic_image_wxagg.

Thanks to matplotlib and wx teams for creating such great software!
"""
import wx
import wx.xrc as xrc
import numpy as np
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg as NavigationToolbar
import matplotlib.cbook as cbook
import matplotlib.cm as cm
from matplotlib.figure import Figure
ERR_TOL = 1e-05

class PlotPanel(wx.Panel):

    def __init__(self, parent):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(parent, -1)
        self.fig = Figure((5, 4), 75)
        self.canvas = FigureCanvas(self, -1, self.fig)
        self.toolbar = NavigationToolbar(self.canvas)
        self.toolbar.Realize()
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        sizer.Add(self.toolbar, 0, wx.GROW)
        self.SetSizer(sizer)
        self.Fit()

    def init_plot_data(self):
        if False:
            return 10
        ax = self.fig.add_subplot()
        x = np.arange(120.0) * 2 * np.pi / 60.0
        y = np.arange(100.0) * 2 * np.pi / 50.0
        (self.x, self.y) = np.meshgrid(x, y)
        z = np.sin(self.x) + np.cos(self.y)
        self.im = ax.imshow(z, cmap=cm.RdBu, origin='lower')
        zmax = np.max(z) - ERR_TOL
        (ymax_i, xmax_i) = np.nonzero(z >= zmax)
        if self.im.origin == 'upper':
            ymax_i = z.shape[0] - ymax_i
        self.lines = ax.plot(xmax_i, ymax_i, 'ko')
        self.toolbar.update()

    def GetToolBar(self):
        if False:
            for i in range(10):
                print('nop')
        return self.toolbar

    def OnWhiz(self, event):
        if False:
            return 10
        self.x += np.pi / 15
        self.y += np.pi / 20
        z = np.sin(self.x) + np.cos(self.y)
        self.im.set_array(z)
        zmax = np.max(z) - ERR_TOL
        (ymax_i, xmax_i) = np.nonzero(z >= zmax)
        if self.im.origin == 'upper':
            ymax_i = z.shape[0] - ymax_i
        self.lines[0].set_data(xmax_i, ymax_i)
        self.canvas.draw()

class MyApp(wx.App):

    def OnInit(self):
        if False:
            print('Hello World!')
        xrcfile = cbook.get_sample_data('embedding_in_wx3.xrc', asfileobj=False)
        print('loading', xrcfile)
        self.res = xrc.XmlResource(xrcfile)
        self.frame = self.res.LoadFrame(None, 'MainFrame')
        self.panel = xrc.XRCCTRL(self.frame, 'MainPanel')
        plot_container = xrc.XRCCTRL(self.frame, 'plot_container_panel')
        sizer = wx.BoxSizer(wx.VERTICAL)
        self.plotpanel = PlotPanel(plot_container)
        self.plotpanel.init_plot_data()
        sizer.Add(self.plotpanel, 1, wx.EXPAND)
        plot_container.SetSizer(sizer)
        whiz_button = xrc.XRCCTRL(self.frame, 'whiz_button')
        whiz_button.Bind(wx.EVT_BUTTON, self.plotpanel.OnWhiz)
        bang_button = xrc.XRCCTRL(self.frame, 'bang_button')
        bang_button.Bind(wx.EVT_BUTTON, self.OnBang)
        self.frame.Show()
        self.SetTopWindow(self.frame)
        return True

    def OnBang(self, event):
        if False:
            for i in range(10):
                print('nop')
        bang_count = xrc.XRCCTRL(self.frame, 'bang_count')
        bangs = bang_count.GetValue()
        bangs = int(bangs) + 1
        bang_count.SetValue(str(bangs))
if __name__ == '__main__':
    app = MyApp()
    app.MainLoop()