"""
======================
Display mathtext in WX
======================

Demonstrates how to convert (math)text to a wx.Bitmap for display in various
controls on wxPython.
"""
from io import BytesIO
import wx
import numpy as np
from matplotlib.backends.backend_wx import NavigationToolbar2Wx
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure
IS_WIN = 'wxMSW' in wx.PlatformInfo

def mathtext_to_wxbitmap(s):
    if False:
        print('Hello World!')
    fig = Figure(facecolor='none')
    text_color = np.array(wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOWTEXT)) / 255
    fig.text(0, 0, s, fontsize=10, color=text_color)
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', pad_inches=0)
    s = buf.getvalue()
    return wx.Bitmap.NewFromPNGData(s, len(s))
functions = [('$\\sin(2 \\pi x)$', lambda x: np.sin(2 * np.pi * x)), ('$\\frac{4}{3}\\pi x^3$', lambda x: 4 / 3 * np.pi * x ** 3), ('$\\cos(2 \\pi x)$', lambda x: np.cos(2 * np.pi * x)), ('$\\log(x)$', lambda x: np.log(x))]

class CanvasFrame(wx.Frame):

    def __init__(self, parent, title):
        if False:
            i = 10
            return i + 15
        super().__init__(parent, -1, title, size=(550, 350))
        self.figure = Figure()
        self.axes = self.figure.add_subplot()
        self.canvas = FigureCanvas(self, -1, self.figure)
        self.change_plot(0)
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.add_buttonbar()
        self.sizer.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        self.add_toolbar()
        menuBar = wx.MenuBar()
        menu = wx.Menu()
        m_exit = menu.Append(wx.ID_EXIT, 'E&xit\tAlt-X', 'Exit this simple sample')
        menuBar.Append(menu, '&File')
        self.Bind(wx.EVT_MENU, self.OnClose, m_exit)
        if IS_WIN:
            menu = wx.Menu()
            for (i, (mt, func)) in enumerate(functions):
                bm = mathtext_to_wxbitmap(mt)
                item = wx.MenuItem(menu, 1000 + i, ' ')
                item.SetBitmap(bm)
                menu.Append(item)
                self.Bind(wx.EVT_MENU, self.OnChangePlot, item)
            menuBar.Append(menu, '&Functions')
        self.SetMenuBar(menuBar)
        self.SetSizer(self.sizer)
        self.Fit()

    def add_buttonbar(self):
        if False:
            i = 10
            return i + 15
        self.button_bar = wx.Panel(self)
        self.button_bar_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.sizer.Add(self.button_bar, 0, wx.LEFT | wx.TOP | wx.GROW)
        for (i, (mt, func)) in enumerate(functions):
            bm = mathtext_to_wxbitmap(mt)
            button = wx.BitmapButton(self.button_bar, 1000 + i, bm)
            self.button_bar_sizer.Add(button, 1, wx.GROW)
            self.Bind(wx.EVT_BUTTON, self.OnChangePlot, button)
        self.button_bar.SetSizer(self.button_bar_sizer)

    def add_toolbar(self):
        if False:
            for i in range(10):
                print('nop')
        'Copied verbatim from embedding_wx2.py'
        self.toolbar = NavigationToolbar2Wx(self.canvas)
        self.toolbar.Realize()
        self.sizer.Add(self.toolbar, 0, wx.LEFT | wx.EXPAND)
        self.toolbar.update()

    def OnChangePlot(self, event):
        if False:
            i = 10
            return i + 15
        self.change_plot(event.GetId() - 1000)

    def change_plot(self, plot_number):
        if False:
            return 10
        t = np.arange(1.0, 3.0, 0.01)
        s = functions[plot_number][1](t)
        self.axes.clear()
        self.axes.plot(t, s)
        self.canvas.draw()

    def OnClose(self, event):
        if False:
            while True:
                i = 10
        self.Destroy()

class MyApp(wx.App):

    def OnInit(self):
        if False:
            i = 10
            return i + 15
        frame = CanvasFrame(None, 'wxPython mathtext demo app')
        self.SetTopWindow(frame)
        frame.Show(True)
        return True
if __name__ == '__main__':
    app = MyApp()
    app.MainLoop()