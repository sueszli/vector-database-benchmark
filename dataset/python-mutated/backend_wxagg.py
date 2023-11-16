import wx
from .backend_agg import FigureCanvasAgg
from .backend_wx import _BackendWx, _FigureCanvasWxBase
from .backend_wx import NavigationToolbar2Wx as NavigationToolbar2WxAgg

class FigureCanvasWxAgg(FigureCanvasAgg, _FigureCanvasWxBase):

    def draw(self, drawDC=None):
        if False:
            return 10
        '\n        Render the figure using agg.\n        '
        FigureCanvasAgg.draw(self)
        self.bitmap = _rgba_to_wx_bitmap(self.get_renderer().buffer_rgba())
        self._isDrawn = True
        self.gui_repaint(drawDC=drawDC)

    def blit(self, bbox=None):
        if False:
            for i in range(10):
                print('nop')
        bitmap = _rgba_to_wx_bitmap(self.get_renderer().buffer_rgba())
        if bbox is None:
            self.bitmap = bitmap
        else:
            srcDC = wx.MemoryDC(bitmap)
            destDC = wx.MemoryDC(self.bitmap)
            x = int(bbox.x0)
            y = int(self.bitmap.GetHeight() - bbox.y1)
            destDC.Blit(x, y, int(bbox.width), int(bbox.height), srcDC, x, y)
            destDC.SelectObject(wx.NullBitmap)
            srcDC.SelectObject(wx.NullBitmap)
        self.gui_repaint()

def _rgba_to_wx_bitmap(rgba):
    if False:
        print('Hello World!')
    'Convert an RGBA buffer to a wx.Bitmap.'
    (h, w, _) = rgba.shape
    return wx.Bitmap.FromBufferRGBA(w, h, rgba)

@_BackendWx.export
class _BackendWxAgg(_BackendWx):
    FigureCanvas = FigureCanvasWxAgg