import win32api
import win32con
import win32ui

class DIBView:

    def __init__(self, doc, dib):
        if False:
            print('Hello World!')
        self.dib = dib
        self.view = win32ui.CreateView(doc)
        self.width = self.height = 0
        self.view.HookMessage(self.OnSize, win32con.WM_SIZE)

    def OnSize(self, params):
        if False:
            while True:
                i = 10
        lParam = params[3]
        self.width = win32api.LOWORD(lParam)
        self.height = win32api.HIWORD(lParam)

    def OnDraw(self, ob, dc):
        if False:
            return 10
        self.view.SetScrollSizes(win32con.MM_TEXT, self.dib.GetSize())
        dibSize = self.dib.GetSize()
        dibRect = (0, 0, dibSize[0], dibSize[1])
        self.dib.Paint(dc)

class DIBDemo:

    def __init__(self, filename, *bPBM):
        if False:
            while True:
                i = 10
        f = open(filename, 'rb')
        dib = win32ui.CreateDIBitmap()
        if len(bPBM) > 0:
            magic = f.readline()
            if magic != 'P6\n':
                print('The file is not a PBM format file')
                raise ValueError('Failed - The file is not a PBM format file')
            rowcollist = f.readline().split()
            cols = int(rowcollist[0])
            rows = int(rowcollist[1])
            f.readline()
            dib.LoadPBMData(f, (cols, rows))
        else:
            dib.LoadWindowsFormatFile(f)
        f.close()
        self.doc = win32ui.CreateDoc()
        self.dibView = DIBView(self.doc, dib)
        self.frame = win32ui.CreateMDIFrame()
        self.frame.LoadFrame()
        self.doc.SetTitle('DIB Demo')
        self.frame.ShowWindow()
        self.frame.ActivateFrame()

    def OnCreateClient(self, createparams, context):
        if False:
            return 10
        self.dibView.view.CreateWindow(self.frame)
        return 1
if __name__ == '__main__':
    import demoutils
    demoutils.NotAScript()