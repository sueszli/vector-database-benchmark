import tensorwatch as tw
from tensorwatch import utils
utils.set_debug_verbosity(4)

def show_mpl():
    if False:
        while True:
            i = 10
    cli = tw.WatcherClient('c:\\temp\\sum.log')
    s1 = cli.open_stream('sum')
    p = tw.LinePlot(title='Demo')
    p.subscribe(s1, xtitle='Index', ytitle='sqrt(ev_i)')
    s1.load()
    p.show()
    tw.plt_loop()

def show_text():
    if False:
        for i in range(10):
            print('nop')
    cli = tw.WatcherClient('c:\\temp\\sum.log')
    s1 = cli.open_stream('sum_2')
    text = tw.Visualizer(s1)
    text.show()
    input('Waiting')
show_mpl()