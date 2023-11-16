import tensorwatch as tw
import time
import math
from tensorwatch import utils
utils.set_debug_verbosity(4)

def mpl_line_plot():
    if False:
        for i in range(10):
            print('nop')
    cli = tw.WatcherClient()
    p = tw.LinePlot(title='Demo')
    s1 = cli.create_stream(event_name='ev_i', expr='map(lambda v:math.sqrt(v.val)*2, l)')
    p.subscribe(s1, xtitle='Index', ytitle='sqrt(ev_i)')
    p.show()
    tw.plt_loop()

def mpl_history_plot():
    if False:
        print('Hello World!')
    cli = tw.WatcherClient()
    p2 = tw.LinePlot(title='History Demo')
    p2s1 = cli.create_stream(event_name='ev_j', expr='map(lambda v:(v.val, math.sqrt(v.val)*2), l)')
    p2.subscribe(p2s1, xtitle='Index', ytitle='sqrt(ev_j)', clear_after_end=True, history_len=15)
    p2.show()
    tw.plt_loop()

def show_stream():
    if False:
        for i in range(10):
            print('nop')
    cli = tw.WatcherClient()
    print('Subscribing to event ev_i...')
    s1 = cli.create_stream(event_name='ev_i', expr='map(lambda v:math.sqrt(v.val), l)')
    r1 = tw.TextVis(title='L1')
    r1.subscribe(s1)
    r1.show()
    print('Subscribing to event ev_j...')
    s2 = cli.create_stream(event_name='ev_j', expr='map(lambda v:v.val*v.val, l)')
    r2 = tw.TextVis(title='L2')
    r2.subscribe(s2)
    r2.show()
    print('Waiting for key...')
    utils.wait_key()

def plotly_line_graph():
    if False:
        while True:
            i = 10
    cli = tw.WatcherClient()
    s1 = cli.create_stream(event_name='ev_i', expr='map(lambda v:(v.x, math.sqrt(v.val)), l)')
    p = tw.plotly.line_plot.LinePlot()
    p.subscribe(s1)
    p.show()
    utils.wait_key()

def plotly_history_graph():
    if False:
        for i in range(10):
            print('nop')
    cli = tw.WatcherClient()
    p = tw.plotly.line_plot.LinePlot(title='Demo')
    s2 = cli.create_stream(event_name='ev_j', expr='map(lambda v:(v.x, v.val), l)')
    p.subscribe(s2, ytitle='ev_j', history_len=15)
    p.show()
    utils.wait_key()
mpl_line_plot()