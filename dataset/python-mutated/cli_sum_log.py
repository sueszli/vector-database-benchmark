import tensorwatch as tw
from tensorwatch import utils
utils.set_debug_verbosity(4)

def show_mpl():
    if False:
        print('Hello World!')
    cli = tw.WatcherClient()
    st_isum = cli.open_stream('isums')
    st_rsum = cli.open_stream('rsums')
    line_plot = tw.Visualizer(st_isum, vis_type='line', xtitle='i', ytitle='isum')
    line_plot.show()
    line_plot2 = tw.Visualizer(st_rsum, vis_type='line', host=line_plot, ytitle='rsum')
    tw.plt_loop()

def show_text():
    if False:
        while True:
            i = 10
    cli = tw.WatcherClient()
    text_vis = tw.Visualizer(st_isum, vis_type='text')
    text_vis.show()
    input('Waiting')
show_mpl()