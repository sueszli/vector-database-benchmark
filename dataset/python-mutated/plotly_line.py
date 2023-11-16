from tensorwatch.watcher_base import WatcherBase
from tensorwatch.plotly.line_plot import LinePlot
from tensorwatch.image_utils import plt_loop
from tensorwatch.stream import Stream
from tensorwatch.lv_types import StreamItem

def main():
    if False:
        while True:
            i = 10
    watcher = WatcherBase()
    line_plot = LinePlot()
    stream = watcher.create_stream(expr='lambda vars:vars.x')
    line_plot.subscribe(stream)
    line_plot.show()
    for i in range(5):
        watcher.observe(x=(i, i * i))
main()