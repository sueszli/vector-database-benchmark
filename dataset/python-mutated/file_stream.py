from tensorwatch.watcher_base import WatcherBase
from tensorwatch.stream import Stream
from tensorwatch.file_stream import FileStream
from tensorwatch import LinePlot
from tensorwatch.image_utils import plt_loop
import tensorwatch as tw

def file_write():
    if False:
        for i in range(10):
            print('nop')
    watcher = WatcherBase()
    stream = watcher.create_stream(expr='lambda vars:(vars.x, vars.x**2)', devices=['c:\\temp\\obs.txt'])
    for i in range(5):
        watcher.observe(x=i)

def file_read():
    if False:
        print('Hello World!')
    watcher = WatcherBase()
    stream = watcher.open_stream(devices=['c:\\temp\\obs.txt'])
    vis = tw.Visualizer(stream, vis_type='mpl-line')
    vis.show()
    plt_loop()

def main():
    if False:
        while True:
            i = 10
    file_write()
    file_read()
main()