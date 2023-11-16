import tensorwatch as tw

def writer():
    if False:
        return 10
    watcher = tw.Watcher(filename='c:\\temp\\test.log', port=None)
    with watcher.create_stream('metric1') as stream1:
        for i in range(3):
            stream1.write((i, i * i))
    with watcher.create_stream('metric2') as stream2:
        for i in range(3):
            stream2.write((i, i * i * i))

def reader1():
    if False:
        while True:
            i = 10
    print('---------------------------reader1---------------------------')
    watcher = tw.Watcher(filename='c:\\temp\\test.log', port=None)
    stream1 = watcher.open_stream('metric1')
    stream1.console_debug = True
    stream1.load()
    stream2 = watcher.open_stream('metric2')
    stream2.console_debug = True
    stream2.load()

def reader2():
    if False:
        for i in range(10):
            print('nop')
    print('---------------------------reader2---------------------------')
    watcher = tw.Watcher(filename='c:\\temp\\test.log', port=None)
    stream1 = watcher.open_stream('metric1')
    for item in stream1.read_all():
        print(item)
    stream2 = watcher.open_stream('metric2')
    for item in stream2.read_all():
        print(item)

def reader3():
    if False:
        return 10
    print('---------------------------reader3---------------------------')
    watcher = tw.Watcher(filename='c:\\temp\\test.log', port=None)
    stream1 = watcher.open_stream('metric1')
    stream2 = watcher.open_stream('metric2')
    vis1 = tw.Visualizer(stream1, vis_type='line')
    vis2 = tw.Visualizer(stream2, vis_type='line', host=vis1)
    vis1.show()
    tw.plt_loop()
writer()
reader1()
reader2()
reader3()