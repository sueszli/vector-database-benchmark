def blocking_input_loop(figure, event_names, timeout, handler):
    if False:
        print('Hello World!')
    "\n    Run *figure*'s event loop while listening to interactive events.\n\n    The events listed in *event_names* are passed to *handler*.\n\n    This function is used to implement `.Figure.waitforbuttonpress`,\n    `.Figure.ginput`, and `.Axes.clabel`.\n\n    Parameters\n    ----------\n    figure : `~matplotlib.figure.Figure`\n    event_names : list of str\n        The names of the events passed to *handler*.\n    timeout : float\n        If positive, the event loop is stopped after *timeout* seconds.\n    handler : Callable[[Event], Any]\n        Function called for each event; it can force an early exit of the event\n        loop by calling ``canvas.stop_event_loop()``.\n    "
    if figure.canvas.manager:
        figure.show()
    cids = [figure.canvas.mpl_connect(name, handler) for name in event_names]
    try:
        figure.canvas.start_event_loop(timeout)
    finally:
        for cid in cids:
            figure.canvas.mpl_disconnect(cid)