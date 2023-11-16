import os
from typing import Optional
import matplotlib.pyplot as plt
from pycaret.internal.logging import get_logger

def show_yellowbrick_in_streamlit(visualizer, outpath=None, clear_figure=False, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Makes the magic happen and a visualizer appear! You can pass in a path to\n    save the figure to disk with various backends, or you can call it with no\n    arguments to show the figure either in a notebook or in a GUI window that\n    pops up on screen.\n\n    Parameters\n    ----------\n    outpath: string, default: None\n        path or None. Save figure to disk or if None show in window\n\n    clear_figure: boolean, default: False\n        When True, this flag clears the figure after saving to file or\n        showing on screen. This is useful when making consecutive plots.\n\n    kwargs: dict\n        generic keyword arguments.\n\n    Notes\n    -----\n    Developers of visualizers don't usually override show, as it is\n    primarily called by the user to render the visualization.\n    "
    import streamlit as st
    visualizer.finalize()
    if outpath is not None:
        plt.savefig(outpath, **kwargs)
    else:
        st.write(visualizer.fig)
    if clear_figure:
        visualizer.fig.clear()
    return visualizer.ax

def show_yellowbrick_plot(visualizer, X_train, y_train, X_test, y_test, name: str, handle_train: str='fit', handle_test: str='score', scale: float=1, save: bool=False, fit_kwargs: Optional[dict]=None, display_format: Optional[str]=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Generic method to handle yellowbrick plots.\n    '
    logger = get_logger()
    visualizer.fig.set_dpi(visualizer.fig.dpi * scale)
    if not fit_kwargs:
        fit_kwargs = {}
    fit_kwargs_and_kwargs = {**fit_kwargs, **kwargs}
    if handle_train == 'draw':
        logger.info('Drawing Model')
        visualizer.draw(X_train, y_train, **kwargs)
    elif handle_train == 'fit':
        logger.info('Fitting Model')
        visualizer.fit(X_train, y_train, **fit_kwargs_and_kwargs)
    elif handle_train == 'fit_transform':
        logger.info('Fitting & Transforming Model')
        visualizer.fit_transform(X_train, y_train, **fit_kwargs_and_kwargs)
    elif handle_train == 'score':
        logger.info('Scoring train set')
        visualizer.score(X_train, y_train, **kwargs)
    if handle_test == 'draw':
        visualizer.draw(X_test, y_test)
    elif handle_test == 'fit':
        visualizer.fit(X_test, y_test, **fit_kwargs)
    elif handle_test == 'fit_transform':
        visualizer.fit_transform(X_test, y_test, **fit_kwargs)
    elif handle_test == 'score':
        logger.info('Scoring test/hold-out set')
        visualizer.score(X_test, y_test)
    plot_filename = f'{name}.png'
    if save:
        if not isinstance(save, bool):
            plot_filename = os.path.join(save, plot_filename)
        logger.info(f"Saving '{plot_filename}'")
        visualizer.show(outpath=plot_filename, clear_figure=True, bbox_inches='tight')
    elif display_format == 'streamlit':
        show_yellowbrick_in_streamlit(visualizer, clear_figure=True)
    else:
        visualizer.show(clear_figure=True)
    logger.info('Visual Rendered Successfully')
    return plot_filename