from glob import glob
import os
import shutil
import plotly
plotly.io.renderers.default = 'sphinx_gallery_png'

def plotly_sg_scraper(block, block_vars, gallery_conf, **kwargs):
    if False:
        return 10
    "Scrape Plotly figures for galleries of examples using\n    sphinx-gallery.\n\n    Examples should use ``plotly.io.show()`` to display the figure with\n    the custom sphinx_gallery renderer.\n\n    Since the sphinx_gallery renderer generates both html and static png\n    files, we simply crawl these files and give them the appropriate path.\n\n    Parameters\n    ----------\n    block : tuple\n        A tuple containing the (label, content, line_number) of the block.\n    block_vars : dict\n        Dict of block variables.\n    gallery_conf : dict\n        Contains the configuration of Sphinx-Gallery\n    **kwargs : dict\n        Additional keyword arguments to pass to\n        :meth:`~matplotlib.figure.Figure.savefig`, e.g. ``format='svg'``.\n        The ``format`` kwarg in particular is used to set the file extension\n        of the output file (currently only 'png' and 'svg' are supported).\n\n    Returns\n    -------\n    rst : str\n        The ReSTructuredText that will be rendered to HTML containing\n        the images.\n\n    Notes\n    -----\n    Add this function to the image scrapers\n    "
    examples_dir = os.path.dirname(block_vars['src_file'])
    pngs = sorted(glob(os.path.join(examples_dir, '*.png')))
    htmls = sorted(glob(os.path.join(examples_dir, '*.html')))
    image_path_iterator = block_vars['image_path_iterator']
    image_names = list()
    seen = set()
    for (html, png) in zip(htmls, pngs):
        if png not in seen:
            seen |= set(png)
            this_image_path_png = next(image_path_iterator)
            this_image_path_html = os.path.splitext(this_image_path_png)[0] + '.html'
            image_names.append(this_image_path_html)
            shutil.move(png, this_image_path_png)
            shutil.move(html, this_image_path_html)
    return figure_rst(image_names, gallery_conf['src_dir'])

def figure_rst(figure_list, sources_dir):
    if False:
        return 10
    "Generate RST for a list of PNG filenames.\n\n    Depending on whether we have one or more figures, we use a\n    single rst call to 'image' or a horizontal list.\n\n    Parameters\n    ----------\n    figure_list : list\n        List of strings of the figures' absolute paths.\n    sources_dir : str\n        absolute path of Sphinx documentation sources\n\n    Returns\n    -------\n    images_rst : str\n        rst code to embed the images in the document\n    "
    figure_paths = [os.path.relpath(figure_path, sources_dir).replace(os.sep, '/').lstrip('/') for figure_path in figure_list]
    images_rst = ''
    if not figure_paths:
        return images_rst
    figure_name = figure_paths[0]
    ext = os.path.splitext(figure_name)[1]
    figure_path = os.path.join('images', os.path.basename(figure_name))
    images_rst = SINGLE_HTML % figure_path
    return images_rst
SINGLE_HTML = '\n.. raw:: html\n    :file: %s\n'