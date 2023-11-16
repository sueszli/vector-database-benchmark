try:
    from bokeh.plotting import figure
    from bokeh.palettes import brewer
    from bokeh.models import Range1d
    from bokeh.embed import components
    from jinja2 import Template
except ImportError:
    pass

def x_label(epoch_axis):
    if False:
        return 10
    '\n    Get the x axis label depending on the boolean epoch_axis.\n\n    Arguments:\n        epoch_axis (bool): If true, use Epoch, if false use Minibatch\n\n    Returns:\n        str: "Epoch" or "Minibatch"\n    '
    return 'Epoch' if epoch_axis else 'Minibatch'

def cost_fig(cost_data, plot_height, plot_width, epoch_axis=True):
    if False:
        print('Hello World!')
    '\n    Generate a figure with lines for each element in cost_data.\n\n    Arguments:\n        cost_data (list): Cost data to plot\n        plot_height (int): Plot height\n        plot_width (int): Plot width\n        epoch_axis (bool, optional): If true, use Epoch, if false use Minibatch\n\n    Returns:\n        bokeh.plotting.figure: cost_data figure\n    '
    fig = figure(plot_height=plot_height, plot_width=plot_width, title='Cost', x_axis_label=x_label(epoch_axis), y_axis_label='Cross Entropy Error (%)')
    num_colors_required = len(cost_data)
    assert num_colors_required <= 11, 'Insufficient colors in predefined palette.'
    colors = list(brewer['Spectral'][max(3, len(cost_data))])
    if num_colors_required < 3:
        colors[0] = brewer['Spectral'][6][0]
        if num_colors_required == 2:
            colors[1] = brewer['Spectral'][6][-1]
    for (name, x, y) in cost_data:
        fig.line(x, y, legend=name, color=colors.pop(0), line_width=2)
    return fig

def hist_fig(hist_data, plot_height, plot_width, x_range=None, epoch_axis=True):
    if False:
        print('Hello World!')
    '\n    Generate a figure with an image plot for hist_data, bins on the Y axis and\n    time on the X axis.\n\n    Arguments:\n        hist_data (tuple): Hist data to plot\n        plot_height (int): Plot height\n        plot_width (int): Plot width\n        x_range (tuple, optional): (start, end) range for x\n        epoch_axis (boolm optional): If true, use Epoch, if false use Minibatch\n\n    Returns:\n        bokeh.plotting.figure: hist_data figure\n    '
    (name, hdata, dh, dw, bins, offset) = hist_data
    if x_range is None:
        x_range = (0, dw)
    fig = figure(plot_height=plot_height, plot_width=plot_width, title=name, x_axis_label=x_label(epoch_axis), x_range=x_range, y_range=(offset, offset + bins))
    fig.image(image=[hdata], x=[0], y=[offset], dw=[dw], dh=[dh], palette='Spectral11')
    return fig

def image_fig(data, h, w, x_range, y_range, plot_size):
    if False:
        i = 10
        return i + 15
    '\n    Helper function to generate a figure\n\n    Arguments:\n        data (int): data to plot\n        h (int): height\n        w (int): width\n        x_range (tuple, optional): (start, end) range for x\n        y_range (tuple, optional): (start, end) range for y\n        plot_size (int): plot size\n\n    Returns:\n        bokeh.plotting.figure: Generated figure\n    '
    fig = figure(x_range=x_range, y_range=y_range, plot_width=plot_size, plot_height=plot_size, toolbar_location=None)
    fig.image_rgba([data], x=[0], y=[0], dw=[w], dh=[h])
    fig.axis.visible = None
    fig.min_border = 0
    return fig

def deconv_figs(layer_name, layer_data, fm_max=8, plot_size=120):
    if False:
        while True:
            i = 10
    '\n    Helper function to generate deconv visualization figures\n\n    Arguments:\n        layer_name (str): Layer name\n        layer_data (list): Layer data to plot\n        fm_max (int): Max layers to process\n        plot_size (int, optional): Plot size\n\n    Returns:\n        tuple if vis_keys, img_keys, fig_dict\n    '
    vis_keys = dict()
    img_keys = dict()
    fig_dict = dict()
    for (fm_num, (fm_name, deconv_data, img_data)) in enumerate(layer_data):
        if fm_num >= fm_max:
            break
        (img_h, img_w) = img_data.shape
        x_range = Range1d(start=0, end=img_w)
        y_range = Range1d(start=0, end=img_h)
        img_fig = image_fig(img_data, img_h, img_w, x_range, y_range, plot_size)
        deconv_fig = image_fig(deconv_data, img_h, img_w, x_range, y_range, plot_size)
        title = '{}_fmap_{:04d}'.format(layer_name, fm_num)
        vis_keys[fm_num] = 'vis_' + title
        img_keys[fm_num] = 'img_' + title
        fig_dict[vis_keys[fm_num]] = deconv_fig
        fig_dict[img_keys[fm_num]] = img_fig
    return (vis_keys, img_keys, fig_dict)

def deconv_summary_page(filename, cost_data, deconv_data):
    if False:
        while True:
            i = 10
    '\n    Generate an HTML page with a Deconv visualization\n\n    Arguments:\n        filename: Output filename\n        cost_data (list): Cost data to plot\n        deconv_data (tuple): deconv data to plot\n    '
    fig_dict = dict()
    cost_key = 'cost_plot'
    fig_dict[cost_key] = cost_fig(cost_data, 300, 533, epoch_axis=True)
    vis_keys = dict()
    img_keys = dict()
    for (layer, layer_data) in deconv_data:
        (lyr_vis_keys, lyr_img_keys, lyr_fig_dict) = deconv_figs(layer, layer_data, fm_max=4)
        vis_keys[layer] = lyr_vis_keys
        img_keys[layer] = lyr_img_keys
        fig_dict.update(lyr_fig_dict)
    (script, div) = components(fig_dict)
    template = Template('\n<!DOCTYPE html>\n<html lang="en">\n    <head>\n        <meta charset="utf-8">\n        <title>{{page_title}}</title>\n        <style> div{float: left;} </style>\n        <link rel="stylesheet"\n              href="http://cdn.pydata.org/bokeh/release/bokeh-0.9.0.min.css"\n              type="text/css" />\n        <script type="text/javascript"\n                src="http://cdn.pydata.org/bokeh/release/bokeh-0.9.0.min.js"></script>\n        {{ script }}\n    </head>\n    <body>\n    <div id=cost_plot style="width:100%; padding:10px">\n      {{ div[cost_key]}}\n    </div>\n\n    {% for layer in sorted_layers %}\n        <div id=Outer{{layer}} style="padding:20px">\n        <div id={{layer}} style="background-color: #C6FFF1; padding:10px">\n        Layer {{layer}}<br>\n        {% for fm in vis_keys[layer].keys() %}\n            <div id={{fm}} style="padding:10px">\n            Feature Map {{fm}}<br>\n            {{ div[vis_keys[layer][fm]] }}\n            {{ div[img_keys[layer][fm]] }}\n            </div>\n        {% endfor %}\n        </div>\n        </div>\n\n        <br><br>\n    {% endfor %}\n    </body>\n</html>\n')
    with open(filename, 'w') as htmlfile:
        htmlfile.write(template.render(page_title='Deconv Visualization', script=script, div=div, cost_key=cost_key, vis_keys=vis_keys, img_keys=img_keys, sorted_layers=sorted(vis_keys)))