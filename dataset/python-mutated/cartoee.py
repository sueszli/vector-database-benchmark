"""The cartoee module contains functions for creating publication-quality maps with cartopy and Earth Engine data."""
import logging
import os
import subprocess
import sys
import warnings
from collections.abc import Iterable
from io import BytesIO
import ee
import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import requests
from matplotlib import cm, colors
from matplotlib import font_manager as mfonts
from .basemaps import custom_tiles
try:
    import cartopy.crs as ccrs
    import cartopy.io.img_tiles as cimgt
    from cartopy.mpl.geoaxes import GeoAxes, GeoAxesSubplot
    from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
    from PIL import Image
except ImportError:
    print('cartopy is not installed. Please see https://scitools.org.uk/cartopy/docs/latest/installing.html#installing for instructions on how to install cartopy.\n')
    print('The easiest way to install cartopy is using conda: conda install -c conda-forge cartopy')

def check_dependencies():
    if False:
        while True:
            i = 10
    'Helper function to check dependencies used for cartoee\n    Dependencies not included in main geemap are: cartopy, PIL, and scipys\n\n    raises:\n        Exception: when conda is not found in path\n        Exception: when auto install fails to install/import packages\n    '
    import importlib
    is_conda = os.path.exists(os.path.join(sys.prefix, 'conda-meta'))
    if not is_conda:
        raise Exception('Auto installation requires `conda`. Please install conda using the following instructions before use: https://docs.conda.io/projects/conda/en/latest/user-guide/install/')
    dependencies = ['cartopy', 'pillow', 'scipy']
    for dependency in dependencies:
        try:
            importlib.import_module(dependency)
        except ImportError:
            logging.info(f'The {dependency} package is not installed. Trying install...')
            logging.info(f'Installing {dependency} ...')
            cmd = f'conda install -c conda-forge {dependency} -y'
            proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            (out, _) = proc.communicate()
            logging.info(out.decode())
    failed = []
    for dependency in dependencies:
        try:
            importlib.import_module(dependency)
        except ImportError:
            failed.append(dependency)
    if len(failed) > 0:
        failed_str = ','.join(failed)
        raise Exception(f"Auto installation failed...the following dependencies were not installed '{failed_str}'")
    else:
        logging.info('All dependencies are successfully imported/installed!')
    return

def get_map(ee_object, proj=None, basemap=None, zoom_level=2, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Wrapper function to create a new cartopy plot with project and adds Earth\n    Engine image results\n    Args:\n        ee_object (ee.Image | ee.FeatureCollection): Earth Engine image result to plot\n        proj (cartopy.crs, optional): Cartopy projection that determines the projection of the resulting plot. By default uses an equirectangular projection, PlateCarree\n        basemap (str, optional): Basemap to use. It can be one of ["ROADMAP", "SATELLITE", "TERRAIN", "HYBRID"] or cartopy.io.img_tiles, such as cimgt.StamenTerrain(). Defaults to None. See https://scitools.org.uk/cartopy/docs/v0.19/cartopy/io/img_tiles.html\n        zoom_level (int, optional): Zoom level of the basemap. Defaults to 2.\n        **kwargs: remaining keyword arguments are passed to addLayer()\n    Returns:\n        ax (cartopy.mpl.geoaxes.GeoAxesSubplot): cartopy GeoAxesSubplot object with Earth Engine results displayed\n    '
    if isinstance(ee_object, ee.geometry.Geometry) or isinstance(ee_object, ee.feature.Feature) or isinstance(ee_object, ee.featurecollection.FeatureCollection):
        features = ee.FeatureCollection(ee_object)
        if 'style' in kwargs and kwargs['style'] is not None:
            style = kwargs['style']
        else:
            style = {}
        props = features.first().propertyNames().getInfo()
        if 'style' in props:
            ee_object = features.style(**{'styleProperty': 'style'})
        else:
            ee_object = features.style(**style)
    elif isinstance(ee_object, ee.imagecollection.ImageCollection):
        ee_object = ee_object.mosaic()
    if proj is None:
        proj = ccrs.PlateCarree()
    if 'style' in kwargs:
        del kwargs['style']
    ax = mpl.pyplot.axes(projection=proj)
    if basemap is not None:
        if isinstance(basemap, str):
            if basemap.upper() in ['ROADMAP', 'SATELLITE', 'TERRAIN', 'HYBRID']:
                basemap = cimgt.GoogleTiles(url=custom_tiles['xyz'][basemap.upper()]['url'])
        try:
            ax.add_image(basemap, zoom_level)
        except Exception as e:
            print('Failed to add basemap: ', e)
    add_layer(ax, ee_object, **kwargs)
    return ax

def add_layer(ax, ee_object, dims=1000, region=None, cmap=None, vis_params=None, **kwargs):
    if False:
        while True:
            i = 10
    "Add an Earth Engine image to a cartopy plot.\n\n    args:\n        ee_object (ee.Image | ee.FeatureCollection): Earth Engine image result to plot.\n        ax (cartopy.mpl.geoaxes.GeoAxesSubplot | cartopy.mpl.geoaxes.GeoAxes): required cartopy GeoAxesSubplot object to add image overlay to\n        dims (list | tuple | int, optional): dimensions to request earth engine result as [WIDTH,HEIGHT]. If only one number is passed, it is used as the maximum, and the other dimension is computed by proportional scaling. Default None and infers dimensions\n        region (list | tuple, optional): geospatial region of the image to render in format [E,S,W,N]. By default, the whole image\n        cmap (str, optional): string specifying matplotlib colormap to colorize image. If cmap is specified visParams cannot contain 'palette' key\n        vis_params (dict, optional): visualization parameters as a dictionary. See https://developers.google.com/earth-engine/image_visualization for options\n\n    returns:\n        ax (cartopy.mpl.geoaxes.GeoAxesSubplot): cartopy GeoAxesSubplot object with Earth Engine results displayed\n\n    raises:\n        ValueError: If `dims` is not of type list, tuple, or int\n        ValueError: If `imgObj` is not of type ee.image.Image\n        ValueError: If `ax` if not of type cartopy.mpl.geoaxes.GeoAxesSubplot '\n    "
    if isinstance(ee_object, ee.geometry.Geometry) or isinstance(ee_object, ee.feature.Feature) or isinstance(ee_object, ee.featurecollection.FeatureCollection):
        features = ee.FeatureCollection(ee_object)
        if 'style' in kwargs and kwargs['style'] is not None:
            style = kwargs['style']
        else:
            style = {}
        props = features.first().propertyNames().getInfo()
        if 'style' in props:
            ee_object = features.style(**{'styleProperty': 'style'})
        else:
            ee_object = features.style(**style)
    elif isinstance(ee_object, ee.imagecollection.ImageCollection):
        ee_object = ee_object.mosaic()
    if type(ee_object) is not ee.image.Image:
        raise ValueError('provided `ee_object` is not of type ee.Image')
    if region is not None:
        map_region = ee.Geometry.Rectangle(region).getInfo()['coordinates']
        view_extent = (region[2], region[0], region[1], region[3])
    else:
        map_region = ee_object.geometry(100).bounds(1).getInfo()['coordinates']
        (x, y) = list(zip(*map_region[0]))
        view_extent = [min(x), max(x), min(y), max(y)]
        if ee_object.bandNames().getInfo() == ['vis-red', 'vis-green', 'vis-blue']:
            warnings.warn(f'The region parameter is not specified. Using the default region {map_region}. Please specify a region if you get a blank image.')
    if type(dims) not in [list, tuple, int]:
        raise ValueError('provided dims not of type list, tuple, or int')
    if type(ax) not in [GeoAxes, GeoAxesSubplot]:
        raise ValueError('provided axes not of type cartopy.mpl.geoaxes.GeoAxes or cartopy.mpl.geoaxes.GeoAxesSubplot')
    args = {'format': 'png', 'crs': 'EPSG:4326'}
    args['region'] = map_region
    if dims:
        args['dimensions'] = dims
    if vis_params:
        keys = list(vis_params.keys())
        if cmap and 'palette' in keys:
            raise KeyError('cannot provide `palette` in vis_params if `cmap` is specified')
        elif cmap:
            args['palette'] = ','.join(build_palette(cmap))
        else:
            pass
        args = {**args, **vis_params}
    url = ee_object.getThumbUrl(args)
    response = requests.get(url)
    if response.status_code != 200:
        error = eval(response.content)['error']
        raise requests.exceptions.HTTPError(f'{error}')
    image = np.array(Image.open(BytesIO(response.content)))
    if image.shape[-1] == 2:
        image = np.concatenate([np.repeat(image[:, :, 0:1], 3, axis=2), image[:, :, -1:]], axis=2)
    ax.imshow(np.squeeze(image), extent=view_extent, origin='upper', transform=ccrs.PlateCarree(), zorder=1)
    return

def build_palette(cmap, n=256):
    if False:
        i = 10
        return i + 15
    "Creates hex color code palette from a matplotlib colormap\n\n    args:\n        cmap (str): string specifying matplotlib colormap to colorize image. If cmap is specified visParams cannot contain 'palette' key\n        n (int, optional): Number of hex color codes to create from colormap. Default is 256\n\n    returns:\n        palette (list[str]): list of hex color codes from matplotlib colormap for n intervals\n    "
    colormap = cm.get_cmap(cmap, n)
    vals = np.linspace(0, 1, n)
    palette = list(map(lambda x: colors.rgb2hex(colormap(x)[:3]), vals))
    return palette

def add_colorbar(ax, vis_params, loc=None, cmap='gray', discrete=False, label=None, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Add a colorbar to the map based on visualization parameters provided\n    args:\n        ax (cartopy.mpl.geoaxes.GeoAxesSubplot | cartopy.mpl.geoaxes.GeoAxes): required cartopy GeoAxesSubplot object to add image overlay to\n        loc (str, optional): string specifying the position\n        vis_params (dict, optional): visualization parameters as a dictionary. See https://developers.google.com/earth-engine/guides/image_visualization for options.\n        **kwargs: remaining keyword arguments are passed to colorbar()\n\n    raises:\n        Warning: If \'discrete\' is true when "palette" key is not in visParams\n        ValueError: If `ax` is not of type cartopy.mpl.geoaxes.GeoAxesSubplot\n        ValueError: If \'cmap\' or "palette" key in visParams is not provided\n        ValueError: If "min" in visParams is not of type scalar\n        ValueError: If "max" in visParams is not of type scalar\n        ValueError: If \'loc\' or \'cax\' keywords are not provided\n        ValueError: If \'loc\' is not of type str or does not equal available options\n    '
    if type(ax) not in [GeoAxes, GeoAxesSubplot]:
        raise ValueError('provided axes not of type cartopy.mpl.geoaxes.GeoAxes or cartopy.mpl.geoaxes.GeoAxesSubplot')
    if loc:
        if type(loc) == str and loc in ['left', 'right', 'bottom', 'top']:
            if 'posOpts' not in kwargs:
                posOpts = {'left': [0.01, 0.25, 0.02, 0.5], 'right': [0.88, 0.25, 0.02, 0.5], 'bottom': [0.25, 0.15, 0.5, 0.02], 'top': [0.25, 0.88, 0.5, 0.02]}
            else:
                posOpts = {'left': kwargs['posOpts'], 'right': kwargs['posOpts'], 'bottom': kwargs['posOpts'], 'top': kwargs['posOpts']}
                del kwargs['posOpts']
            cax = ax.figure.add_axes(posOpts[loc])
            if loc == 'left':
                mpl.pyplot.subplots_adjust(left=0.18)
            elif loc == 'right':
                mpl.pyplot.subplots_adjust(right=0.85)
            else:
                pass
        else:
            raise ValueError('provided loc not of type str. options are "left", "top", "right", or "bottom"')
    elif 'cax' in kwargs:
        cax = kwargs['cax']
        kwargs = {key: kwargs[key] for key in kwargs.keys() if key != 'cax'}
    else:
        raise ValueError('loc or cax keywords must be specified')
    vis_keys = list(vis_params.keys())
    if vis_params:
        if 'min' in vis_params:
            vmin = vis_params['min']
            if type(vmin) not in (int, float):
                raise ValueError('provided min value not of scalar type')
        else:
            vmin = 0
        if 'max' in vis_params:
            vmax = vis_params['max']
            if type(vmax) not in (int, float):
                raise ValueError('provided max value not of scalar type')
        else:
            vmax = 1
        if 'opacity' in vis_params:
            alpha = vis_params['opacity']
            if type(alpha) not in (int, float):
                raise ValueError('provided opacity value of not type scalar')
        elif 'alpha' in kwargs:
            alpha = kwargs['alpha']
        else:
            alpha = 1
        if cmap is not None:
            if discrete:
                warnings.warn('discrete keyword used when "palette" key is supplied with visParams, creating a continuous colorbar...')
            cmap = mpl.pyplot.get_cmap(cmap)
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        if 'palette' in vis_keys:
            hexcodes = vis_params['palette']
            hexcodes = [i if i[0] == '#' else '#' + i for i in hexcodes]
            if discrete:
                cmap = mpl.colors.ListedColormap(hexcodes)
                vals = np.linspace(vmin, vmax, cmap.N + 1)
                norm = mpl.colors.BoundaryNorm(vals, cmap.N)
            else:
                cmap = mpl.colors.LinearSegmentedColormap.from_list('custom', hexcodes, N=256)
                norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        elif cmap is not None:
            if discrete:
                warnings.warn('discrete keyword used when "palette" key is supplied with visParams, creating a continuous colorbar...')
            cmap = mpl.pyplot.get_cmap(cmap)
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        else:
            raise ValueError('cmap keyword or "palette" key in visParams must be provided')
    tick_font_size = None
    if 'tick_font_size' in kwargs:
        tick_font_size = kwargs.pop('tick_font_size')
    label_font_family = None
    if 'label_font_family' in kwargs:
        label_font_family = kwargs.pop('label_font_family')
    label_font_size = None
    if 'label_font_size' in kwargs:
        label_font_size = kwargs.pop('label_font_size')
    cb = mpl.colorbar.ColorbarBase(cax, norm=norm, alpha=alpha, cmap=cmap, **kwargs)
    if label is not None:
        if label_font_size is not None and label_font_family is not None:
            cb.set_label(label, fontsize=label_font_size, family=label_font_family)
        elif label_font_size is not None and label_font_family is None:
            cb.set_label(label, fontsize=label_font_size)
        elif label_font_size is None and label_font_family is not None:
            cb.set_label(label, family=label_font_family)
        else:
            cb.set_label(label)
    elif 'bands' in vis_keys:
        cb.set_label(vis_params['bands'])
    if tick_font_size is not None:
        cb.ax.tick_params(labelsize=tick_font_size)

def _buffer_box(bbox, interval):
    if False:
        for i in range(10):
            print('nop')
    'Helper function to buffer a bounding box to the nearest multiple of interval\n\n    args:\n        bbox (list[float]): list of float values specifying coordinates, expects order to be [W,E,S,N]\n        interval (float): float specifying multiple at which to buffer coordianates to\n\n    returns:\n        extent (tuple[float]): returns tuple of buffered coordinates rounded to interval in order of [W,E,S,N]\n    '
    if bbox[0] % interval != 0:
        xmin = bbox[0] - bbox[0] % interval
    else:
        xmin = bbox[0]
    if bbox[1] % interval != 0:
        xmax = bbox[1] + (interval - bbox[1] % interval)
    else:
        xmax = bbox[1]
    if bbox[2] % interval != 0:
        ymin = bbox[2] - bbox[2] % interval
    else:
        ymin = bbox[2]
    if bbox[3] % interval != 0:
        ymax = bbox[3] + (interval - bbox[3] % interval)
    else:
        ymax = bbox[3]
    return (xmin, xmax, ymin, ymax)

def bbox_to_extent(bbox):
    if False:
        i = 10
        return i + 15
    'Helper function to reorder a list of coordinates from [W,S,E,N] to [W,E,S,N]\n\n    args:\n        bbox (list[float]): list (or tuple) or coordinates in the order of [W,S,E,N]\n\n    returns:\n        extent (tuple[float]): tuple of coordinates in the order of [W,E,S,N]\n    '
    return (bbox[0], bbox[2], bbox[1], bbox[3])

def add_gridlines(ax, interval=None, n_ticks=None, xs=None, ys=None, buffer_out=True, xtick_rotation='horizontal', ytick_rotation='horizontal', **kwargs):
    if False:
        print('Hello World!')
    'Helper function to add gridlines and format ticks to map\n\n    args:\n        ax (cartopy.mpl.geoaxes.GeoAxesSubplot | cartopy.mpl.geoaxes.GeoAxes): required cartopy GeoAxesSubplot object to add the gridlines to\n        interval (float | list[float], optional): float specifying an interval at which to create gridlines, units are decimal degrees. lists will be interpreted a [x_interval, y_interval]. default = None\n        n_ticks (int | list[int], optional): integer specifying number gridlines to create within map extent. lists will be interpreted a [nx, ny]. default = None\n        xs (list, optional): list of x coordinates to create gridlines. default = None\n        ys (list, optional): list of y coordinates to create gridlines. default = None\n        buffer_out (boolean, optional): boolean option to buffer out the extent to insure coordinates created cover map extent. default=true\n        xtick_rotation (str | float, optional):\n        ytick_rotation (str | float, optional):\n        **kwargs: remaining keyword arguments are passed to gridlines()\n\n    raises:\n        ValueError: if all interval, n_ticks, or (xs,ys) are set to None\n\n    '
    view_extent = ax.get_extent()
    extent = view_extent
    if xs is not None:
        xmain = xs
    elif interval is not None:
        if isinstance(interval, Iterable):
            xspace = interval[0]
        else:
            xspace = interval
        if buffer_out:
            extent = _buffer_box(extent, xspace)
        xmain = np.arange(extent[0], extent[1] + xspace, xspace)
    elif n_ticks is not None:
        if isinstance(n_ticks, Iterable):
            n_x = n_ticks[0]
        else:
            n_x = n_ticks
        xmain = np.linspace(extent[0], extent[1], n_x)
    else:
        raise ValueError('one of variables interval, n_ticks, or xs must be defined. If you would like default gridlines, please use `ax.gridlines()`')
    if ys is not None:
        ymain = ys
    elif interval is not None:
        if isinstance(interval, Iterable):
            yspace = interval[1]
        else:
            yspace = interval
        if buffer_out:
            extent = _buffer_box(extent, yspace)
        ymain = np.arange(extent[2], extent[3] + yspace, yspace)
    elif n_ticks is not None:
        if isinstance(n_ticks, Iterable):
            n_y = n_ticks[1]
        else:
            n_y = n_ticks
        ymain = np.linspace(extent[2], extent[3], n_y)
    else:
        raise ValueError('one of variables interval, n_ticks, or ys must be defined. If you would like default gridlines, please use `ax.gridlines()`')
    ax.gridlines(xlocs=xmain, ylocs=ymain, **kwargs)
    xin = xmain[(xmain >= view_extent[0]) & (xmain <= view_extent[1])]
    yin = ymain[(ymain >= view_extent[2]) & (ymain <= view_extent[3])]
    ax.set_xticks(xin, crs=ccrs.PlateCarree())
    ax.set_yticks(yin, crs=ccrs.PlateCarree())
    ax.set_xticklabels(xin, rotation=xtick_rotation, ha='center')
    ax.set_yticklabels(yin, rotation=ytick_rotation, va='center')
    ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
    ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)
    return

def pad_view(ax, factor=0.05):
    if False:
        for i in range(10):
            print('nop')
    'Function to pad area around the view extent of a map, used for visual appeal\n\n    args:\n        ax (cartopy.mpl.geoaxes.GeoAxesSubplot | cartopy.mpl.geoaxes.GeoAxes): required cartopy GeoAxesSubplot object to pad view extent\n        factor (float | list[float], optional): factor to pad view extent accepts float [0-1] of a list of floats which will be interpreted at [xfactor, yfactor]\n\n    '
    view_extent = ax.get_extent()
    if isinstance(factor, Iterable):
        (xfactor, yfactor) = factor
    else:
        (xfactor, yfactor) = (factor, factor)
    x_diff = view_extent[1] - view_extent[0]
    y_diff = view_extent[3] - view_extent[2]
    xmin = view_extent[0] - x_diff * xfactor
    xmax = view_extent[1] + x_diff * xfactor
    ymin = view_extent[2] - y_diff * yfactor
    ymax = view_extent[3] + y_diff * yfactor
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(xmin, xmax)
    return

def add_north_arrow(ax, text='N', xy=(0.1, 0.1), arrow_length=0.1, text_color='black', arrow_color='black', fontsize=20, width=5, headwidth=15, ha='center', va='center'):
    if False:
        return 10
    'Add a north arrow to the map.\n\n    Args:\n        ax (cartopy.mpl.geoaxes.GeoAxesSubplot | cartopy.mpl.geoaxes.GeoAxes): required cartopy GeoAxesSubplot object.\n        text (str, optional): Text for north arrow. Defaults to "N".\n        xy (tuple, optional): Location of the north arrow. Each number representing the percentage length of the map from the lower-left cornor. Defaults to (0.1, 0.1).\n        arrow_length (float, optional): Length of the north arrow. Defaults to 0.1 (10% length of the map).\n        text_color (str, optional): Text color. Defaults to "black".\n        arrow_color (str, optional): North arrow color. Defaults to "black".\n        fontsize (int, optional): Text font size. Defaults to 20.\n        width (int, optional): Width of the north arrow. Defaults to 5.\n        headwidth (int, optional): head width of the north arrow. Defaults to 15.\n        ha (str, optional): Horizontal alignment. Defaults to "center".\n        va (str, optional): Vertical alignment. Defaults to "center".\n    '
    ax.annotate(text, xy=xy, xytext=(xy[0], xy[1] - arrow_length), color=text_color, arrowprops=dict(facecolor=arrow_color, width=width, headwidth=headwidth), ha=ha, va=va, fontsize=fontsize, xycoords=ax.transAxes)
    return

def convert_SI(val, unit_in, unit_out):
    if False:
        print('Hello World!')
    'Unit converter.\n\n    Args:\n        val (float): The value to convert.\n        unit_in (str): The input unit.\n        unit_out (str): The output unit.\n\n    Returns:\n        float: The value after unit conversion.\n    '
    SI = {'cm': 0.01, 'm': 1.0, 'km': 1000.0, 'inch': 0.0254, 'foot': 0.3048, 'mile': 1609.34}
    return val * SI[unit_in] / SI[unit_out]

def add_scale_bar(ax, metric_distance=4, unit='km', at_x=(0.05, 0.5), at_y=(0.08, 0.11), max_stripes=5, ytick_label_margins=0.25, fontsize=8, font_weight='bold', rotation=0, zorder=999, paddings={'xmin': 0.05, 'xmax': 0.05, 'ymin': 1.5, 'ymax': 0.5}, bbox_kwargs={'facecolor': 'white', 'edgecolor': 'black', 'alpha': 0.5}):
    if False:
        i = 10
        return i + 15
    '\n    Add a scale bar to the map.\n\n    Args:\n        ax (cartopy.mpl.geoaxes.GeoAxesSubplot | cartopy.mpl.geoaxes.GeoAxes): required cartopy GeoAxesSubplot object.\n        metric_distance (int | float, optional): length in meters of each region of the scale bar. Default to 4.\n        unit (str, optional): scale bar distance unit. Default to "km"\n        at_x (float, optional): target axes X coordinates (0..1) of box (= left, right). Default to (0.05, 0.2).\n        at_y (float, optional): axes Y coordinates (0..1) of box (= lower, upper). Default to (0.08, 0.11).\n        max_stripes (int, optional): typical/maximum number of black+white regions. Default to 5.\n        ytick_label_margins (float, optional): Location of distance labels on the Y axis. Default to 0.25.\n        fontsize (int, optional): scale bar text size. Default to 8.\n        font_weight (str, optional):font weight. Default to \'bold\'.\n        rotation (int, optional): rotation of the length labels for each region of the scale bar. Default to 0.\n        zorder (float, optional): z order of the text bounding box.\n        paddings (dict, optional): boundaries of the box that contains the scale bar.\n        bbox_kwargs (dict, optional): style of the box containing the scale bar.\n\n    '
    warnings.filterwarnings('ignore')

    def _crs_coord_project(crs_target, xcoords, ycoords, crs_source):
        if False:
            i = 10
            return i + 15
        'metric coordinates (x, y) from cartopy.crs_source'
        axes_coords = crs_target.transform_points(crs_source, xcoords, ycoords)
        return axes_coords

    def _add_bbox(ax, list_of_patches, paddings={}, bbox_kwargs={}):
        if False:
            print('Hello World!')
        '\n        Description:\n            This helper function adds a box behind the scalebar:\n                Code inspired by: https://stackoverflow.com/questions/17086847/box-around-text-in-matplotlib\n\n        '
        zorder = list_of_patches[0].get_zorder() - 1
        xmin = min([t.get_window_extent().xmin for t in list_of_patches])
        xmax = max([t.get_window_extent().xmax for t in list_of_patches])
        ymin = min([t.get_window_extent().ymin for t in list_of_patches])
        ymax = max([t.get_window_extent().ymax for t in list_of_patches])
        (xmin, ymin) = ax.transData.inverted().transform((xmin, ymin))
        (xmax, ymax) = ax.transData.inverted().transform((xmax, ymax))
        xmin = xmin - (xmax - xmin) * paddings['xmin']
        ymin = ymin - (ymax - ymin) * paddings['ymin']
        xmax = xmax + (xmax - xmin) * paddings['xmax']
        ymax = ymax + (ymax - ymin) * paddings['ymax']
        width = xmax - xmin
        height = ymax - ymin
        rect = patches.Rectangle((xmin, ymin), width, height, facecolor=bbox_kwargs['facecolor'], edgecolor=bbox_kwargs['edgecolor'], alpha=bbox_kwargs['alpha'], transform=ax.projection, fill=True, clip_on=False, zorder=zorder)
        ax.add_patch(rect)
        return ax
    old_proj = ax.projection
    ax.projection = ccrs.PlateCarree()
    (lon_0, lon_1, lat_0, lat_1) = ax.get_extent(ax.projection.as_geodetic())
    central_lon = np.mean([lon_0, lon_1])
    central_lat = np.mean([lat_0, lat_1])
    proj = ccrs.EquidistantConic(central_longitude=central_lon, central_latitude=central_lat)
    (x0, _, y0, y1) = ax.get_extent(proj)
    ymean = np.mean([y0, y1])
    (axfrac_ini, _) = at_x
    (ayfrac_ini, ayfrac_final) = at_y
    converted_metric_distance = convert_SI(metric_distance, unit, 'm')
    xcoords = []
    ycoords = []
    xlabels = []
    for i in range(0, 1 + max_stripes):
        dx = converted_metric_distance * i + x0
        xlabels.append(metric_distance * i)
        xcoords.append(dx)
        ycoords.append(ymean)
    xcoords = np.asanyarray(xcoords)
    ycoords = np.asanyarray(ycoords)
    (x_targets, _, _) = _crs_coord_project(ax.projection, xcoords, ycoords, proj).T
    x_targets = [x + axfrac_ini * (lon_1 - lon_0) for x in x_targets]
    transform = ax.projection
    (xl0, xl1) = (x_targets[0], x_targets[-1])
    (yl0, yl1) = [lat_0 + ay_frac * (lat_1 - lat_0) for ay_frac in [ayfrac_ini, ayfrac_final]]
    y_margin = (yl1 - yl0) * ytick_label_margins
    fill_colors = ['black', 'white']
    i_color = 0
    filled_boxs = []
    for (xi0, xi1) in zip(x_targets[:-1], x_targets[1:]):
        filled_box = plt.fill((xi0, xi1, xi1, xi0, xi0), (yl0, yl0, yl1, yl1, yl0), fill_colors[i_color], transform=transform, clip_on=False, zorder=zorder)
        filled_boxs.append(filled_box[0])
        plt.plot((xi0, xi1, xi1, xi0, xi0), (yl0, yl0, yl1, yl1, yl0), 'black', clip_on=False, transform=transform, zorder=zorder)
        i_color = 1 - i_color
    _add_bbox(ax, filled_boxs, bbox_kwargs=bbox_kwargs, paddings=paddings)
    for x in x_targets:
        plt.plot((x, x), (yl0, yl0 - y_margin), 'black', transform=transform, zorder=zorder, clip_on=False)
    font_props = mfonts.FontProperties(size=fontsize, weight=font_weight)
    plt.text(0.5 * (xl0 + xl1), yl1 + y_margin, unit, color='black', verticalalignment='bottom', horizontalalignment='center', fontproperties=font_props, transform=transform, clip_on=False, zorder=zorder)
    for (x, xlabel) in zip(x_targets, xlabels):
        plt.text(x, yl0 - 2 * y_margin, '{:g}'.format(xlabel), verticalalignment='top', horizontalalignment='center', fontproperties=font_props, transform=transform, rotation=rotation, clip_on=False, zorder=zorder + 1)
    ax.projection = old_proj
    ax.get_figure().canvas.draw()

def add_scale_bar_lite(ax, length=None, xy=(0.5, 0.05), linewidth=3, fontsize=20, color='black', unit='km', ha='center', va='bottom'):
    if False:
        return 10
    'Add a lite version of scale bar to the map. Reference: https://stackoverflow.com/a/50674451/2676166\n\n    Args:\n        ax (cartopy.mpl.geoaxes.GeoAxesSubplot | cartopy.mpl.geoaxes.GeoAxes): required cartopy GeoAxesSubplot object.\n        length ([type], optional): Length of the scale car. Defaults to None.\n        xy (tuple, optional): Location of the north arrow. Each number representing the percentage length of the map from the lower-left cornor. Defaults to (0.1, 0.1).\n        linewidth (int, optional): Line width of the scale bar. Defaults to 3.\n        fontsize (int, optional): Text font size. Defaults to 20.\n        color (str, optional): Color for the scale bar. Defaults to "black".\n        unit (str, optional): Length unit for the scale bar. Defaults to "km".\n        ha (str, optional): Horizontal alignment. Defaults to "center".\n        va (str, optional): Vertical alignment. Defaults to "bottom".\n\n    '
    allow_units = ['cm', 'm', 'km', 'inch', 'foot', 'mile']
    if unit not in allow_units:
        print('The unit must be one of the following: {}'.format(', '.join(allow_units)))
        return
    num = length
    (llx0, llx1, lly0, lly1) = ax.get_extent(ccrs.PlateCarree())
    sbllx = (llx1 + llx0) / 2
    sblly = lly0 + (lly1 - lly0) * xy[1]
    tmc = ccrs.TransverseMercator(sbllx, sblly, approx=True)
    (x0, x1, y0, y1) = ax.get_extent(tmc)
    sbx = x0 + (x1 - x0) * xy[0]
    sby = y0 + (y1 - y0) * xy[1]
    if not length:
        length = (x1 - x0) / 5000
        ndim = int(np.floor(np.log10(length)))
        length = round(length, -ndim)

        def scale_number(x):
            if False:
                i = 10
                return i + 15
            if str(x)[0] in ['1', '2', '5']:
                return int(x)
            else:
                return scale_number(x - 10 ** ndim)
        length = scale_number(length)
        num = length
    else:
        length = convert_SI(length, unit, 'km')
    bar_xs = [sbx - length * 500, sbx + length * 500]
    ax.plot(bar_xs, [sby, sby], transform=tmc, color=color, linewidth=linewidth)
    ax.text(sbx, sby, str(num) + ' ' + unit, transform=tmc, horizontalalignment=ha, verticalalignment=va, color=color, fontsize=fontsize)
    return

def create_legend(linewidth=None, linestyle=None, color=None, marker=None, markersize=None, markeredgewidth=None, markeredgecolor=None, markerfacecolor=None, markerfacecoloralt=None, fillstyle=None, antialiased=None, dash_capstyle=None, solid_capstyle=None, dash_joinstyle=None, solid_joinstyle=None, pickradius=5, drawstyle=None, markevery=None, **kwargs):
    if False:
        return 10
    if linewidth is None and marker is None:
        raise ValueError('Either linewidth or marker must be specified.')

def add_legend(ax, legend_elements=None, loc='lower right', font_size=14, font_weight='normal', font_color='black', font_family=None, title=None, title_fontize=16, title_fontproperties=None, **kwargs):
    if False:
        i = 10
        return i + 15
    'Adds a legend to the map. The legend elements can be formatted as:\n    legend_elements = [Line2D([], [], color=\'#00ffff\', lw=2, label=\'Coastline\'),\n        Line2D([], [], marker=\'o\', color=\'#A8321D\', label=\'City\', markerfacecolor=\'#A8321D\', markersize=10, ls =\'\')]\n        For more legend properties, see: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html\n\n    Args:\n        ax (cartopy.mpl.geoaxes.GeoAxesSubplot | cartopy.mpl.geoaxes.GeoAxes): required cartopy GeoAxesSubplot object.\n        legend_elements (list, optional): A list of legend elements. Defaults to None.\n        loc (str, optional): Location of the legend, can be any of [\'upper left\', \'upper right\', \'lower left\', \'lower right\']. Defaults to "lower right".\n        font_size(int|string, optional): Font size. Either an absolute font size or an relative value of \'xx-small\', \'x-small\', \'small\', \'medium\', \'large\', \'x-large\', \'xx-large\'. defaults to 14.\n        font_weight(string|int, optional): Font weight. A numeric value in the range 0-1000 or one of \'ultralight\', \'light\', \'normal\' (default), \'regular\', \'book\', \'medium\', \'roman\', \'semibold\', \'demibold\', \'demi\', \'bold\', \'heavy\', \'extra bold\', \'black\'. Defaults to \'normal\'.\n        font_color(str, optional): Text color. Defaults to "black".\n        font_family(string, optional): Name of font family. Set to a font family like \'SimHei\' if you want to show Chinese in the legend. Defaults to None.\n    Raises:\n        Exception: If the legend fails to add.\n    '
    from matplotlib.lines import Line2D
    if title_fontize is not None and title_fontproperties is not None:
        raise ValueError('title_fontize and title_fontproperties cannot be both set.')
    elif title_fontize is not None:
        kwargs['title_fontsize'] = title_fontize
    elif title_fontproperties is not None:
        kwargs['title_fontproperties'] = title_fontproperties
    try:
        if legend_elements is None:
            legend_elements = [Line2D([], [], color='#00ffff', lw=2, label='Coastline'), Line2D([], [], marker='o', color='#A8321D', label='City', markerfacecolor='#A8321D', markersize=10, ls='')]
        if font_family is not None:
            fontdict = {'family': font_family, 'size': font_size, 'weight': font_weight}
        else:
            fontdict = {'size': font_size, 'weight': font_weight}
        leg = ax.legend(handles=legend_elements, loc=loc, prop=fontdict, title=title, **kwargs)
        if font_color != 'black':
            for text in leg.get_texts():
                text.set_color(font_color)
        return
    except Exception as e:
        raise Exception(e)

def get_image_collection_gif(ee_ic, out_dir, out_gif, vis_params, region, cmap=None, proj=None, fps=10, mp4=False, grid_interval=None, plot_title='', date_format='YYYY-MM-dd', fig_size=(10, 10), dpi_plot=100, file_format='png', north_arrow_dict={}, scale_bar_dict={}, verbose=True):
    if False:
        i = 10
        return i + 15
    'Download all the images in an image collection and use them to generate a gif/video.\n    Args:\n        ee_ic (object): ee.ImageCollection\n        out_dir (str): The output directory of images and video.\n        out_gif (str): The name of the gif file.\n        vis_params (dict): Visualization parameters as a dictionary.\n        region (list | tuple): Geospatial region of the image to render in format [E,S,W,N].\n        fps (int, optional): Video frames per second. Defaults to 10.\n        mp4 (bool, optional): Whether to create mp4 video.\n        grid_interval (float | tuple[float]): Float specifying an interval at which to create gridlines, units are decimal degrees. lists will be interpreted a (x_interval, y_interval), such as (0.1, 0.1). Defaults to None.\n        plot_title (str): Plot title. Defaults to "".\n        date_format (str, optional): A pattern, as described at http://joda-time.sourceforge.net/apidocs/org/joda/time/format/DateTimeFormat.html. Defaults to "YYYY-MM-dd".\n        fig_size (tuple, optional): Size of the figure.\n        dpi_plot (int, optional): The resolution in dots per inch of the plot.\n        file_format (str, optional): Either \'png\' or \'jpg\'.\n        north_arrow_dict (dict, optional): Parameters for the north arrow. See https://geemap.org/cartoee/#geemap.cartoee.add_north_arrow. Defaults to {}.\n        scale_bar_dict (dict, optional): Parameters for the scale bar. See https://geemap.org/cartoee/#geemap.cartoee.add_scale_bar. Defaults. to {}.\n        verbose (bool, optional): Whether or not to print text when the program is running. Defaults to True.\n    '
    from .geemap import png_to_gif, jpg_to_gif
    out_dir = os.path.abspath(out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_gif = os.path.join(out_dir, out_gif)
    count = int(ee_ic.size().getInfo())
    names = ee_ic.aggregate_array('system:index').getInfo()
    images = ee_ic.toList(count)
    dates = ee_ic.aggregate_array('system:time_start')
    dates = dates.map(lambda d: ee.Date(d).format(date_format)).getInfo()
    digits = len(str(len(dates)))
    img_list = []
    for (i, date) in enumerate(dates):
        image = ee.Image(images.get(i))
        name = str(i + 1).zfill(digits) + '.' + file_format
        out_img = os.path.join(out_dir, name)
        img_list.append(out_img)
        if verbose:
            print(f'Downloading {i + 1}/{count}: {name} ...')
        fig = plt.figure(figsize=fig_size)
        fig.patch.set_facecolor('white')
        ax = get_map(image, region=region, vis_params=vis_params, cmap=cmap, proj=proj)
        if grid_interval is not None:
            add_gridlines(ax, interval=grid_interval, linestyle=':')
        if len(plot_title) > 0:
            ax.set_title(label=plot_title + ' ' + date + '\n', fontsize=15)
        if len(scale_bar_dict) > 0:
            add_scale_bar_lite(ax, **scale_bar_dict)
        if len(north_arrow_dict) > 0:
            add_north_arrow(ax, **north_arrow_dict)
        plt.savefig(fname=out_img, dpi=dpi_plot, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.clf()
        plt.close()
    out_gif = os.path.abspath(out_gif)
    if file_format == 'png':
        png_to_gif(out_dir, out_gif, fps)
    elif file_format == 'jpg':
        jpg_to_gif(out_dir, out_gif, fps)
    if verbose:
        print(f'GIF saved to {out_gif}')
    if mp4:
        video_filename = out_gif.replace('.gif', '.mp4')
        try:
            import cv2
        except ImportError:
            print('Installing opencv-python ...')
            subprocess.check_call(['python', '-m', 'pip', 'install', 'opencv-python'])
            import cv2
        output_video_file_name = os.path.join(out_dir, video_filename)
        frame = cv2.imread(img_list[0])
        (height, width, _) = frame.shape
        frame_size = (width, height)
        fps_video = fps
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        def convert_frames_to_video(input_list, output_video_file_name, fps_video, frame_size):
            if False:
                for i in range(10):
                    print('nop')
            'Convert frames to video\n\n            Args:\n\n                input_list (list): Downloaded Image Name List.\n                output_video_file_name (str): The name of the video file in the image directory.\n                fps_video (int): Video frames per second.\n                frame_size (tuple): Frame size.\n            '
            out = cv2.VideoWriter(output_video_file_name, fourcc, fps_video, frame_size)
            num_frames = len(input_list)
            for i in range(num_frames):
                img_path = input_list[i]
                img = cv2.imread(img_path)
                out.write(img)
            out.release()
            cv2.destroyAllWindows()
        convert_frames_to_video(input_list=img_list, output_video_file_name=output_video_file_name, fps_video=fps_video, frame_size=frame_size)
        if verbose:
            print(f'MP4 saved to {output_video_file_name}')

def savefig(fig, fname, dpi='figure', bbox_inches='tight', **kwargs):
    if False:
        i = 10
        return i + 15
    "Save figure to file. It wraps the matplotlib.pyplot.savefig() function.\n            See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html for more details.\n\n    Args:\n        fig (matplotlib.figure.Figure): The figure to save.\n        fname (str): A path to a file, or a Python file-like object.\n        dpi (int | str, optional): The resolution in dots per inch. If 'figure', use the figure's dpi value. Defaults to 'figure'.\n        bbox_inches (str, optional): Bounding box in inches: only the given portion of the figure is saved.\n            If 'tight', try to figure out the tight bbox of the figure.\n        kwargs (dict, optional): Additional keyword arguments are passed on to the savefig() method.\n    "
    fig.savefig(fname=fname, dpi=dpi, bbox_inches=bbox_inches, **kwargs)