"""Module for creating interactive maps with plotly."""
import os
import numpy as np
import pandas as pd
import ipywidgets as widgets
from .basemaps import xyz_to_plotly
from .common import *
from .osm import *
from . import examples
try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    raise ImportError("This module requires the plotly package. Please install it using 'pip install plotly'.")
basemaps = xyz_to_plotly()

class Canvas:
    """The widgets.HBox containing the map and a toolbar."""

    def __init__(self, map, map_min_width='90%', map_max_width='98%', map_refresh=False, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "Initialize the Canvas.\n\n        Args:\n            map (go.FigureWidget): The map to display.\n            map_min_width (str, optional): The minimum width of the map. Defaults to '90%'.\n            map_max_width (str, optional): The maximum width of the map. Defaults to '98%'.\n            map_refresh (bool, optional): Whether to refresh the map when the map is resized. Defaults to False.\n        "
        from .toolbar import plotly_toolbar
        map_widget = widgets.Output(layout=widgets.Layout(width=map_max_width))
        with map_widget:
            display(map)
        self.map = map
        self.map_min_width = map_min_width
        self.map_max_width = map_max_width
        self.map_refresh = map_refresh
        self.map_widget = map_widget
        container_widget = widgets.VBox()
        self.container_widget = container_widget
        toolbar_widget = plotly_toolbar(self)
        sidebar_widget = widgets.VBox([toolbar_widget, container_widget])
        canvas = widgets.HBox([map_widget, sidebar_widget])
        self.canvas = canvas
        self.toolbar_widget = toolbar_widget

    def toolbar_reset(self):
        if False:
            i = 10
            return i + 15
        'Reset the toolbar so that no tool is selected.'
        if hasattr(self, '_toolbar'):
            toolbar_grid = self._toolbar
            for tool in toolbar_grid.children:
                tool.value = False

class Map(go.FigureWidget):
    """The Map class inherits the Plotly FigureWidget class. More info at https://plotly.com/python/figurewidget."""

    def __init__(self, center=(20, 0), zoom=1, basemap='open-street-map', height=600, **kwargs):
        if False:
            i = 10
            return i + 15
        'Initializes a map. More info at https://plotly.com/python/mapbox-layers/\n\n        Args:\n            center (tuple, optional): Center of the map. Defaults to (20, 0).\n            zoom (int, optional): Zoom level of the map. Defaults to 1.\n            basemap (str, optional): Can be one of string from "open-street-map", "carto-positron", "carto-darkmatter", "stamen-terrain", "stamen-toner" or "stamen-watercolor" . Defaults to \'open-street-map\'.\n            height (int, optional): Height of the map. Defaults to 600.\n        '
        if 'ee_initialize' not in kwargs.keys():
            kwargs['ee_initialize'] = True
        if kwargs['ee_initialize']:
            ee_initialize()
        kwargs.pop('ee_initialize')
        super().__init__(**kwargs)
        self.add_scattermapbox()
        self.update_layout({'mapbox': {'style': basemap, 'center': {'lat': center[0], 'lon': center[1]}, 'zoom': zoom}, 'margin': {'r': 0, 't': 0, 'l': 0, 'b': 0}, 'height': height})

    def show(self, toolbar=True, map_min_width='91%', map_max_width='98%', refresh=False, **kwargs):
        if False:
            while True:
                i = 10
        "Shows the map.\n\n        Args:\n            toolbar (bool, optional): Whether to show the toolbar. Defaults to True.\n            map_min_width (str, optional): The minimum width of the map. Defaults to '91%'.\n            map_max_width (str, optional): The maximum width of the map. Defaults to '98%'.\n            refresh (bool, optional): Whether to refresh the map when the map is resized. Defaults to False.\n\n        Returns:\n            Canvas: [description]\n        "
        if not toolbar:
            super().show(**kwargs)
        else:
            canvas = Canvas(self, map_min_width=map_min_width, map_max_width=map_max_width, map_refresh=refresh)
            return canvas.canvas

    def clear_controls(self):
        if False:
            i = 10
            return i + 15
        'Removes all controls from the map.'
        config = {'scrollZoom': True, 'displayModeBar': False, 'editable': True, 'showLink': False, 'displaylogo': False}
        self.show(toolbar=False, config=config)

    def add_controls(self, controls):
        if False:
            while True:
                i = 10
        "Adds controls to the map.\n\n        Args:\n            controls (list): List of controls to add, e.g., ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape'] See https://bit.ly/33Tmqxr\n        "
        if isinstance(controls, str):
            controls = [controls]
        elif not isinstance(controls, list):
            raise ValueError('Controls must be a string or a list of strings. See https://bit.ly/33Tmqxr')
        self.update_layout(modebar_add=controls)

    def remove_controls(self, controls):
        if False:
            return 10
        'Removes controls to the map.\n\n        Args:\n            controls (list): List of controls to remove, e.g., ["zoomin", "zoomout", "toimage", "pan", "resetview"]. See https://bit.ly/3Jk7wkb\n        '
        if isinstance(controls, str):
            controls = [controls]
        elif not isinstance(controls, list):
            raise ValueError('Controls must be a string or a list of strings. See https://bit.ly/3Jk7wkb')
        self.update_layout(modebar_remove=controls)

    def set_center(self, lat, lon, zoom=None):
        if False:
            i = 10
            return i + 15
        'Sets the center of the map.\n\n        Args:\n            lat (float): Latitude.\n            lon (float): Longitude.\n            zoom (int, optional): Zoom level of the map. Defaults to None.\n        '
        self.update_layout(mapbox=dict(center=dict(lat=lat, lon=lon), zoom=zoom if zoom is not None else self.layout.mapbox.zoom))

    def add_basemap(self, basemap='ROADMAP'):
        if False:
            while True:
                i = 10
        "Adds a basemap to the map.\n\n        Args:\n            basemap (str, optional): Can be one of string from basemaps. Defaults to 'ROADMAP'.\n        "
        if basemap not in basemaps:
            raise ValueError(f"Basemap {basemap} not found. Choose from {','.join(basemaps.keys())}")
        if basemap in self.get_tile_layers():
            self.remove_basemap(basemap)
        layers = list(self.layout.mapbox.layers) + [basemaps[basemap]]
        self.update_layout(mapbox_layers=layers)

    def remove_basemap(self, name):
        if False:
            print('Hello World!')
        'Removes a basemap from the map.\n\n        Args:\n            name (str): Name of the basemap to remove.\n        '
        layers = list(self.layout.mapbox.layers)
        layers = [layer for layer in layers if layer['name'] != name]
        self.layout.mapbox.layers = layers

    def add_mapbox_layer(self, style, access_token=None):
        if False:
            print('Hello World!')
        'Adds a mapbox layer to the map.\n\n        Args:\n            layer (str | dict): Layer to add. Can be "basic", "streets", "outdoors", "light", "dark", "satellite", or "satellite-streets". See https://plotly.com/python/mapbox-layers/ and https://docs.mapbox.com/mapbox-gl-js/style-spec/\n            access_token (str, optional): The Mapbox Access token. It can be set as an environment variable "MAPBOX_TOKEN". Defaults to None.\n        '
        if access_token is None:
            access_token = os.environ.get('MAPBOX_TOKEN')
        self.update_layout(mapbox_style=style, mapbox_layers=[], mapbox_accesstoken=access_token)

    def add_layer(self, layer, name=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Adds a layer to the map.\n\n        Args:\n            layer (plotly.graph_objects): Layer to add.\n            name (str, optional): Name of the layer. Defaults to None.\n        '
        if isinstance(name, str):
            layer.name = name
        self.add_trace(layer, **kwargs)

    def remove_layer(self, name):
        if False:
            i = 10
            return i + 15
        'Removes a layer from the map.\n\n        Args:\n            name (str): Name of the layer to remove.\n        '
        if name in self.get_data_layers():
            self.data = [layer for layer in self.data if layer.name != name]
        elif name in self.get_tile_layers():
            self.layout.mapbox.layers = [layer for layer in self.layout.mapbox.layers if layer['name'] != name]

    def clear_layers(self, clear_basemap=False):
        if False:
            for i in range(10):
                print('nop')
        'Clears all layers from the map.\n\n        Args:\n            clear_basemap (bool, optional): If True, clears the basemap. Defaults to False.\n        '
        if clear_basemap:
            self.data = []
        elif len(self.data) > 1:
            self.data = self.data[:1]

    def get_layers(self):
        if False:
            print('Hello World!')
        'Returns a dictionary of all layers in the map.\n        Returns:\n            dict: A dictionary of all layers in the map.\n        '
        layers = {}
        for layer in self.layout.mapbox.layers:
            if layer['name'] is not None:
                layers[layer['name']] = layer
        for layer in self.data:
            if layer.name is not None and layer.name != 'trace 0':
                layers[layer.name] = layer
        return layers

    def get_tile_layers(self):
        if False:
            print('Hello World!')
        'Returns a dictionary of tile layers in the map.\n\n        Returns:\n            dict: A dictionary of tile layers in the map.\n        '
        layers = {}
        for layer in self.layout.mapbox.layers:
            if layer['name'] is not None:
                layers[layer['name']] = layer
        return layers

    def get_data_layers(self):
        if False:
            i = 10
            return i + 15
        'Returns a dictionary of data layers in the map.\n\n        Returns:\n            dict: A dictionary of data layers in the map.\n        '
        layers = {}
        for layer in self.data:
            if layer.name is not None and layer.name != 'trace 0':
                layers[layer.name] = layer
        return layers

    def find_layer_index(self, name):
        if False:
            for i in range(10):
                print('nop')
        'Finds the index of a layer.\n\n        Args:\n            name (str): Name of the layer to find.\n\n        Returns:\n            int: Index of the layer.\n        '
        for (i, layer) in enumerate(self.data):
            if layer.name == name:
                return i
        for (i, layer) in enumerate(self.layout.mapbox.layers):
            if layer['name'] == name:
                return i
        return None

    def set_layer_visibility(self, name, show=True):
        if False:
            return 10
        'Sets the visibility of a layer.\n\n        Args:\n            name (str): Name of the layer to set.\n            show (bool, optional): If True, shows the layer. Defaults to True.\n        '
        if name in self.get_tile_layers():
            index = self.find_layer_index(name)
            self.layout.mapbox.layers[index].visible = show
        elif name in self.get_data_layers():
            index = self.find_layer_index(name)
            self.data[index].visible = show
        else:
            print(f'Layer {name} not found.')

    def set_layer_opacity(self, name, opacity=1):
        if False:
            print('Hello World!')
        'Sets the visibility of a layer.\n\n        Args:\n            name (str): Name of the layer to set.\n            opacity (float, optional): Opacity of the layer. Defaults to 1.\n        '
        if name in self.get_tile_layers():
            index = self.find_layer_index(name)
            self.layout.mapbox.layers[index].opacity = opacity
        elif name in self.get_data_layers():
            index = self.find_layer_index(name)
            layer = self.data[index]
            if hasattr(layer, 'opacity'):
                layer.opacity = opacity
            elif hasattr(layer, 'marker'):
                layer.marker.opacity = opacity
        else:
            print(f'Layer {name} not found.')

    def add_tile_layer(self, url, name='TileLayer', attribution='', opacity=1.0, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Adds a TileLayer to the map.\n\n        Args:\n            url (str): The URL of the tile layer.\n            name (str, optional): Name of the layer. Defaults to \'TileLayer\'.\n            attribution (str): The attribution to use. Defaults to "".\n            opacity (float, optional): The opacity of the layer. Defaults to 1.\n        '
        layer = {'below': 'traces', 'sourcetype': 'raster', 'sourceattribution': attribution, 'source': [url], 'opacity': opacity, 'name': name}
        layers = list(self.layout.mapbox.layers) + [layer]
        self.update_layout(mapbox_layers=layers)

    def add_ee_layer(self, ee_object, vis_params={}, name=None, shown=True, opacity=1.0, **kwargs):
        if False:
            return 10
        "Adds a given EE object to the map as a layer.\n\n        Args:\n            ee_object (Collection|Feature|Image|MapId): The object to add to the map.\n            vis_params (dict, optional): The visualization parameters. Defaults to {}.\n            name (str, optional): The name of the layer. Defaults to 'Layer N'.\n            shown (bool, optional): A flag indicating whether the layer should be on by default. Defaults to True.\n            opacity (float, optional): The layer's opacity represented as a number between 0 and 1. Defaults to 1.\n        "
        from box import Box
        image = None
        if vis_params is None:
            vis_params = {}
        if name is None:
            layer_count = len(self.layout.mapbox.layers)
            name = 'Layer ' + str(layer_count + 1)
        if not isinstance(ee_object, ee.Image) and (not isinstance(ee_object, ee.ImageCollection)) and (not isinstance(ee_object, ee.FeatureCollection)) and (not isinstance(ee_object, ee.Feature)) and (not isinstance(ee_object, ee.Geometry)):
            err_str = "\n\nThe image argument in 'addLayer' function must be an instance of one of ee.Image, ee.Geometry, ee.Feature or ee.FeatureCollection."
            raise AttributeError(err_str)
        if isinstance(ee_object, ee.geometry.Geometry) or isinstance(ee_object, ee.feature.Feature) or isinstance(ee_object, ee.featurecollection.FeatureCollection):
            features = ee.FeatureCollection(ee_object)
            width = 2
            if 'width' in vis_params:
                width = vis_params['width']
            color = '000000'
            if 'color' in vis_params:
                color = vis_params['color']
            image_fill = features.style(**{'fillColor': color}).updateMask(ee.Image.constant(0.5))
            image_outline = features.style(**{'color': color, 'fillColor': '00000000', 'width': width})
            image = image_fill.blend(image_outline)
        elif isinstance(ee_object, ee.image.Image):
            image = ee_object
        elif isinstance(ee_object, ee.imagecollection.ImageCollection):
            image = ee_object.mosaic()
        if 'palette' in vis_params:
            if isinstance(vis_params['palette'], tuple):
                vis_params['palette'] = list(vis_params['palette'])
            if isinstance(vis_params['palette'], Box):
                try:
                    vis_params['palette'] = vis_params['palette']['default']
                except Exception as e:
                    print('The provided palette is invalid.')
                    raise Exception(e)
            elif isinstance(vis_params['palette'], str):
                vis_params['palette'] = check_cmap(vis_params['palette'])
            elif not isinstance(vis_params['palette'], list):
                raise ValueError('The palette must be a list of colors or a string or a Box object.')
        map_id_dict = ee.Image(image).getMapId(vis_params)
        url = map_id_dict['tile_fetcher'].url_format
        self.add_tile_layer(url, name=name, attribution='Google Earth Engine', opacity=opacity, **kwargs)
        self.set_layer_visibility(name=name, show=shown)
    addLayer = add_ee_layer

    def add_cog_layer(self, url, name='Untitled', attribution='', opacity=1.0, bands=None, titiler_endpoint=None, **kwargs):
        if False:
            return 10
        'Adds a COG TileLayer to the map.\n\n        Args:\n            url (str): The URL of the COG tile layer, e.g., \'https://opendata.digitalglobe.com/events/california-fire-2020/pre-event/2018-02-16/pine-gulch-fire20/1030010076004E00.tif\'\n            name (str, optional): The layer name to use for the layer. Defaults to \'Untitled\'.\n            attribution (str, optional): The attribution to use. Defaults to \'\'.\n            opacity (float, optional): The opacity of the layer. Defaults to 1.\n            bands (list, optional): The bands to use. Defaults to None.\n            titiler_endpoint (str, optional): Titiler endpoint. Defaults to "https://titiler.xyz".\n            **kwargs: Arbitrary keyword arguments, including bidx, expression, nodata, unscale, resampling, rescale, color_formula, colormap, colormap_name, return_mask. See https://developmentseed.org/titiler/endpoints/cog/ and https://cogeotiff.github.io/rio-tiler/colormap/. To select a certain bands, use bidx=[1, 2, 3]\n        '
        tile_url = cog_tile(url, bands, titiler_endpoint, **kwargs)
        center = cog_center(url, titiler_endpoint)
        self.add_tile_layer(tile_url, name, attribution, opacity)
        self.set_center(lon=center[0], lat=center[1], zoom=10)

    def add_stac_layer(self, url=None, collection=None, item=None, assets=None, bands=None, titiler_endpoint=None, name='STAC Layer', attribution='', opacity=1.0, **kwargs):
        if False:
            return 10
        'Adds a STAC TileLayer to the map.\n\n        Args:\n            url (str): HTTP URL to a STAC item, e.g., https://canada-spot-ortho.s3.amazonaws.com/canada_spot_orthoimages/canada_spot5_orthoimages/S5_2007/S5_11055_6057_20070622/S5_11055_6057_20070622.json\n            collection (str): The Microsoft Planetary Computer STAC collection ID, e.g., landsat-8-c2-l2.\n            item (str): The Microsoft Planetary Computer STAC item ID, e.g., LC08_L2SP_047027_20201204_02_T1.\n            assets (str | list): The Microsoft Planetary Computer STAC asset ID, e.g., ["SR_B7", "SR_B5", "SR_B4"].\n            bands (list): A list of band names, e.g., ["SR_B7", "SR_B5", "SR_B4"]\n            titiler_endpoint (str, optional): Titiler endpoint, e.g., "https://titiler.xyz", "planetary-computer", "pc". Defaults to None.\n            name (str, optional): The layer name to use for the layer. Defaults to \'STAC Layer\'.\n            attribution (str, optional): The attribution to use. Defaults to \'\'.\n            opacity (float, optional): The opacity of the layer. Defaults to 1.\n        '
        tile_url = stac_tile(url, collection, item, assets, bands, titiler_endpoint, **kwargs)
        center = stac_center(url, collection, item, titiler_endpoint)
        self.add_tile_layer(tile_url, name, attribution, opacity)
        self.set_center(lon=center[0], lat=center[1], zoom=10)

    def add_planet_by_month(self, year=2016, month=1, api_key=None, token_name='PLANET_API_KEY', name=None, attribution='', opacity=1.0):
        if False:
            while True:
                i = 10
        'Adds Planet global mosaic by month to the map. To get a Planet API key, see https://developers.planet.com/quickstart/apis/\n\n        Args:\n            year (int, optional): The year of Planet global mosaic, must be >=2016. Defaults to 2016.\n            month (int, optional): The month of Planet global mosaic, must be 1-12. Defaults to 1.\n            api_key (str, optional): The Planet API key. Defaults to None.\n            token_name (str, optional): The environment variable name of the API key. Defaults to "PLANET_API_KEY".\n            name (str, optional): Name of the layer. Defaults to \'TileLayer\'.\n            attribution (str): The attribution to use. Defaults to "".\n            opacity (float, optional): The opacity of the layer. Defaults to 1.\n        '
        if name is None:
            name = str(year) + '-' + str(month).zfill(2)
        tile_url = planet_by_month(year, month, api_key, token_name)
        self.add_tile_layer(tile_url, name=name, attribution=attribution, opacity=opacity)

    def add_planet_by_quarter(self, year=2016, quarter=1, api_key=None, token_name='PLANET_API_KEY', name=None, attribution='', opacity=1.0):
        if False:
            print('Hello World!')
        'Adds Planet global mosaic by month to the map. To get a Planet API key, see https://developers.planet.com/quickstart/apis/\n\n        Args:\n            year (int, optional): The year of Planet global mosaic, must be >=2016. Defaults to 2016.\n            quarter (int, optional): The quarter of Planet global mosaic, must be 1-4. Defaults to 1.\n            api_key (str, optional): The Planet API key. Defaults to None.\n            token_name (str, optional): The environment variable name of the API key. Defaults to "PLANET_API_KEY".\n            name (str, optional): Name of the layer. Defaults to \'TileLayer\'.\n            attribution (str): The attribution to use. Defaults to "".\n            opacity (float, optional): The opacity of the layer. Defaults to 1.\n        '
        if name is None:
            name = str(year) + '-' + 'q' + str(quarter)
        tile_url = planet_by_quarter(year, quarter, api_key, token_name)
        self.add_tile_layer(tile_url, name=name, attribution=attribution, opacity=opacity)

    def save(self, file, format=None, width=None, height=None, scale=None, **kwargs):
        if False:
            i = 10
            return i + 15
        "Convert a map to a static image and write it to a file or writeable object\n\n        Args:\n            file (str): A string representing a local file path or a writeable object (e.g. a pathlib.Path object or an open file descriptor)\n            format (str, optional): The desired image format. One of png, jpg, jpeg, webp, svg, pdf, eps. Defaults to None.\n            width (int, optional): The width of the exported image in layout pixels. If the `scale` property is 1.0, this will also be the width of the exported image in physical pixels.. Defaults to None.\n            height (int, optional): The height of the exported image in layout pixels. If the `scale` property is 1.0, this will also be the height of the exported image in physical pixels.. Defaults to None.\n            scale (int, optional): The scale factor to use when exporting the figure. A scale factor larger than 1.0 will increase the image resolution with respect to the figure's layout pixel dimensions. Whereas as scale factor of less than 1.0 will decrease the image resolution.. Defaults to None.\n        "
        self.write_image(file, format=format, width=width, height=height, scale=scale, **kwargs)

    def add_choropleth_map(self, data, name=None, z=None, colorscale='Viridis', **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Adds a choropleth map to the map.\n\n        Args:\n            data (str): File path to vector data, e.g., https://raw.githubusercontent.com/giswqs/leafmap/master/examples/data/countries.geojson\n            name (str, optional): Name of the layer. Defaults to None.\n            z (str, optional): Z value of the data. Defaults to None.\n            colorscale (str, optional): Color scale of the data. Defaults to "Viridis".\n        '
        check_package('geopandas')
        import json
        import geopandas as gpd
        gdf = gpd.read_file(data).to_crs(epsg=4326)
        geojson = json.loads(gdf.to_json())
        self.add_choroplethmapbox(geojson=geojson, locations=gdf.index, z=gdf[z], name=name, colorscale=colorscale, **kwargs)

    def add_scatter_plot_demo(self, **kwargs):
        if False:
            i = 10
            return i + 15
        'Adds a scatter plot to the map.'
        lons = np.random.random(1000) * 360.0
        lats = np.random.random(1000) * 180.0 - 90.0
        z = np.random.random(1000) * 50.0
        self.add_scattermapbox(lon=lons, lat=lats, marker={'color': z}, name='Random points', **kwargs)

    def add_heatmap(self, data, latitude='latitude', longitude='longitude', z='value', radius=10, colorscale=None, name='Heat map', **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Adds a heat map to the map. Reference: https://plotly.com/python/mapbox-density-heatmaps\n\n        Args:\n            data (str | pd.DataFrame): File path or HTTP URL to the input file or a . For example, https://raw.githubusercontent.com/plotly/datasets/master/earthquakes-23k.csv\n            latitude (str, optional): The column name of latitude. Defaults to "latitude".\n            longitude (str, optional): The column name of longitude. Defaults to "longitude".\n            z (str, optional): The column name of z values. Defaults to "value".\n            radius (int, optional): Radius of each “point” of the heatmap. Defaults to 25.\n            colorscale (str, optional): Color scale of the data, e.g., Viridis. See https://plotly.com/python/builtin-colorscales. Defaults to None.\n            name (str, optional): Layer name to use. Defaults to "Heat map".\n\n        '
        if isinstance(data, str):
            df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            raise ValueError('data must be a DataFrame or a file path.')
        heatmap = go.Densitymapbox(lat=df[latitude], lon=df[longitude], z=df[z], radius=radius, colorscale=colorscale, name=name, **kwargs)
        self.add_trace(heatmap)

    def add_heatmap_demo(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Adds a heatmap to the map.'
        quakes = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/earthquakes-23k.csv')
        heatmap = go.Densitymapbox(lat=quakes.Latitude, lon=quakes.Longitude, z=quakes.Magnitude, radius=10, name='Earthquake', **kwargs)
        self.add_basemap('Esri.WorldTopoMap')
        self.add_trace(heatmap)

    def add_gdf(self, gdf, label_col=None, color_col=None, labels=None, opacity=1.0, zoom=None, color_continuous_scale='Viridis', **kwargs):
        if False:
            i = 10
            return i + 15
        'Adds a GeoDataFrame to the map.\n\n        Args:\n            gdf (GeoDataFrame): A GeoDataFrame.\n            label_col (str, optional): The column name of locations. Defaults to None.\n            color_col (str, optional): The column name of color. Defaults to None.\n        '
        check_package('geopandas', 'https://geopandas.org')
        import geopandas as gpd
        if isinstance(gdf, str):
            gdf = gpd.read_file(gdf)
        if not isinstance(gdf, gpd.GeoDataFrame):
            raise ValueError('gdf must be a GeoDataFrame.')
        gdf = gdf.to_crs(epsg=4326)
        (center_lon, center_lat) = gdf_centroid(gdf)
        if isinstance(label_col, str):
            gdf = gdf.set_index(label_col)
            if label_col == color_col:
                gdf[label_col] = gdf.index
            label_col = gdf.index
        elif label_col is None:
            label_col = gdf.index
        if isinstance(color_col, str):
            if color_col not in gdf.columns:
                raise ValueError(f"color must be a column name in the GeoDataFrame. Can be one of {','.join(gdf.columns)} ")
        fig = px.choropleth_mapbox(gdf, geojson=gdf.geometry, locations=label_col, color=color_col, color_continuous_scale=color_continuous_scale, opacity=opacity, labels=labels, **kwargs)
        self.add_traces(fig.data)
        self.set_center(center_lat, center_lon, zoom)

def fix_widget_error():
    if False:
        i = 10
        return i + 15
    "\n    Fix FigureWidget - 'mapbox._derived' Value Error.\n    Adopted from: https://github.com/plotly/plotly.py/issues/2570#issuecomment-738735816\n    "
    import shutil
    basedatatypesPath = os.path.join(os.path.dirname(os.__file__), 'site-packages', 'plotly', 'basedatatypes.py')
    backup_file = basedatatypesPath.replace('.py', '_bk.py')
    shutil.copyfile(basedatatypesPath, backup_file)
    with open(basedatatypesPath, 'r') as f:
        lines = f.read()
    find = 'if not BaseFigure._is_key_path_compatible(key_path_str, self.layout):'
    replace = 'if not BaseFigure._is_key_path_compatible(key_path_str, self.layout):\n                if key_path_str == "mapbox._derived":\n                    return'
    lines = lines.replace(find, replace)
    with open(basedatatypesPath, 'w') as f:
        f.write(lines)