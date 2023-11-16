""""
Test Vector Layers
------------------

"""
import json
from folium import Map
from folium.utilities import get_bounds, normalize
from folium.vector_layers import Circle, CircleMarker, Polygon, PolyLine, Rectangle, path_options

def test_circle():
    if False:
        i = 10
        return i + 15
    m = Map()
    radius = 10000
    popup = f'I am {radius} meters'
    location = [-27.551667, -48.478889]
    circle = Circle(location=location, radius=radius, color='black', weight=2, fill_opacity=0.6, opacity=1, fill=True, popup=popup)
    circle.add_to(m)
    expected_options = {'bubblingMouseEvents': True, 'color': 'black', 'dashArray': None, 'dashOffset': None, 'fill': True, 'fillColor': 'black', 'fillOpacity': 0.6, 'fillRule': 'evenodd', 'lineCap': 'round', 'lineJoin': 'round', 'opacity': 1, 'radius': radius, 'stroke': True, 'weight': 2}
    m._repr_html_()
    expected_rendered = f'\n    var {circle.get_name()} = L.circle(\n    {location},\n    {{\n    "bubblingMouseEvents": true,\n    "color": "black",\n    "dashArray": null,\n    "dashOffset": null,\n    "fill": true,\n    "fillColor": "black",\n    "fillOpacity": 0.6,\n    "fillRule": "evenodd",\n    "lineCap": "round",\n    "lineJoin": "round",\n    "opacity": 1,\n    "radius": {radius},\n    "stroke": true,\n    "weight": 2\n    }}\n    )\n    .addTo({m.get_name()});\n    '
    rendered = circle._template.module.script(circle)
    assert normalize(rendered) == normalize(expected_rendered)
    assert circle.get_bounds() == [location, location]
    assert json.dumps(circle.to_dict()) == circle.to_json()
    assert circle.location == [-27.551667, -48.478889]
    assert circle.options == expected_options

def test_circle_marker():
    if False:
        print('Hello World!')
    m = Map()
    radius = 50
    popup = f'I am {radius} pixels'
    location = [-27.55, -48.8]
    circle_marker = CircleMarker(location=location, radius=radius, color='black', weight=2, fill_opacity=0.6, opacity=1, fill=True, popup=popup)
    circle_marker.add_to(m)
    options = {'bubblingMouseEvents': True, 'color': 'black', 'dashArray': None, 'dashOffset': None, 'fill': True, 'fillColor': 'black', 'fillOpacity': 0.6, 'fillRule': 'evenodd', 'lineCap': 'round', 'lineJoin': 'round', 'opacity': 1, 'radius': radius, 'stroke': True, 'weight': 2}
    m._repr_html_()
    expected_bounds = [location, location]
    expected_rendered = f'\n    var {circle_marker.get_name()} = L.circleMarker(\n    {location},\n    {{\n    "bubblingMouseEvents": true,\n    "color": "black",\n    "dashArray": null,\n    "dashOffset": null,\n    "fill": true,\n    "fillColor": "black",\n    "fillOpacity": 0.6,\n    "fillRule": "evenodd",\n    "lineCap": "round",\n    "lineJoin": "round",\n    "opacity": 1,\n    "radius": {radius},\n    "stroke": true,\n    "weight": 2\n    }}\n    )\n    .addTo({m.get_name()});\n    '
    rendered = circle_marker._template.module.script(circle_marker)
    assert normalize(rendered) == normalize(expected_rendered)
    assert circle_marker.get_bounds() == expected_bounds
    assert json.dumps(circle_marker.to_dict()) == circle_marker.to_json()
    assert circle_marker.location == location
    assert circle_marker.options == options

def test_rectangle():
    if False:
        for i in range(10):
            print('nop')
    m = Map()
    location = [[45.6, -122.8], [45.61, -122.7]]
    rectangle = Rectangle(bounds=location, popup='I am a rectangle', color='black', weight=2, fill_opacity=0.6, opacity=1, fill=True)
    rectangle.add_to(m)
    expected_options = {'bubblingMouseEvents': True, 'color': 'black', 'dashArray': None, 'dashOffset': None, 'fill': True, 'fillColor': 'black', 'fillOpacity': 0.6, 'fillRule': 'evenodd', 'lineCap': 'round', 'lineJoin': 'round', 'noClip': False, 'opacity': 1, 'smoothFactor': 1.0, 'stroke': True, 'weight': 2}
    m._repr_html_()
    expected_rendered = f'\n    var {rectangle.get_name()} = L.rectangle(\n    {location},\n    {{\n    "bubblingMouseEvents": true,\n    "color": "black",\n    "dashArray": null,\n    "dashOffset": null,\n    "fill": true,\n    "fillColor": "black",\n    "fillOpacity": 0.6,\n    "fillRule": "evenodd",\n    "lineCap": "round",\n    "lineJoin": "round",\n    "noClip": false,\n    "opacity": 1,\n    "smoothFactor": 1.0,\n    "stroke": true,\n    "weight": 2\n    }}\n    )\n    .addTo({m.get_name()});\n    '
    rendered = rectangle._template.module.script(rectangle)
    assert normalize(rendered) == normalize(expected_rendered)
    assert rectangle.get_bounds() == location
    assert json.dumps(rectangle.to_dict()) == rectangle.to_json()
    assert rectangle.options == expected_options

def test_polygon_marker():
    if False:
        for i in range(10):
            print('nop')
    m = Map()
    locations = [[35.6636, 139.7634], [35.6629, 139.7664], [35.6663, 139.7706], [35.6725, 139.7632], [35.6728, 139.7627], [35.672, 139.7606], [35.6682, 139.7588], [35.6663, 139.7627]]
    polygon = Polygon(locations=locations, popup='I am a polygon')
    polygon.add_to(m)
    expected_options = {'bubblingMouseEvents': True, 'color': '#3388ff', 'dashArray': None, 'dashOffset': None, 'fill': False, 'fillColor': '#3388ff', 'fillOpacity': 0.2, 'fillRule': 'evenodd', 'lineCap': 'round', 'lineJoin': 'round', 'noClip': False, 'opacity': 1.0, 'smoothFactor': 1.0, 'stroke': True, 'weight': 3}
    m._repr_html_()
    expected_rendered = f'\n    var {polygon.get_name()} = L.polygon(\n    {locations},\n    {{\n    "bubblingMouseEvents": true,\n    "color": "#3388ff",\n    "dashArray": null,\n    "dashOffset": null,\n    "fill": false,\n    "fillColor": "#3388ff",\n    "fillOpacity": 0.2,\n    "fillRule": "evenodd",\n    "lineCap": "round",\n    "lineJoin": "round",\n    "noClip": false,\n    "opacity": 1.0,\n    "smoothFactor": 1.0,\n    "stroke": true,\n    "weight": 3\n    }}\n    )\n    .addTo({m.get_name()});\n    '
    rendered = polygon._template.module.script(polygon)
    assert normalize(rendered) == normalize(expected_rendered)
    assert polygon.get_bounds() == get_bounds(locations)
    assert json.dumps(polygon.to_dict()) == polygon.to_json()
    assert polygon.options == expected_options

def test_polyline():
    if False:
        i = 10
        return i + 15
    m = Map()
    locations = [[40.0, -80.0], [45.0, -80.0]]
    polyline = PolyLine(locations=locations, popup='I am PolyLine')
    polyline.add_to(m)
    expected_options = {'smoothFactor': 1.0, 'noClip': False, 'bubblingMouseEvents': True, 'color': '#3388ff', 'dashArray': None, 'dashOffset': None, 'fill': False, 'fillColor': '#3388ff', 'fillOpacity': 0.2, 'fillRule': 'evenodd', 'lineCap': 'round', 'lineJoin': 'round', 'opacity': 1.0, 'stroke': True, 'weight': 3}
    m._repr_html_()
    expected_rendered = f'\n    var {polyline.get_name()} = L.polyline(\n    {locations},\n    {{\n    "bubblingMouseEvents": true,\n    "color": "#3388ff",\n    "dashArray": null,\n    "dashOffset": null,\n    "fill": false,\n    "fillColor": "#3388ff",\n    "fillOpacity": 0.2,\n    "fillRule": "evenodd",\n    "lineCap": "round",\n    "lineJoin": "round",\n    "noClip": false,\n    "opacity": 1.0,\n    "smoothFactor": 1.0,\n    "stroke": true,\n    "weight": 3\n    }}\n    )\n    .addTo({m.get_name()});\n    '
    rendered = polyline._template.module.script(polyline)
    assert normalize(rendered) == normalize(expected_rendered)
    assert polyline.get_bounds() == get_bounds(locations)
    assert json.dumps(polyline.to_dict()) == polyline.to_json()
    assert polyline.options == expected_options

def test_mulyipolyline():
    if False:
        while True:
            i = 10
    m = Map()
    locations = [[[45.51, -122.68], [37.77, -122.43], [34.04, -118.2]], [[40.78, -73.91], [41.83, -87.62], [32.76, -96.72]]]
    multipolyline = PolyLine(locations=locations, popup='MultiPolyLine')
    multipolyline.add_to(m)
    expected_options = {'smoothFactor': 1.0, 'noClip': False, 'bubblingMouseEvents': True, 'color': '#3388ff', 'dashArray': None, 'dashOffset': None, 'fill': False, 'fillColor': '#3388ff', 'fillOpacity': 0.2, 'fillRule': 'evenodd', 'lineCap': 'round', 'lineJoin': 'round', 'opacity': 1.0, 'stroke': True, 'weight': 3}
    m._repr_html_()
    expected_rendered = f'\n    var {multipolyline.get_name()} = L.polyline(\n    {locations},\n    {{\n    "bubblingMouseEvents": true,\n    "color": "#3388ff",\n    "dashArray": null,\n    "dashOffset": null,\n    "fill": false,\n    "fillColor": "#3388ff",\n    "fillOpacity": 0.2,\n    "fillRule": "evenodd",\n    "lineCap": "round",\n    "lineJoin": "round",\n    "noClip": false,\n    "opacity": 1.0,\n    "smoothFactor": 1.0,\n    "stroke": true,\n    "weight": 3\n    }}\n    )\n    .addTo({m.get_name()});\n    '
    rendered = multipolyline._template.module.script(multipolyline)
    assert normalize(rendered) == normalize(expected_rendered)
    assert multipolyline.get_bounds() == get_bounds(locations)
    assert json.dumps(multipolyline.to_dict()) == multipolyline.to_json()
    assert multipolyline.options == expected_options

def test_path_options_lower_camel_case():
    if False:
        return 10
    options = path_options(fill_color='red', fillOpacity=0.3)
    assert options['fillColor'] == 'red'
    assert options['fillOpacity'] == 0.3