import json

from bokeh.models import GeoJSONDataSource
from bokeh.plotting import figure, show
from bokeh.sampledata.sample_geojson import geojson

data = json.loads(geojson)

for i in range(len(data['features'])):
    data['features'][i]['properties']['Color'] = ['blue', 'red'][i%2]

geo_source = GeoJSONDataSource(geojson=json.dumps(data))

TOOLTIPS = [('Organisation', '@OrganisationName')]

p = figure(background_fill_color="lightgrey", tooltips=TOOLTIPS)

p.scatter(x='x', y='y', size=15, color='Color', alpha=0.7, source=geo_source)

show(p)
