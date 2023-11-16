import json
from flask import Flask
from jinja2 import Template
from bokeh.embed import json_item
from bokeh.plotting import figure
from bokeh.resources import CDN
from bokeh.sampledata.iris import flowers
app = Flask(__name__)
page = Template('\n<!DOCTYPE html>\n<html lang="en">\n<head>\n  {{ resources }}\n</head>\n\n<body>\n  <div id="myplot"></div>\n  <div id="myplot2"></div>\n  <script>\n  fetch(\'/plot\')\n    .then(function(response) { return response.json(); })\n    .then(function(item) { return Bokeh.embed.embed_item(item); })\n  </script>\n  <script>\n  fetch(\'/plot2\')\n    .then(function(response) { return response.json(); })\n    .then(function(item) { return Bokeh.embed.embed_item(item, "myplot2"); })\n  </script>\n</body>\n')
colormap = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
colors = [colormap[x] for x in flowers['species']]

def make_plot(x, y):
    if False:
        return 10
    p = figure(title='Iris Morphology', sizing_mode='fixed', width=400, height=400)
    p.xaxis.axis_label = x
    p.yaxis.axis_label = y
    p.circle(flowers[x], flowers[y], color=colors, fill_alpha=0.2, size=10)
    return p

@app.route('/')
def root():
    if False:
        i = 10
        return i + 15
    return page.render(resources=CDN.render())

@app.route('/plot')
def plot():
    if False:
        return 10
    p = make_plot('petal_width', 'petal_length')
    return json.dumps(json_item(p, 'myplot'))

@app.route('/plot2')
def plot2():
    if False:
        i = 10
        return i + 15
    p = make_plot('sepal_width', 'sepal_length')
    return json.dumps(json_item(p))
if __name__ == '__main__':
    app.run()