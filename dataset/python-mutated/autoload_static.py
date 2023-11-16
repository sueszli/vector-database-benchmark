from jinja2 import Template
from tornado.ioloop import IOLoop
from tornado.web import Application, RequestHandler
from bokeh.embed import autoload_static
from bokeh.plotting import figure
from bokeh.resources import CDN
from bokeh.sampledata.iris import flowers
from bokeh.util.browser import view
template = Template('\n<!doctype html>\n\n<html lang="en">\n<head>\n  <meta charset="utf-8">\n</head>\n\n<body>\n  <div>\n    The plot embedded below is a standalone plot that was embedded using\n    <fixed>autoload_static</fixed>. For more information see the\n    <a target="_blank" href="https://docs.bokeh.org/en/latest/docs/user_guide/embed.html#autoload-scripts">\n    documentation</a>.\n  </div>\n  {{ script|safe }}\n</body>\n</html>\n')

class IndexHandler(RequestHandler):

    def initialize(self, script):
        if False:
            i = 10
            return i + 15
        self.script = script

    def get(self):
        if False:
            while True:
                i = 10
        self.write(template.render(script=self.script))

class JSHandler(RequestHandler):

    def initialize(self, js):
        if False:
            for i in range(10):
                print('nop')
        self.js = js

    def get(self):
        if False:
            i = 10
            return i + 15
        self.write(self.js)

def make_plot():
    if False:
        print('Hello World!')
    colormap = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
    colors = [colormap[x] for x in flowers['species']]
    p = figure(title='Iris Morphology')
    p.xaxis.axis_label = 'Petal Length'
    p.yaxis.axis_label = 'Petal Width'
    p.circle(flowers['petal_length'], flowers['petal_width'], color=colors, fill_alpha=0.2, size=10)
    return p
if __name__ == '__main__':
    print('Opening Tornado app with embedded Bokeh plot on http://localhost:8080/')
    (js, script) = autoload_static(make_plot(), CDN, 'embed.js')
    app = Application([('/', IndexHandler, dict(script=script)), ('/embed.js', JSHandler, dict(js=js))])
    app.listen(8080)
    io_loop = IOLoop.current()
    io_loop.add_callback(view, 'http://localhost:8080/')
    io_loop.start()