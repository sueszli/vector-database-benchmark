""" flask_example.py

    Required packages:
    - flask
    - folium

    Usage:

    Start the flask server by running:

        $ python flask_example.py

    And then head to http://127.0.0.1:5000/ in your browser to see the map displayed

"""
from flask import Flask, render_template_string
import folium
app = Flask(__name__)

@app.route('/')
def fullscreen():
    if False:
        for i in range(10):
            print('nop')
    'Simple example of a fullscreen map.'
    m = folium.Map()
    return m.get_root().render()

@app.route('/iframe')
def iframe():
    if False:
        i = 10
        return i + 15
    'Embed a map as an iframe on a page.'
    m = folium.Map()
    m.get_root().width = '800px'
    m.get_root().height = '600px'
    iframe = m.get_root()._repr_html_()
    return render_template_string('\n            <!DOCTYPE html>\n            <html>\n                <head></head>\n                <body>\n                    <h1>Using an iframe</h1>\n                    {{ iframe|safe }}\n                </body>\n            </html>\n        ', iframe=iframe)

@app.route('/components')
def components():
    if False:
        while True:
            i = 10
    'Extract map components and put those on a page.'
    m = folium.Map(width=800, height=600)
    m.get_root().render()
    header = m.get_root().header.render()
    body_html = m.get_root().html.render()
    script = m.get_root().script.render()
    return render_template_string('\n            <!DOCTYPE html>\n            <html>\n                <head>\n                    {{ header|safe }}\n                </head>\n                <body>\n                    <h1>Using components</h1>\n                    {{ body_html|safe }}\n                    <script>\n                        {{ script|safe }}\n                    </script>\n                </body>\n            </html>\n        ', header=header, body_html=body_html, script=script)
if __name__ == '__main__':
    app.run(debug=True)