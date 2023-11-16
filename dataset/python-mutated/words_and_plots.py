from bokeh.layouts import layout
from bokeh.models import ColumnDataSource, Div, HoverTool, Paragraph
from bokeh.plotting import figure, show
from bokeh.sampledata.glucose import data
from bokeh.sampledata.iris import flowers

def text():
    if False:
        for i in range(10):
            print('nop')
    return Paragraph(text='\n        Bacon ipsum dolor amet hamburger brisket prosciutto, pork ball tip andouille\n        sausage landjaeger filet mignon ribeye ground round. Jerky fatback cupim\n        landjaeger meatball pork loin corned beef, frankfurter short ribs short loin\n        bresaola capicola chuck kevin. Andouille biltong turkey, tail t-bone ribeye\n        short loin tongue prosciutto kielbasa short ribs boudin. Swine beef ribs\n        tri-tip filet mignon bresaola boudin beef meatball venison leberkas fatback\n        strip steak landjaeger drumstick prosciutto.\n        Bacon ipsum dolor amet hamburger brisket prosciutto, pork ball tip andouille\n        sausage landjaeger filet mignon ribeye ground round. Jerky fatback cupim\n        landjaeger meatball pork loin corned beef, frankfurter short ribs short loin\n        bresaola capicola chuck kevin. Andouille biltong turkey, tail t-bone ribeye\n        short loin tongue prosciutto kielbasa short ribs boudin. Swine beef ribs\n        tri-tip filet mignon bresaola boudin beef meatball venison leberkas fatback\n        strip steak landjaeger drumstick prosciutto.\n        ')

def scatter():
    if False:
        for i in range(10):
            print('nop')
    colormap = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
    source = ColumnDataSource(flowers)
    source.data['colors'] = [colormap[x] for x in flowers['species']]
    s = figure(title='Iris Morphology')
    s.xaxis.axis_label = 'Petal Length'
    s.yaxis.axis_label = 'Petal Width'
    s.scatter('petal_length', 'petal_width', color='colors', source=source, fill_alpha=0.2, size=10, legend_group='species')
    legend = s.legend[0]
    legend.border_line_color = None
    legend.orientation = 'horizontal'
    legend.location = 'center_left'
    s.above.append(legend)
    return s

def hover_plot():
    if False:
        print('Hello World!')
    x = data.loc['2010-10-06'].index.to_series()
    y = data.loc['2010-10-06']['glucose']
    p = figure(width=800, height=400, x_axis_type='datetime', tools='', toolbar_location=None, title='Hover over points')
    p.line(x, y, line_dash='4 4', line_width=1, color='gray')
    cr = p.scatter(x, y, size=20, fill_color='grey', alpha=0.1, line_color=None, hover_fill_color='firebrick', hover_alpha=0.5, hover_line_color=None)
    p.add_tools(HoverTool(tooltips=None, renderers=[cr], mode='hline'))
    return p

def intro():
    if False:
        i = 10
        return i + 15
    return Div(text='\n        <h3>Welcome to Layout!</h3>\n        <p>Hopefully you\'ll see from the code, that the layout tries to get out of your way\n        and do the right thing. Of course, it might not always, so please report bugs as you\n        find them and attach them to the epic we\'re creating <a href="">here</a>.</p>\n        <p>This is an example of <code>scale_width</code> mode (happy to continue the conversations\n        about what to name the modes). In <code>scale_width</code> everything responds to the width\n        that\'s available to it. Plots alter their height to maintain their aspect ratio, and widgets\n        are allowed to grow as tall as they need to accommodate themselves. Often times widgets\n        stay the same height, but text is a good example of a widget that doesn\'t.</p>\n        <h4>I want to stress that this was all written in python. There is no templating or\n        use of <code>bokeh.embed</code>.</h4>\n    ')
show(layout([[intro()], [text(), scatter()], [text()], [hover_plot(), text()]], sizing_mode='scale_width'))