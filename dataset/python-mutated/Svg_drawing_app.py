import remi.gui as gui
from remi import start, App
import math

class SvgPolygon(gui.SvgPolyline):

    def __init__(self, _maxlen=None, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super(SvgPolygon, self).__init__(_maxlen, *args, **kwargs)
        self.type = 'polygon'

    def set_stroke(self, width=1, color='black'):
        if False:
            i = 10
            return i + 15
        'Sets the stroke properties.\n\n        Args:\n            width (int): stroke width\n            color (str): stroke color\n        '
        self.attributes['stroke'] = color
        self.attributes['stroke-width'] = str(width)

    def set_fill(self, color='black'):
        if False:
            for i in range(10):
                print('nop')
        'Sets the fill color.\n\n        Args:\n            color (str): stroke color\n        '
        self.style['fill'] = color
        self.attributes['fill'] = color

    def add_arrow_coord(self, line, arrow_height, arrow_width, recess):
        if False:
            print('Hello World!')
        ' Determine the coordinates of an arrow head polygon\n            with height (h) and width (w) and recess (r)\n            pointing from the one but last to the last point of (poly)line (line).\n            Note that the coordinates of an SvgLine and an SvgPolyline\n            are stored in different variables.\n        '
        if line.type == 'polyline':
            xe = line.coordsX[-1]
            ye = line.coordsY[-1]
            xp = line.coordsX[-2]
            yp = line.coordsY[-2]
        else:
            xe = line.attributes['x2']
            ye = line.attributes['y2']
            xp = line.attributes['x1']
            yp = line.attributes['y1']
        h = arrow_height
        if arrow_width == 0:
            w = arrow_height / 3
        else:
            w = arrow_width
        r = recess
        self.add_coord(xe, ye)
        dx = xe - xp
        dy = ye - yp
        de = math.sqrt(dx ** 2 + dy ** 2)
        xh = xe - h * dx / de
        yh = ye - h * dy / de
        x1 = xh + w * dy / de
        y1 = yh - w * dx / de
        self.add_coord(x1, y1)
        x2 = xe - (h - r) * dx / de
        y2 = ye - (h - r) * dy / de
        self.add_coord(x2, y2)
        x3 = xh - w * dy / de
        y3 = yh + w * dx / de
        self.add_coord(x3, y3)

class MyApp(App):
    """ Example drawing by Andries van Renssen
        including connected rectangular boxes, polylines and rhombusses.
    """

    def __init__(self, *args):
        if False:
            i = 10
            return i + 15
        super(MyApp, self).__init__(*args)

    def main(self):
        if False:
            print('Hello World!')
        self.frame = gui.VBox(width='100%', height='80%', style={'overflow': 'auto', 'background-color': '#eeffdd'})
        self.sheet = gui.Svg(width='100%', height='100%')
        self.screen_width = 1000
        self.screen_height = 600
        self.int_id = 0
        self.sheet.set_viewbox(0, 0, self.screen_width, self.screen_height)
        self.frame.append(self.sheet)
        nr_of_boxes = 2
        box_names = ['Activity-A', 'Activity-B']
        self.Draw_a_drawing_of_one_sheet(nr_of_boxes, box_names)
        return self.frame

    def Draw_a_drawing_of_one_sheet(self, nr_of_boxes, box_names):
        if False:
            return 10
        ' Draw a drawing with two boxes, each with a name inside\n            and a polyline between the midpoints of the sides of the boxes,\n            with half-way the polyline a rhombus with an id included.\n        '
        thickness = 2
        center_x = []
        center_y = []
        mid_points = []
        box_width = 100
        box_height = 100
        delta_x = self.screen_width / (nr_of_boxes + 1)
        delta_y = self.screen_height / (nr_of_boxes + 1)
        for box_nr in range(0, nr_of_boxes):
            center_x.append(delta_x + box_nr * delta_x)
            center_y.append(delta_y + box_nr * delta_y)
            name = box_names[box_nr]
            ident = str(box_nr + 1)
            mid_points.append(self.box_type_1(center_x[box_nr], center_y[box_nr], name, ident, box_width, box_height))
        x2 = mid_points[0][3][0]
        y2 = mid_points[0][3][1]
        x1 = x2 - 150
        y1 = y2
        line_0 = gui.SvgLine(x1, y1, x2, y2)
        line_0.set_stroke(width=thickness, color='black')
        self.sheet.append(line_0)
        head_0 = SvgPolygon(4)
        arrow_height = 20
        arrow_width = arrow_height / 3
        recess = arrow_height / 5
        head_0.add_arrow_coord(line_0, arrow_height, arrow_width, recess)
        head_0.set_stroke(width=thickness, color='black')
        head_0.set_fill(color='blue')
        self.sheet.append(head_0)
        x = (center_x[0] + center_x[1]) / 2
        y = (center_y[0] + center_y[1]) / 2
        self.int_id += 1
        str_id = str(self.int_id)
        hor_size = 15
        vert_size = 25
        rhombus = self.rhombus_polygon(x, y, str_id, hor_size, vert_size)
        line_1_points = []
        line_1_points.append(mid_points[0][2])
        corner = [rhombus[0][0], mid_points[0][2][1]]
        line_1_points.append(corner)
        line_1_points.append(rhombus[0])
        line1 = gui.SvgPolyline(_maxlen=4)
        for pt in line_1_points:
            line1.add_coord(*pt)
        line1.set_stroke(width=thickness, color='black')
        self.sheet.append(line1)
        line_2_points = []
        line_2_points.append(rhombus[1])
        corner = [rhombus[1][0], mid_points[1][3][1]]
        line_2_points.append(corner)
        line_2_points.append(mid_points[1][3])
        line2 = gui.SvgPolyline(_maxlen=4)
        for pt in line_2_points:
            line2.add_coord(pt[0], pt[1])
        line2.set_stroke(width=thickness, color='black')
        self.sheet.append(line2)
        head = SvgPolygon(4)
        head.add_arrow_coord(line2, arrow_height, arrow_width, recess)
        head.set_stroke(width=thickness, color='black')
        head.set_fill(color='blue')
        self.sheet.append(head)

    def box_type_1(self, X, Y, name, ident, box_width, box_height):
        if False:
            i = 10
            return i + 15
        ' Draw a rectangular box of box_width and box_height\n            with name and ident,\n            on sheet with (X,Y) as its center on the canvas\n            Return midpts = N(x,y), S(x,y), E(x,y), W(x,y).\n        '
        boxW2 = box_width / 2
        boxH2 = box_height / 2
        (x0, y0) = (X - boxW2, Y - boxH2)
        (x1, y1) = (X + boxW2, Y + boxH2)
        width = x1 - x0
        height = y1 - y0
        box = gui.SvgRectangle(x0, y0, width, height)
        box.set_stroke(width=2, color='black')
        box.set_fill(color='yellow')
        box_name = gui.SvgText(X, Y, name)
        box_name.attributes['text-anchor'] = 'middle'
        box_id = gui.SvgText(X, Y + 15, str(ident))
        box_id.attributes['text-anchor'] = 'middle'
        self.sheet.append([box, box_name, box_id])
        mid_north = [X, Y - boxH2]
        mid_south = [X, Y + boxH2]
        mid_east = [X + boxW2, Y]
        mid_west = [X - boxW2, Y]
        return (mid_north, mid_south, mid_east, mid_west)

    def rhombus_polygon(self, X, Y, str_id, hor_size, vert_size):
        if False:
            for i in range(10):
                print('nop')
        ' Draw a rhombus polygon.\n            Horizontal size (-hor_size, +hor_size) and\n            vertical size (-vert_size, +vert_size).\n            with its center on position X,Y\n            and with its str_id as text in the middle.\n        '
        (x0, y0) = (X - hor_size, Y)
        (x1, y1) = (X, Y - vert_size)
        (x2, y2) = (X + hor_size, Y)
        (x3, y3) = (X, Y + vert_size)
        polygon = SvgPolygon(4)
        polygon.set_stroke(width=2, color='black')
        poly_name = gui.SvgText(X, Y + 5, str_id)
        poly_name.attributes['text-anchor'] = 'middle'
        self.sheet.append([polygon, poly_name])
        mid_north = [x1, y1]
        mid_south = [x3, y3]
        mid_east = [x2, y2]
        mid_west = [x0, y0]
        polygon.add_coord(*mid_north)
        polygon.add_coord(*mid_east)
        polygon.add_coord(*mid_south)
        polygon.add_coord(*mid_west)
        return (mid_north, mid_south, mid_east, mid_west)
if __name__ == '__main__':
    start(MyApp, address='127.0.0.1', port=8081, multiple_instance=False, enable_file_cache=True, update_interval=0.1, start_browser=True)