from .node import Node

class Vector(Node):

    def __init__(self, node):
        if False:
            i = 10
            return i + 15
        super().__init__(node)

    def color(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Returns HEX form of element RGB color (str)\n        '
        try:
            color = self.node['fills'][0]['color']
            (r, g, b, *_) = [int(color.get(i, 0) * 255) for i in 'rgba']
            return f'#{r:02X}{g:02X}{b:02X}'
        except Exception:
            return '#FFFFFF'

    def size(self):
        if False:
            for i in range(10):
                print('nop')
        bbox = self.node['absoluteBoundingBox']
        width = bbox['width']
        height = bbox['height']
        return (width, height)

    def position(self, frame):
        if False:
            return 10
        bbox = self.node['absoluteBoundingBox']
        x = bbox['x']
        y = bbox['y']
        frame_bbox = frame.node['absoluteBoundingBox']
        frame_x = frame_bbox['x']
        frame_y = frame_bbox['y']
        x = abs(x - frame_x)
        y = abs(y - frame_y)
        return (x, y)

class Star(Vector):

    def __init__(self, node):
        if False:
            i = 10
            return i + 15
        super().__init__(node)

class Ellipse(Vector):

    def __init__(self, node):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(node)

class RegularPolygon(Vector):

    def __init__(self, node):
        if False:
            i = 10
            return i + 15
        super().__init__(node)

class Rectangle(Vector):

    def __init__(self, node, frame):
        if False:
            while True:
                i = 10
        super().__init__(node)
        (self.x, self.y) = self.position(frame)
        (self.width, self.height) = self.size()
        self.fill_color = self.color()

    @property
    def corner_radius(self):
        if False:
            print('Hello World!')
        return self.node.get('cornerRadius')

    @property
    def rectangle_corner_radii(self):
        if False:
            i = 10
            return i + 15
        return self.node.get('rectangleCornerRadii')

    def to_code(self):
        if False:
            i = 10
            return i + 15
        return f'\ncanvas.create_rectangle(\n    {self.x},\n    {self.y},\n    {self.x + self.width},\n    {self.y + self.height},\n    fill="{self.fill_color}",\n    outline="")\n'

class Line(Rectangle):

    def __init__(self, node, frame):
        if False:
            i = 10
            return i + 15
        super().__init__(node, frame)

    def color(self) -> str:
        if False:
            print('Hello World!')
        'Returns HEX form of element RGB color (str)\n        '
        try:
            color = self.node['strokes'][0]['color']
            (r, g, b, *_) = [int(color.get(i, 0) * 255) for i in 'rgba']
            return f'#{r:02X}{g:02X}{b:02X}'
        except Exception:
            return '#FFFFFF'

    def size(self):
        if False:
            while True:
                i = 10
        (width, height) = super().size()
        return (width + self.node['strokeWeight'], height + self.node['strokeWeight'])

    def position(self, frame):
        if False:
            i = 10
            return i + 15
        (x, y) = super().position(frame)
        return (x - self.node['strokeWeight'], y - self.node['strokeWeight'])

class UnknownElement(Vector):

    def __init__(self, node, frame):
        if False:
            print('Hello World!')
        super().__init__(node)
        (self.x, self.y) = self.position(frame)
        (self.width, self.height) = self.size()

    def to_code(self):
        if False:
            i = 10
            return i + 15
        return f'\ncanvas.create_rectangle(\n    {self.x},\n    {self.y},\n    {self.x + self.width},\n    {self.y + self.height},\n    fill="#000000",\n    outline="")\n'