import time
import board
import displayio
import vectorio

def increment_color(shape):
    if False:
        print('Hello World!')
    if shape.color_index + 1 < len(shape.pixel_shader):
        shape.color_index += 1
    else:
        shape.color_index = 0
display = board.DISPLAY
main_group = displayio.Group()
palette = displayio.Palette(4)
palette[0] = 1201808
palette[1] = 3455888
palette[2] = 11145760
palette[3] = 11142330
circle = vectorio.Circle(pixel_shader=palette, radius=25, x=25, y=25)
main_group.append(circle)
rectangle = vectorio.Rectangle(pixel_shader=palette, width=50, height=50, x=25, y=75)
main_group.append(rectangle)
points = [(5, 5), (70, 20), (35, 35), (20, 70)]
polygon = vectorio.Polygon(pixel_shader=palette, points=points, x=145, y=55)
main_group.append(polygon)
display.show(main_group)
while True:
    for x in range(25, display.width - 25):
        circle.x = x
        time.sleep(0.01)
    increment_color(circle)
    increment_color(rectangle)
    increment_color(polygon)
    for x in range(display.width - 25, 25, -1):
        circle.x = x
        time.sleep(0.01)
    increment_color(circle)
    increment_color(rectangle)
    increment_color(polygon)