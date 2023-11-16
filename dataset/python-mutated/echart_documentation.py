from nicegui import ui
from ..documentation_tools import text_demo

def main_demo() -> None:
    if False:
        print('Hello World!')
    from random import random
    echart = ui.echart({'xAxis': {'type': 'value'}, 'yAxis': {'type': 'category', 'data': ['A', 'B'], 'inverse': True}, 'legend': {'textStyle': {'color': 'gray'}}, 'series': [{'type': 'bar', 'name': 'Alpha', 'data': [0.1, 0.2]}, {'type': 'bar', 'name': 'Beta', 'data': [0.3, 0.4]}]})

    def update():
        if False:
            i = 10
            return i + 15
        echart.options['series'][0]['data'][0] = random()
        echart.update()
    ui.button('Update', on_click=update)

def more() -> None:
    if False:
        for i in range(10):
            print('nop')

    @text_demo('EChart with clickable points', '\n        You can register a callback for an event when a series point is clicked.\n    ')
    def clickable_points() -> None:
        if False:
            print('Hello World!')
        ui.echart({'xAxis': {'type': 'category'}, 'yAxis': {'type': 'value'}, 'series': [{'type': 'line', 'data': [20, 10, 30, 50, 40, 30]}]}, on_point_click=ui.notify)

    @text_demo('EChart with dynamic properties', '\n        Dynamic properties can be passed to chart elements to customize them such as apply an axis label format.\n        To make a property dynamic, prefix a colon ":" to the property name.\n    ')
    def dynamic_properties() -> None:
        if False:
            return 10
        ui.echart({'xAxis': {'type': 'category'}, 'yAxis': {'axisLabel': {':formatter': 'value => "$" + value'}}, 'series': [{'type': 'line', 'data': [5, 8, 13, 21, 34, 55]}]})