from nicegui import ui
from ..documentation_tools import text_demo

def main_demo() -> None:
    if False:
        i = 10
        return i + 15
    from random import random
    chart = ui.highchart({'title': False, 'chart': {'type': 'bar'}, 'xAxis': {'categories': ['A', 'B']}, 'series': [{'name': 'Alpha', 'data': [0.1, 0.2]}, {'name': 'Beta', 'data': [0.3, 0.4]}]}).classes('w-full h-64')

    def update():
        if False:
            for i in range(10):
                print('nop')
        chart.options['series'][0]['data'][0] = random()
        chart.update()
    ui.button('Update', on_click=update)

def more() -> None:
    if False:
        print('Hello World!')

    @text_demo('Chart with extra dependencies', '\n        To use a chart type that is not included in the default dependencies, you can specify extra dependencies.\n        This demo shows a solid gauge chart.\n    ')
    def extra_dependencies() -> None:
        if False:
            return 10
        ui.highchart({'title': False, 'chart': {'type': 'solidgauge'}, 'yAxis': {'min': 0, 'max': 1}, 'series': [{'data': [0.42]}]}, extras=['solid-gauge']).classes('w-full h-64')

    @text_demo('Chart with draggable points', '\n        This chart allows dragging the series points.\n        You can register callbacks for the following events:\n        \n        - `on_point_click`: called when a point is clicked\n        - `on_point_drag_start`: called when a point drag starts\n        - `on_point_drag`: called when a point is dragged\n        - `on_point_drop`: called when a point is dropped\n    ')
    def drag() -> None:
        if False:
            i = 10
            return i + 15
        ui.highchart({'title': False, 'plotOptions': {'series': {'stickyTracking': False, 'dragDrop': {'draggableY': True, 'dragPrecisionY': 1}}}, 'series': [{'name': 'A', 'data': [[20, 10], [30, 20], [40, 30]]}, {'name': 'B', 'data': [[50, 40], [60, 50], [70, 60]]}]}, extras=['draggable-points'], on_point_click=lambda e: ui.notify(f'Click: {e}'), on_point_drag_start=lambda e: ui.notify(f'Drag start: {e}'), on_point_drop=lambda e: ui.notify(f'Drop: {e}')).classes('w-full h-64')